"""Phase-5 pipeline runner: load → search → evaluate → interpret → track.

One command produces, per symbol, the honest deflated verdict and logs it to the
local MLflow store::

    uv run python -m options_system.models.run --symbols MES MNQ

It assembles the fast leak-free matrix, runs the in-CV regularization search,
evaluates the selected config through CPCV/PBO/DSR (skill **over beta**), explains
it with SHAP, persists a JSON summary under ``data/models/runs/`` and logs the run
to MLflow. Nothing here trades, backtests economically, or promotes a model — it
only answers *is there a real, deflated edge?*
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from config.settings import Settings

from ..common.logging import get_logger
from ..validation.config import ValidationConfig
from .config import ModelConfig
from .dataset import load_training_matrix
from .evaluate_model import _runs_dir, evaluate_model, save_model_run
from .interpret import explain
from .lgbm import build_estimator
from .tracking import log_run
from .tune import run_search

logger = get_logger(__name__)


def run_symbol(
    symbol: str,
    mcfg: ModelConfig | None = None,
    vcfg: ValidationConfig | None = None,
    *,
    with_macro: bool = True,
    log_mlflow: bool = True,
    save: bool = True,
    interpret: bool = True,
) -> dict[str, Any]:
    """Run the full evaluation for one symbol and return the JSON-able summary.

    ``with_macro`` chooses the input set: price+macro (default, Phase-6) or the
    price-only Phase-5 baseline. The model spec, CV machinery and verdict gates are
    identical either way — only the feature matrix changes (a clean controlled
    experiment). Price-only runs are saved under a ``_price_only`` suffix.
    """
    mcfg = mcfg or ModelConfig.load()
    vcfg = vcfg or ValidationConfig.load()
    suffix = "" if with_macro else "_price_only"
    tag = "price+macro" if with_macro else "price-only"

    logger.info(f"[{symbol}] assembling matrix ({tag}, timeout={mcfg.target.timeout_handling})")
    tm = load_training_matrix(
        symbol, timeout_handling=mcfg.target.timeout_handling, with_macro=with_macro
    )
    logger.info(
        f"[{symbol}] n={tm.n} eff_n={tm.uniqueness.sum():.0f} "
        f"feat={len(tm.feature_cols)} (+{len(tm.macro_cols)} macro) — "
        f"searching ({mcfg.search.n_trials} trials)"
    )
    search = run_search(tm, mcfg, vcfg)
    logger.info(
        f"[{symbol}] selected {search.selected_overrides} — evaluating through CPCV/PBO/DSR"
    )
    summary = evaluate_model(tm, search, mcfg, vcfg)

    estimator = None
    if interpret or log_mlflow:
        estimator = build_estimator(mcfg, search.selected_overrides)
        estimator.fit(tm.X, tm.y_dir, sample_weight=tm.weight)  # full-data fit: SHAP + artifact

    if interpret:
        png = _runs_dir() / f"{symbol}{suffix}_shap.png"
        summary["shap"] = explain(
            tm, mcfg, search.selected_overrides, estimator=estimator, out_png=png
        )

    run_id = None
    if log_mlflow:
        png = Path(summary.get("shap", {}).get("summary_plot") or "")
        run_id = log_run(
            summary,
            summary.get("shap"),
            estimator=estimator,
            shap_png=png if png.name else None,
        )
    summary["mlflow_run_id"] = run_id

    if save:
        save_model_run(summary, suffix=suffix)
    return summary


def _gate_row(summary: dict[str, Any]) -> dict[str, Any]:
    """Flatten the headline gate metrics + verdict from a run summary (for comparison)."""
    p = summary["pooled_kfold"]
    return {
        "verdict": summary["verdict"],
        "n_features": summary["n_features"],
        "directional_accuracy": p["directional_accuracy"],
        "excess_sharpe": p["excess_sharpe"],
        "long_benchmark_sharpe": p["long_benchmark_sharpe"],
        "excess_dsr": p["excess_dsr"],
        "pbo": summary["pbo"]["pbo"] if summary["pbo"] else None,
        "cpcv_excess_sharpe_mean": summary["cpcv"]["excess_sharpe"]["mean"],
    }


def compare_symbol(
    symbol: str,
    mcfg: ModelConfig | None = None,
    vcfg: ValidationConfig | None = None,
    *,
    log_mlflow: bool = True,
    interpret: bool = True,
) -> dict[str, Any]:
    """Run BOTH price-only and price+macro for ``symbol`` and return a side-by-side.

    The two runs are identical except for the feature matrix (same labels, same CV,
    same verdict gates), so the comparison isolates whether macro features add a
    real, deflated, beta-beating edge over the Phase-5 price-only baseline. The
    comparison dict is saved to ``data/models/runs/<symbol>_comparison.json``.
    """
    mcfg = mcfg or ModelConfig.load()
    vcfg = vcfg or ValidationConfig.load()
    price = run_symbol(
        symbol, mcfg, vcfg, with_macro=False, log_mlflow=log_mlflow, interpret=interpret
    )
    macro = run_symbol(
        symbol, mcfg, vcfg, with_macro=True, log_mlflow=log_mlflow, interpret=interpret
    )
    comparison = {
        "symbol": symbol,
        "model_version": mcfg.model_version,
        "feature_version": macro["feature_version"],
        "label_version": macro["label_version"],
        "macro_feature_version": macro["macro_feature_version"],
        "price_only": _gate_row(price),
        "price_plus_macro": _gate_row(macro),
        "verdict_changed": price["verdict"] != macro["verdict"],
    }
    path = _runs_dir() / f"{symbol}_comparison.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(comparison, fh, indent=2, default=str)
    return comparison


def _print_comparison(c: dict[str, Any]) -> None:
    po, pm = c["price_only"], c["price_plus_macro"]
    logger.info(f"[{c['symbol']}] PRICE-ONLY vs PRICE+MACRO (mfv={c['macro_feature_version']})")
    logger.info(
        f"  price-only : verdict={po['verdict']!r} acc={po['directional_accuracy']} "
        f"excessSR={po['excess_sharpe']} (long={po['long_benchmark_sharpe']}) "
        f"excessDSR={po['excess_dsr']} PBO={po['pbo']} CPCVexSR={po['cpcv_excess_sharpe_mean']}"
    )
    logger.info(
        f"  price+macro: verdict={pm['verdict']!r} acc={pm['directional_accuracy']} "
        f"excessSR={pm['excess_sharpe']} (long={pm['long_benchmark_sharpe']}) "
        f"excessDSR={pm['excess_dsr']} PBO={pm['pbo']} CPCVexSR={pm['cpcv_excess_sharpe_mean']} "
        f"[{pm['n_features']} feat]"
    )
    logger.info(f"  verdict changed: {c['verdict_changed']}")


def _print_verdict(summary: dict[str, Any]) -> None:
    p = summary["pooled_kfold"]
    pbo = summary["pbo"]["pbo"] if summary["pbo"] else "n/a"
    cx = summary["cpcv"]["excess_sharpe"]
    min_acc = summary["verdict_thresholds"]["min_directional_accuracy"]
    logger.info(
        f"[{summary['symbol']}] VERDICT: {summary['verdict'].upper()}  "
        f"(dir_acc={p['directional_accuracy']} vs {min_acc}, "
        f"PBO={pbo}, excess_DSR={p['excess_dsr']}, "
        f"excess_sharpe={p['excess_sharpe']} vs long={p['long_benchmark_sharpe']})"
    )
    logger.info(
        f"[{summary['symbol']}] CPCV excess-Sharpe paths: mean={cx['mean']} "
        f"median={cx['median']} min={cx['min']} max={cx['max']}"
    )
    if summary.get("shap"):
        logger.info(f"[{summary['symbol']}] SHAP top: {summary['shap']['top_features'][:6]}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="models.run", description=__doc__)
    p.add_argument("--symbols", nargs="+", default=None, help="default: settings.record_symbols")
    p.add_argument("--no-mlflow", action="store_true", help="skip MLflow logging")
    p.add_argument("--no-interpret", action="store_true", help="skip SHAP")
    p.add_argument(
        "--no-macro",
        action="store_true",
        help="use the price-only Phase-5 feature set (skip macro features)",
    )
    p.add_argument(
        "--compare",
        action="store_true",
        help="run BOTH price-only and price+macro and print the side-by-side verdict comparison",
    )
    args = p.parse_args(argv)

    symbols = args.symbols or Settings().record_symbols
    mcfg, vcfg = ModelConfig.load(), ValidationConfig.load()
    logger.info(f"model_version={mcfg.model_version} symbols={symbols} compare={args.compare}")
    for symbol in symbols:
        if args.compare:
            c = compare_symbol(
                symbol, mcfg, vcfg, log_mlflow=not args.no_mlflow, interpret=not args.no_interpret
            )
            _print_comparison(c)
        else:
            summary = run_symbol(
                symbol,
                mcfg,
                vcfg,
                with_macro=not args.no_macro,
                log_mlflow=not args.no_mlflow,
                interpret=not args.no_interpret,
            )
            _print_verdict(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
