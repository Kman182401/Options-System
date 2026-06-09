"""Phase-5 pipeline runner: load → search → evaluate → interpret → track.

One command produces, per symbol, the honest deflated verdict and logs it to the
local MLflow store::

    uv run python -m options_system.models.run --symbols MES MNQ

It assembles the fast leak-free matrix, runs the in-CV regularization search,
evaluates the selected config through CPCV/PBO/DSR (skill **over beta**), explains
it with SHAP, persists a JSON summary under ``data/models/runs/`` and logs the run
to MLflow. Nothing here trades, backtests economically, or promotes a model — it
only answers *is there a real, deflated edge?*

Input sets are selectable as clean controlled experiments (same labels, same CV,
same verdict gates — only the feature matrix changes):

* default — price + macro (Phase 6 canonical), saved as ``<symbol>.json``.
* ``--no-macro`` — price-only Phase-5 baseline, ``<symbol>_price_only.json``.
* ``--with-ta`` — price + macro + the opt-in v2 TA layer, ``<symbol>_macro_ta.json``.
* ``--compare`` — price-only vs price+macro side-by-side.
* ``--compare-ta`` — price+macro (baseline) vs price+macro+TA (candidate),
  ``<symbol>_ta_comparison.json``. The canonical ``<symbol>.json`` is never
  overwritten by the TA experiment.
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
from .evaluate_model import VERDICT_EDGE, _runs_dir, evaluate_model, save_model_run
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
    with_ta: bool = False,
    log_mlflow: bool = True,
    save: bool = True,
    interpret: bool = True,
) -> dict[str, Any]:
    """Run the full evaluation for one symbol and return the JSON-able summary.

    ``with_macro`` / ``with_ta`` choose the input set. The model spec, CV machinery
    and verdict gates are identical for every combination (a clean controlled
    experiment) — only the feature matrix changes. Runs are saved under a suffix so
    they never clobber each other: ``""`` (price+macro, canonical), ``_price_only``,
    ``_macro_ta`` (price+macro+TA), ``_price_ta``.
    """
    mcfg = mcfg or ModelConfig.load()
    vcfg = vcfg or ValidationConfig.load()
    if with_ta:
        suffix = "_macro_ta" if with_macro else "_price_ta"
        tag = "price+macro+ta" if with_macro else "price+ta"
    else:
        suffix = "" if with_macro else "_price_only"
        tag = "price+macro" if with_macro else "price-only"

    logger.info(f"[{symbol}] assembling matrix ({tag}, timeout={mcfg.target.timeout_handling})")
    tm = load_training_matrix(
        symbol,
        timeout_handling=mcfg.target.timeout_handling,
        with_macro=with_macro,
        with_ta=with_ta,
    )
    logger.info(
        f"[{symbol}] n={tm.n} eff_n={tm.uniqueness.sum():.0f} "
        f"feat={len(tm.feature_cols)} (+{len(tm.macro_cols)} macro +{len(tm.ta_cols)} ta) — "
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


def _delta(candidate: float | None, baseline: float | None) -> float | None:
    """``candidate - baseline`` rounded, or ``None`` if either side is missing."""
    if candidate is None or baseline is None:
        return None
    return round(candidate - baseline, 6)


def _ta_conclusion(baseline_gate: dict[str, Any], candidate_gate: dict[str, Any]) -> str:
    """Plain-English conclusion based STRICTLY on the gate verdicts + headline metric.

    The headline metric is the deflated, beta-netted excess DSR (falling back to the
    raw excess Sharpe only if DSR is unavailable). No thresholds are tuned here; this
    only narrates what the unchanged verdict gates already decided.
    """
    bv, cv = baseline_gate["verdict"], candidate_gate["verdict"]
    if cv == VERDICT_EDGE:
        return "TA cleared all gates"
    if bv == VERDICT_EDGE and cv != VERDICT_EDGE:
        return "TA worsened: removed the baseline edge"

    def _headline(g: dict[str, Any]) -> float | None:
        return g["excess_dsr"] if g["excess_dsr"] is not None else g["excess_sharpe"]

    b, c = _headline(baseline_gate), _headline(candidate_gate)
    if b is None or c is None:
        return "TA did not clear gates (headline metric unavailable for comparison)"
    if c > b:
        return "TA improved but did not clear gates"
    if c < b:
        return "TA worsened"
    return "TA made no difference and did not clear gates"


def compare_ta_symbol(
    symbol: str,
    mcfg: ModelConfig | None = None,
    vcfg: ValidationConfig | None = None,
    *,
    log_mlflow: bool = True,
    interpret: bool = True,
) -> dict[str, Any]:
    """Run price+macro (baseline) vs price+macro+TA (candidate) side-by-side.

    The two runs are identical except for the TA columns (same labels, same CV, same
    verdict gates), isolating whether the opt-in v2 TA layer adds a real, deflated,
    beta-beating edge over the canonical Phase-6 price+macro baseline. The baseline
    run is executed with ``save=False`` so the canonical ``<symbol>.json`` verdict
    file is never overwritten by this experiment; the candidate is saved as
    ``<symbol>_macro_ta.json`` and the comparison as ``<symbol>_ta_comparison.json``.
    """
    mcfg = mcfg or ModelConfig.load()
    vcfg = vcfg or ValidationConfig.load()
    baseline = run_symbol(
        symbol,
        mcfg,
        vcfg,
        with_macro=True,
        with_ta=False,
        log_mlflow=log_mlflow,
        interpret=interpret,
        save=False,
    )
    candidate = run_symbol(
        symbol,
        mcfg,
        vcfg,
        with_macro=True,
        with_ta=True,
        log_mlflow=log_mlflow,
        interpret=interpret,
    )
    b, c = _gate_row(baseline), _gate_row(candidate)
    comparison = {
        "symbol": symbol,
        "baseline_input_set": "price+macro",
        "candidate_input_set": "price+macro+ta",
        "model_version": mcfg.model_version,
        "feature_version": candidate["feature_version"],
        "label_version": candidate["label_version"],
        "macro_feature_version": candidate["macro_feature_version"],
        "ta_feature_version": candidate["ta_feature_version"],
        "baseline": b,
        "candidate": c,
        "verdict_changed": baseline["verdict"] != candidate["verdict"],
        "delta_directional_accuracy": _delta(c["directional_accuracy"], b["directional_accuracy"]),
        "delta_excess_sharpe": _delta(c["excess_sharpe"], b["excess_sharpe"]),
        "delta_excess_dsr": _delta(c["excess_dsr"], b["excess_dsr"]),
        "delta_pbo": _delta(c["pbo"], b["pbo"]),
        "conclusion": _ta_conclusion(b, c),
    }
    path = _runs_dir() / f"{symbol}_ta_comparison.json"
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


def _print_ta_comparison(c: dict[str, Any]) -> None:
    b, ca = c["baseline"], c["candidate"]
    logger.info(f"[{c['symbol']}] PRICE+MACRO vs PRICE+MACRO+TA (tafv={c['ta_feature_version']})")
    logger.info(
        f"  baseline   : verdict={b['verdict']!r} acc={b['directional_accuracy']} "
        f"excessSR={b['excess_sharpe']} (long={b['long_benchmark_sharpe']}) "
        f"excessDSR={b['excess_dsr']} PBO={b['pbo']} CPCVexSR={b['cpcv_excess_sharpe_mean']}"
    )
    logger.info(
        f"  +TA        : verdict={ca['verdict']!r} acc={ca['directional_accuracy']} "
        f"excessSR={ca['excess_sharpe']} (long={ca['long_benchmark_sharpe']}) "
        f"excessDSR={ca['excess_dsr']} PBO={ca['pbo']} CPCVexSR={ca['cpcv_excess_sharpe_mean']} "
        f"[{ca['n_features']} feat]"
    )
    logger.info(
        f"  Δacc={c['delta_directional_accuracy']} ΔexcessSR={c['delta_excess_sharpe']} "
        f"ΔexcessDSR={c['delta_excess_dsr']} ΔPBO={c['delta_pbo']}"
    )
    logger.info(f"  verdict changed: {c['verdict_changed']} — {c['conclusion']}")


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
        "--with-ta",
        action="store_true",
        help="add the opt-in v2 TA layer to the (macro default) input set",
    )
    p.add_argument(
        "--compare",
        action="store_true",
        help="run BOTH price-only and price+macro and print the side-by-side verdict comparison",
    )
    p.add_argument(
        "--compare-ta",
        action="store_true",
        help="run price+macro (baseline) vs price+macro+TA (candidate) side-by-side",
    )
    args = p.parse_args(argv)

    symbols = args.symbols or Settings().record_symbols
    mcfg, vcfg = ModelConfig.load(), ValidationConfig.load()
    logger.info(
        f"model_version={mcfg.model_version} symbols={symbols} "
        f"compare={args.compare} compare_ta={args.compare_ta} with_ta={args.with_ta}"
    )
    for symbol in symbols:
        if args.compare_ta:
            c = compare_ta_symbol(
                symbol, mcfg, vcfg, log_mlflow=not args.no_mlflow, interpret=not args.no_interpret
            )
            _print_ta_comparison(c)
        elif args.compare:
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
                with_ta=args.with_ta,
                log_mlflow=not args.no_mlflow,
                interpret=not args.no_interpret,
            )
            _print_verdict(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
