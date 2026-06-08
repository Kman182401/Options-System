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
    log_mlflow: bool = True,
    save: bool = True,
    interpret: bool = True,
) -> dict[str, Any]:
    """Run the full evaluation for one symbol and return the JSON-able summary."""
    mcfg = mcfg or ModelConfig.load()
    vcfg = vcfg or ValidationConfig.load()

    logger.info(f"[{symbol}] assembling matrix (timeout={mcfg.target.timeout_handling})")
    tm = load_training_matrix(symbol, timeout_handling=mcfg.target.timeout_handling)
    logger.info(
        f"[{symbol}] n={tm.n} eff_n={tm.uniqueness.sum():.0f} — "
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
        png = _runs_dir() / f"{symbol}_shap.png"
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
        save_model_run(summary)
    return summary


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
    args = p.parse_args(argv)

    symbols = args.symbols or Settings().record_symbols
    mcfg, vcfg = ModelConfig.load(), ValidationConfig.load()
    logger.info(f"model_version={mcfg.model_version} symbols={symbols}")
    for symbol in symbols:
        summary = run_symbol(
            symbol,
            mcfg,
            vcfg,
            log_mlflow=not args.no_mlflow,
            interpret=not args.no_interpret,
        )
        _print_verdict(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
