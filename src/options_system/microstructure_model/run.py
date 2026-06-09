"""Phase-14 micro pipeline runner: load → search → evaluate → interpret → track.

One command produces, per symbol, the honest deflated verdict for the microstructure
signal model and logs it to the local MLflow store::

    uv run python -m options_system.microstructure_model.run --symbols ES NQ \\
        --start 2026-01-26 --end 2026-06-06

It assembles the leak-free 3-class micro matrix, runs the in-CV search (fold-local
class weighting), evaluates the selected config through CPCV / PBO / DSR on the gross
signal-return proxy, optionally explains it with SHAP, persists a JSON summary under
``data/micro_models/runs/`` and logs the run to MLflow. Nothing here trades,
backtests economically, or promotes a model — it only answers *is there a real,
deflated microstructure signal edge candidate?* An "edge candidate" authorises ONLY
the next phase (an economic backtest with realistic costs), never live trading.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from typing import Any

from ..common.logging import get_logger
from ..microstructure.config import MicrostructureConfig
from ..microstructure.model_config import MicroModelConfig
from ..validation.config import ValidationConfig
from .dataset import load_micro_matrix
from .evaluate import _runs_dir, evaluate_micro_model, save_micro_run
from .interpret import explain
from .lgbm import build_micro_estimator, effective_sample_weight, fold_local_class_weights
from .tune import run_micro_search

logger = get_logger(__name__)


def run_micro_symbol(
    symbol: str,
    mmcfg: MicroModelConfig | None = None,
    vcfg: ValidationConfig | None = None,
    *,
    start: datetime | None = None,
    end: datetime | None = None,
    log_mlflow: bool = True,
    save: bool = True,
    interpret: bool = True,
    rebuild_cache: bool = False,
) -> dict[str, Any]:
    """Run the full micro evaluation for one symbol and return the JSON-able summary."""
    mmcfg = mmcfg or MicroModelConfig.load()
    vcfg = vcfg or ValidationConfig.load()

    logger.info(f"[{symbol}] assembling micro matrix (mm={mmcfg.micro_model_version})")
    mtm = load_micro_matrix(symbol, start=start, end=end, mmcfg=mmcfg, rebuild_cache=rebuild_cache)
    logger.info(
        f"[{symbol}] n={mtm.n} eff_n={mtm.effective_n:.0f} feat={len(mtm.feature_cols)} — "
        f"searching ({mmcfg.search.n_trials} trials, metric={mmcfg.search.selection_metric})"
    )
    search = run_micro_search(mtm, mmcfg, vcfg)
    logger.info(f"[{symbol}] selected {search.selected_overrides} — evaluating via CPCV/PBO/DSR")
    summary = evaluate_micro_model(mtm, search, mmcfg, vcfg)

    estimator = None
    if interpret or log_mlflow:
        # Full-data fit with GLOBAL class weights (artifact + SHAP background only —
        # never a performance number; the leak-safe CV owns those).
        estimator = build_micro_estimator(mmcfg, search.selected_overrides)
        cw = fold_local_class_weights(
            mtm.y,
            mtm.sample_weight,
            use_sample_weight=mmcfg.class_weighting.use_sample_weight_in_balance,
        )
        estimator.fit(
            mtm.X, mtm.y, sample_weight=effective_sample_weight(mtm.y, mtm.sample_weight, cw)
        )

    if interpret:
        png = _runs_dir() / f"{symbol}_shap.png"
        summary["shap"] = explain(
            mtm, mmcfg, search.selected_overrides, estimator=estimator, out_png=png
        )

    run_id = None
    if log_mlflow:
        from pathlib import Path

        from .tracking import log_run

        png = Path(summary.get("shap", {}).get("summary_plot") or "")
        run_id = log_run(
            summary, summary.get("shap"), estimator=estimator, shap_png=png if png.name else None
        )
    summary["mlflow_run_id"] = run_id

    if save:
        save_micro_run(summary)
    return summary


def _print_verdict(summary: dict[str, Any]) -> None:
    c = summary["classification"]
    sig = summary["signal_return"]
    pbo = summary["pbo"]["pbo"] if summary["pbo"] else "n/a"
    gs = summary["cpcv"]["gross_sharpe"]
    logger.info(
        f"[{summary['symbol']}] VERDICT: {summary['verdict'].upper()}  "
        f"(macroF1={c['macro_f1']} action={summary['action_rate']} "
        f"gross_SR={sig['gross_sharpe']} gross_DSR={sig['gross_dsr']} "
        f"mean_gross={sig['mean_gross_return']} PBO={pbo})"
    )
    logger.info(
        f"[{summary['symbol']}] CPCV gross-Sharpe paths: mean={gs['mean']} "
        f"median={gs['median']} min={gs['min']} max={gs['max']}  "
        f"pred_balance={summary['pred_balance']}"
    )
    failed = [k for k, v in summary["verdict_checks"].items() if not v]
    logger.info(f"[{summary['symbol']}] gates failed: {failed or 'none'}")
    if summary.get("shap", {}).get("available"):
        logger.info(f"[{summary['symbol']}] SHAP top: {summary['shap']['top_features'][:6]}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="microstructure_model.run", description=__doc__)
    p.add_argument("--symbols", nargs="+", default=None, help="default: microstructure instruments")
    p.add_argument("--start", help="label window start YYYY-MM-DD (UTC); default: all available")
    p.add_argument("--end", help="label window end YYYY-MM-DD (UTC); default: all available")
    p.add_argument("--no-mlflow", action="store_true", help="skip MLflow logging")
    p.add_argument("--no-interpret", action="store_true", help="skip SHAP interpretation")
    p.add_argument("--rebuild-cache", action="store_true", help="rebuild the micro-matrix cache")
    args = p.parse_args(argv)

    symbols = args.symbols or MicrostructureConfig.load().symbols()
    mmcfg, vcfg = MicroModelConfig.load(), ValidationConfig.load()

    def _parse(d: str | None) -> datetime | None:
        return datetime.fromisoformat(d).replace(tzinfo=UTC) if d else None

    logger.info(f"micro_model_version={mmcfg.micro_model_version} symbols={symbols}")
    for symbol in symbols:
        summary = run_micro_symbol(
            symbol,
            mmcfg,
            vcfg,
            start=_parse(args.start),
            end=_parse(args.end),
            log_mlflow=not args.no_mlflow,
            interpret=not args.no_interpret,
            rebuild_cache=args.rebuild_cache,
        )
        _print_verdict(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
