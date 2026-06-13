"""Phase-19 sentiment micro-model A/B edge verdict — orchestrator.

For each symbol (ES, NQ separately) this runs TWO arms through the *unchanged* Phase-14
micro pipeline and the Phase-4 leak-safe validation framework, on the *identical*
supported-region row set::

    uv run python -m options_system.microstructure_model.phase19_ab --symbols ES NQ

- **Baseline (B)** — ``with_sentiment=False`` → the 17 m1 OFI features (stamp ``mm1``).
- **Treatment (T)** — ``with_sentiment=True`` → OFI **+** the ``s2`` sentiment block
  attached on ``t0`` (stamp ``mm2``).

The sentiment block is the SOLE difference between the arms. Every other element — bars,
labels, target, folds, class weighting, the 8-config search, the selection metric, the
seed, the trial count, and the SIX verdict gates — is read unchanged from
``config/micro_model.yaml`` (mm1). The frozen contract is
``docs/PHASE19_PREREGISTRATION.md``; this run must not deviate from it.

It produces a **signal edge verdict only** (gross signal-return proxy, no costs). It is
NOT a strategy and NOT an economic backtest; it authorizes NO live trading. Per-symbol,
never pooled. Reads only the local lakes — no Databento, no IBKR, no network, no spend.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from config.settings import Settings

from ..common.logging import get_logger
from ..microstructure.model_config import MicroModelConfig
from ..sentiment.config import SentimentConfig
from ..validation.config import ValidationConfig
from .dataset import MicroTrainingMatrix, load_micro_matrix
from .evaluate import VERDICT_EDGE, evaluate_micro_model
from .interpret import explain
from .lgbm import build_micro_estimator, effective_sample_weight, fold_local_class_weights
from .phase19_config import ArmCfg, Phase19Config
from .tune import run_micro_search

logger = get_logger(__name__)


def _runs_dir() -> Path:
    return Path(Settings().data_dir) / "phase19_ab" / "runs"


# --------------------------------------------------------------------------- #
# One arm
# --------------------------------------------------------------------------- #
def _full_data_estimator(mtm: MicroTrainingMatrix, mmcfg: MicroModelConfig, overrides: dict) -> Any:
    """Full-data fit with global class weights — SHAP background/artifact only.

    Never a performance number (the leak-safe CV owns those); mirrors the Phase-14
    runner's interpret-fit exactly so the two stay identical.
    """
    est = build_micro_estimator(mmcfg, overrides)
    cw = fold_local_class_weights(
        mtm.y,
        mtm.sample_weight,
        use_sample_weight=mmcfg.class_weighting.use_sample_weight_in_balance,
    )
    est.fit(mtm.X, mtm.y, sample_weight=effective_sample_weight(mtm.y, mtm.sample_weight, cw))
    return est


def run_arm(
    symbol: str,
    arm: ArmCfg,
    *,
    mmcfg: MicroModelConfig,
    vcfg: ValidationConfig,
    scfg: SentimentConfig,
    start: datetime,
    end: datetime,
    interpret: bool,
    rebuild_cache: bool,
) -> tuple[MicroTrainingMatrix, dict[str, Any]]:
    """Build the arm's matrix and run it through the unchanged search + evaluation."""
    mtm = load_micro_matrix(
        symbol,
        start=start,
        end=end,
        mmcfg=mmcfg,
        with_sentiment=arm.with_sentiment,
        scfg=scfg if arm.with_sentiment else None,
        version_stamp=arm.model_version,
        rebuild_cache=rebuild_cache,
    )
    search = run_micro_search(mtm, mmcfg, vcfg)
    summary = evaluate_micro_model(mtm, search, mmcfg, vcfg)
    summary["arm"] = arm.name
    summary["with_sentiment"] = arm.with_sentiment
    # SHAP only for the Treatment arm — to show which sent_* features (if any) it uses.
    if interpret and arm.with_sentiment:
        out_png = _runs_dir() / f"{symbol}_{arm.name}_shap.png"
        out_png.parent.mkdir(parents=True, exist_ok=True)
        est = _full_data_estimator(mtm, mmcfg, search.selected_overrides)
        summary["shap"] = explain(
            mtm, mmcfg, search.selected_overrides, estimator=est, out_png=out_png
        )
    return mtm, summary


def assert_identical_rows(b: MicroTrainingMatrix, t: MicroTrainingMatrix, symbol: str) -> None:
    """Both arms MUST see the identical row set per symbol — abort the A/B otherwise.

    The row gate runs on the m1 features only and the sentiment attach preserves every
    row, so the two matrices are expected to align exactly; a mismatch is a real bug
    that would invalidate the controlled comparison, so we fail loudly rather than
    silently compare apples to oranges.
    """
    ok = (
        b.n == t.n
        and np.array_equal(b.t0, t.t0)
        and np.array_equal(b.t1, t.t1)
        and np.array_equal(b.y, t.y)
        and np.allclose(b.ret_t1, t.ret_t1, equal_nan=True)
        and np.allclose(b.sample_weight, t.sample_weight)
        and np.allclose(b.uniqueness_weight, t.uniqueness_weight)
    )
    if not ok:
        raise RuntimeError(
            f"[{symbol}] baseline (n={b.n}) and treatment (n={t.n}) arms did NOT run on "
            "the identical row set — the A/B is invalid; aborting."
        )


# --------------------------------------------------------------------------- #
# Attribution + decision (pure, exactly as pre-registered)
# --------------------------------------------------------------------------- #
def attribute(baseline: dict[str, Any], treatment: dict[str, Any]) -> dict[str, Any]:
    """Frozen attribution logic. Primary verdict = does Treatment clear all six gates?

    T pass + B fail → attributable to sentiment. T pass + B pass → not cleanly
    attributable (the restricted rows alone moved the baseline) → flag for scrutiny.
    T fail → null for that symbol, regardless of B.
    """
    b_pass = baseline["verdict"] == VERDICT_EDGE
    t_pass = treatment["verdict"] == VERDICT_EDGE
    if t_pass and not b_pass:
        attribution = "attributable_to_sentiment"
    elif t_pass and b_pass:
        attribution = "not_cleanly_attributable"
    else:
        attribution = "null"
    return {"baseline_pass": b_pass, "treatment_pass": t_pass, "attribution": attribution}


def decide(symbol_results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Frozen decision rule over the per-symbol attribution blocks.

    T clears all six on a symbol → "sentiment edge candidate" (authorizes ONLY a future
    Phase 20 economic backtest for that symbol; never live trading); a single-symbol
    candidate while the other fails is flagged **fragile**. T fails on both → sentiment
    is the fifth honest null.
    """
    per_symbol: dict[str, str] = {}
    candidates: list[str] = []
    for sym, res in symbol_results.items():
        t_pass = res["attribution"]["treatment_pass"]
        per_symbol[sym] = "sentiment_edge_candidate" if t_pass else "null"
        if t_pass:
            candidates.append(sym)

    n = len(symbol_results)
    fragile = bool(candidates) and len(candidates) < n
    if not candidates:
        overall = "no_significant_edge"
        note = (
            "sentiment is the fifth honest null; remaining forks are MBP-10 "
            "(paid, billing-gated, blocked) or a horizon/regime redesign — no sentiment "
            "re-tuning."
        )
    elif fragile:
        overall = "sentiment_edge_candidate_fragile"
        note = (
            f"single-symbol candidate ({', '.join(candidates)}) with the other symbol "
            "failing — flagged FRAGILE (the Phase-14 NQ near-miss is the precedent); "
            "authorizes ONLY a Phase 20 economic backtest for the passing symbol, never "
            "live trading."
        )
    else:
        overall = "sentiment_edge_candidate"
        note = (
            "authorizes ONLY a future Phase 20 economic backtest (realistic costs/"
            "slippage) per passing symbol — never live trading."
        )
    return {
        "per_symbol": per_symbol,
        "candidates": candidates,
        "fragile": fragile,
        "overall": overall,
        "note": note,
    }


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def _arm_view(summary: dict[str, Any]) -> dict[str, Any]:
    """A compact, comparison-friendly slice of a full arm summary (for the verdict file)."""
    sig = summary["signal_return"]
    cpcv = summary["cpcv"]["gross_sharpe"]
    return {
        "arm": summary["arm"],
        "model_version": summary["micro_model_version"],
        "n_samples": summary["n_samples"],
        "effective_n": summary["effective_n"],
        "n_features": summary["n_features"],
        "verdict": summary["verdict"],
        "verdict_checks": summary["verdict_checks"],
        "selected_overrides": summary["selected_overrides"],
        "metrics": {
            "pbo": summary["pbo"]["pbo"] if summary["pbo"] else None,
            "gross_dsr": sig["gross_dsr"],
            "mean_gross_return": sig["mean_gross_return"],
            "action_rate": summary["action_rate"],
            "macro_f1": summary["classification"]["macro_f1"],
            "cpcv_median_gross_sharpe": cpcv["median"],
        },
        "shap_top": (summary.get("shap") or {}).get("top_features", [])[:8],
    }


def save_arm_run(summary: dict[str, Any], *, runs_dir: Path | None = None) -> Path:
    """Persist one full arm evaluation under ``data/phase19_ab/runs/<symbol>_<arm>.json``."""
    d = runs_dir or _runs_dir()
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{summary['symbol']}_{summary['arm']}.json"
    with path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)
    return path


def save_verdict(verdict: dict[str, Any], *, runs_dir: Path | None = None) -> Path:
    """Persist the combined A/B verdict under ``data/phase19_ab/runs/verdict.json``."""
    d = runs_dir or _runs_dir()
    d.mkdir(parents=True, exist_ok=True)
    path = d / "verdict.json"
    with path.open("w", encoding="utf-8") as fh:
        json.dump(verdict, fh, indent=2, default=str)
    return path


def read_verdict(*, runs_dir: Path | None = None) -> dict[str, Any] | None:
    """Load the saved combined A/B verdict, or ``None`` if no run has been saved."""
    path = (runs_dir or _runs_dir()) / "verdict.json"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------- #
# Per-symbol + full run
# --------------------------------------------------------------------------- #
def run_phase19_symbol(
    symbol: str,
    p19cfg: Phase19Config,
    mmcfg: MicroModelConfig,
    vcfg: ValidationConfig,
    scfg: SentimentConfig,
    *,
    interpret: bool = True,
    rebuild_cache: bool = False,
    log_mlflow: bool = True,
    save: bool = True,
) -> dict[str, Any]:
    """Run both arms for one symbol → attribution + the compact per-symbol result."""
    start, end = p19cfg.window.start_dt(), p19cfg.window.end_dt()
    logger.info(f"[{symbol}] Phase-19 A/B (mm1 vs mm2) on t0∈[{start.date()}..{end.date()}]")

    mtm_b, sum_b = run_arm(
        symbol,
        p19cfg.baseline,
        mmcfg=mmcfg,
        vcfg=vcfg,
        scfg=scfg,
        start=start,
        end=end,
        interpret=interpret,
        rebuild_cache=rebuild_cache,
    )
    mtm_t, sum_t = run_arm(
        symbol,
        p19cfg.treatment,
        mmcfg=mmcfg,
        vcfg=vcfg,
        scfg=scfg,
        start=start,
        end=end,
        interpret=interpret,
        rebuild_cache=rebuild_cache,
    )
    assert_identical_rows(mtm_b, mtm_t, symbol)

    attribution = attribute(sum_b, sum_t)

    if save:
        save_arm_run(sum_b)
        save_arm_run(sum_t)
    if log_mlflow:
        from .tracking import log_run

        for summary in (sum_b, sum_t):
            log_run(summary, summary.get("shap"), experiment=p19cfg.mlflow_experiment)

    return {
        "symbol": symbol,
        "row_region": {"start": start.date().isoformat(), "end": end.date().isoformat()},
        "n_rows": mtm_b.n,
        "effective_n": round(mtm_b.effective_n, 2),
        "baseline": _arm_view(sum_b),
        "treatment": _arm_view(sum_t),
        "attribution": attribution,
    }


def run_phase19(
    symbols: list[str] | None = None,
    *,
    interpret: bool = True,
    rebuild_cache: bool = False,
    log_mlflow: bool = True,
    save: bool = True,
) -> dict[str, Any]:
    """Run the full Phase-19 A/B for all symbols → the combined verdict (and persist it)."""
    p19cfg = Phase19Config.load()
    mmcfg = MicroModelConfig.load()
    vcfg = ValidationConfig.load()
    scfg = SentimentConfig.load()

    # Pin the sentiment feature version to the pre-registered s2 (guard against drift).
    if scfg.aggregation.feature_version != p19cfg.sentiment_feature_version:
        raise ValueError(
            f"sentiment aggregation feature_version={scfg.aggregation.feature_version} "
            f"!= pre-registered {p19cfg.sentiment_feature_version} — the s2 block changed; "
            "reconcile before running the frozen A/B."
        )

    syms = symbols or p19cfg.symbols
    symbol_results: dict[str, dict[str, Any]] = {}
    for symbol in syms:
        symbol_results[symbol] = run_phase19_symbol(
            symbol,
            p19cfg,
            mmcfg,
            vcfg,
            scfg,
            interpret=interpret,
            rebuild_cache=rebuild_cache,
            log_mlflow=log_mlflow,
            save=save,
        )

    decision = decide(symbol_results)
    verdict = {
        "phase": "19",
        "phase19_version": p19cfg.phase19_version,
        "preregistration": "docs/PHASE19_PREREGISTRATION.md",
        "micro_model_version_baseline": p19cfg.baseline.model_version,
        "micro_model_version_treatment": p19cfg.treatment.model_version,
        "sentiment_feature_version": p19cfg.sentiment_feature_version,
        "window": {"start": p19cfg.window.start, "end": p19cfg.window.end},
        "verdict_thresholds": mmcfg.verdict.model_dump(mode="json"),
        "symbols": symbol_results,
        "decision": decision,
        "cost_disclaimer": "gross signal-return proxy only; not an executable backtest "
        "(no commissions or slippage); authorizes no strategy/backtest/live trading.",
    }
    if save:
        save_verdict(verdict)
    return verdict


def _print_verdict(verdict: dict[str, Any]) -> None:
    logger.info("=== Phase-19 sentiment A/B verdict ===")
    for sym, res in verdict["symbols"].items():
        b, t = res["baseline"], res["treatment"]
        logger.info(
            f"[{sym}] rows={res['n_rows']} effN={res['effective_n']} | "
            f"B(mm1)={b['verdict']} gates_failed="
            f"{[k for k, v in b['verdict_checks'].items() if not v] or 'none'} | "
            f"T(mm2)={t['verdict']} gates_failed="
            f"{[k for k, v in t['verdict_checks'].items() if not v] or 'none'} | "
            f"attribution={res['attribution']['attribution']}"
        )
        if t.get("shap_top"):
            logger.info(f"[{sym}] T SHAP top: {t['shap_top'][:6]}")
    d = verdict["decision"]
    logger.info(f"DECISION: {d['overall']} (candidates={d['candidates'] or 'none'}) — {d['note']}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="microstructure_model.phase19_ab", description=__doc__)
    p.add_argument("--symbols", nargs="+", default=None, help="default: config phase19.symbols")
    p.add_argument("--no-mlflow", action="store_true", help="skip MLflow logging")
    p.add_argument("--no-interpret", action="store_true", help="skip SHAP interpretation")
    p.add_argument("--rebuild-cache", action="store_true", help="rebuild the micro-matrix cache")
    args = p.parse_args(argv)

    verdict = run_phase19(
        symbols=args.symbols,
        interpret=not args.no_interpret,
        rebuild_cache=args.rebuild_cache,
        log_mlflow=not args.no_mlflow,
    )
    _print_verdict(verdict)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
