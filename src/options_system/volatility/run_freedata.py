"""Phase-24 — free-data incremental-value verdict (over the confirmed Phase-23 h=1 baseline).

Frozen contract: ``docs/PHASE24_PREREGISTRATION.md`` / ``config/phase24_freedata.yaml``. Run::

    uv run python -m options_system.volatility.run_freedata --symbols MES MNQ

For each free-data block (``x1`` market-data, ``s3`` GKG news-tone), per symbol, this asks whether
adding the block to the confirmed Phase-23 model lowers OOS QLIKE at **h = 1** — significantly
(one-sided DM), regime-robustly (G2), and across walk-forward folds (G3). The baseline arm is the
**exact** Phase-23 feature set (the modeling core is inherited verbatim), so the only difference
between the two arms of each test is the block; both arms are evaluated on the same OOS rows.

A block whose OOS coverage is below the pre-registered threshold is **DEFERRED** (reported, not run
as a null) — ``s3`` is deferred until its GKG backfill covers the test window.

Forecast-skill verdict only — forecast skill is not tradeable money; authorizes no strategy /
backtest / risk / execution / live trading. Reads only local lakes; no Databento/IBKR/network/spend.
"""

from __future__ import annotations

import argparse
import json
from glob import glob as _glob
from typing import Any

import numpy as np

from config.settings import Settings

from ..common.logging import get_logger
from ..data.store import DuckStore
from ..marketdata.config import MarketDataConfig
from ..marketdata.lake import MarketDailyLake
from ..validation._purge import train_indices
from ..validation.forecast_stats import diebold_mariano, qlike_from_log
from .config import VolatilityConfig
from .config_freedata import Arm, Phase24Config
from .dataset import VolatilityMatrix, build_daily_base, make_matrix
from .lgbm import build_vol_estimator, fit_vol_fold
from .run import _MIN_TRAIN, regime_labels, treatment_shap

logger = get_logger(__name__)


def _runs_dir():
    return Settings().data_dir / "volatility" / "runs_fd"


# --------------------------------------------------------------------------- #
# Treatment-only anchored expanding-window OOS forecasts (one arm)
# --------------------------------------------------------------------------- #
def treat_oos(mtm: VolatilityMatrix, vcfg: VolatilityConfig) -> dict[str, Any]:
    """Pooled OOS treatment forecasts + per-row fold ids for one feature set (the Phase-21/23 WF).

    Identical anchored-expanding walk-forward and leak-safe purge as Phase 23, but only the LightGBM
    treatment arm (no benchmarks) — used to score the baseline and the augmented feature sets on the
    same folds/rows so their QLIKE is directly comparable.
    """
    n = mtm.n
    t0, t1 = mtm.t0, mtm.t1
    oos_start = np.datetime64(vcfg.walk_forward.oos_start)
    oos_mask = t0 >= oos_start
    if not oos_mask.any():
        raise ValueError(f"[{mtm.symbol}] no OOS rows at/after {vcfg.walk_forward.oos_start}")
    oos_all = np.flatnonzero(oos_mask)
    folds = [f for f in np.array_split(oos_all, vcfg.walk_forward.n_steps) if f.size]
    embargo = mtm.horizon - 1

    pred = np.full(n, np.nan)
    fold_id = np.full(n, -1, dtype=int)
    scored = np.zeros(n, dtype=bool)
    for fi, test_idx in enumerate(folds):
        first = int(test_idx[0])
        candidates = np.arange(0, first)
        train_idx = train_indices(t0, t1, test_idx, n, embargo, candidates=candidates)
        if train_idx.size < _MIN_TRAIN:
            continue
        est = build_vol_estimator(vcfg)
        fit_vol_fold(
            est,
            mtm.x_treat,
            mtm.y,
            t0=t0,
            t1=t1,
            train_idx=train_idx,
            n=n,
            inner_val_fraction=vcfg.lgbm.inner_val_fraction,
            embargo_bars=embargo,
        )
        pred[test_idx] = est.predict(mtm.x_treat[test_idx])
        fold_id[test_idx] = fi
        scored[test_idx] = True

    s = scored
    return {
        "idx": np.flatnonzero(s),
        "y_true": mtm.y[s],
        "fcast": pred[s],
        "fold_id": fold_id[s],
        "rv": mtm.rv[s],
        "t0": mtm.t0[s],
    }


# --------------------------------------------------------------------------- #
# Coverage (cheap, lake-level — avoids the expensive augmented build when deferring)
# --------------------------------------------------------------------------- #
def _coverage_fraction(oos_t0: np.ndarray, dmin: np.datetime64, dmax: np.datetime64) -> float:
    """Fraction of OOS decision days whose date falls within a block's available [dmin, dmax]."""
    if oos_t0.size == 0:
        return 0.0
    lo = oos_t0 >= dmin
    hi = oos_t0 < (dmax + np.timedelta64(1, "D"))
    return float(np.mean(lo & hi))


def _marketdata_date_range() -> tuple[np.datetime64, np.datetime64] | None:
    mdcfg = MarketDataConfig.load()
    md = MarketDailyLake(dataset=mdcfg.storage.dataset).read()
    if md.is_empty():
        return None
    return np.datetime64(str(md["obs_date"].min())), np.datetime64(str(md["obs_date"].max()))


def _gkg_covered_dates() -> set[str] | None:
    """The SET of GKG day-partitions present under ``data/sentiment_gkg_scores/date=YYYY-MM-DD``."""
    base = Settings().data_dir / "sentiment_gkg_scores"
    parts = {p.rsplit("date=", 1)[-1].rstrip("/") for p in _glob(str(base / "date=*"))}
    parts = {p for p in parts if p}
    return parts or None


def _block_coverage(arm: Arm, oos_t0: np.ndarray) -> tuple[float, str]:
    """OOS-coverage of a block over the decision days → (fraction, note).

    Block-specific so an interrupted/gappy backfill cannot falsely pass the gate:

    * ``marketdata`` — daily FRED data joined as-of with forward-fill, so a day is covered iff the
      lake range spans it (a single missing business day still carries the prior value); the range
      is the correct measure.
    * ``gkg`` — measured by **day-partition set membership** (each OOS day must actually have a GKG
      partition present), NOT the min/max range, so a missing interior slot is correctly *not*
      counted. (The chronological bulk backfill fills contiguously, but this is robust to gaps.)
    """
    if arm.block == "marketdata":
        rng = _marketdata_date_range()
        if rng is None:
            return 0.0, "marketdata lake empty"
        cov = _coverage_fraction(oos_t0, rng[0], rng[1])
        return cov, f"marketdata lake covers [{rng[0]} .. {rng[1]}] (as-of, forward-filled)"
    dates = _gkg_covered_dates()
    if not dates:
        return 0.0, "gkg lake empty"
    oos_days = oos_t0.astype("datetime64[D]").astype(str)
    cov = float(np.mean([d in dates for d in oos_days])) if oos_days.size else 0.0
    return cov, f"gkg has {len(dates)} day-partitions; OOS coverage by per-day membership"


# --------------------------------------------------------------------------- #
# One arm: baseline vs augmented incremental comparison + gates
# --------------------------------------------------------------------------- #
def _block_col_idx(mtm: VolatilityMatrix, prefix: str) -> list[int]:
    return [i for i, c in enumerate(mtm.treat_cols) if c.startswith(prefix)]


def evaluate_arm(
    arm: Arm,
    symbol: str,
    p24: Phase24Config,
    base_oos: dict[str, Any],
    mtm_base: VolatilityMatrix,
    regime_full: np.ndarray,
    *,
    store: DuckStore,
    interpret: bool,
) -> dict[str, Any]:
    """Incremental verdict for one free-data block vs the confirmed baseline (or a deferral)."""
    oos_t0 = mtm_base.t0[mtm_base.t0 >= np.datetime64(p24.core.walk_forward.oos_start)]
    cov, cov_note = _block_coverage(arm, oos_t0)
    thresh = p24.coverage.min_oos_fraction
    if cov < thresh:
        logger.info(
            f"[{symbol}] arm {arm.key}: coverage {cov:.3f} < {thresh} — DEFERRED ({cov_note})"
        )
        return {
            "key": arm.key,
            "block": arm.block,
            "status": "deferred_coverage",
            "oos_coverage": round(cov, 4),
            "coverage_threshold": thresh,
            "coverage_note": cov_note,
            "candidate": False,
        }

    # Build the augmented base (baseline features + this block) and score it on the SAME rows/folds.
    aug_cfg = p24.augmented_core(arm)
    aug_base = build_daily_base(symbol, aug_cfg, store=store)
    mtm_aug = make_matrix(aug_base, p24.core.horizons.primary)
    if not np.array_equal(mtm_aug.session_date, mtm_base.session_date):
        raise ValueError(
            f"[{symbol}] arm {arm.key}: augmented rows differ from baseline — cannot pair"
        )

    # Precise OOS coverage from the built block columns (≥1 non-null block feature per OOS row).
    blk = _block_col_idx(mtm_aug, arm.col_prefix)
    oos_mask = mtm_aug.t0 >= np.datetime64(p24.core.walk_forward.oos_start)
    precise_cov = (
        float(np.mean(np.isfinite(mtm_aug.x_treat[np.flatnonzero(oos_mask)][:, blk]).any(axis=1)))
        if blk
        else 0.0
    )

    aug_oos = treat_oos(mtm_aug, aug_cfg)
    if not (
        np.array_equal(aug_oos["idx"], base_oos["idx"])
        and np.allclose(aug_oos["y_true"], base_oos["y_true"])
    ):
        raise ValueError(f"[{symbol}] arm {arm.key}: OOS rows misaligned between arms")

    y = base_oos["y_true"]
    ql_base = qlike_from_log(y, base_oos["fcast"])
    ql_aug = qlike_from_log(y, aug_oos["fcast"])
    mean_base, mean_aug = float(np.mean(ql_base)), float(np.mean(ql_aug))
    dm = diebold_mariano(ql_base, ql_aug, horizon=mtm_base.horizon, alpha=p24.core.dm.alpha)
    g1 = bool(mean_aug < mean_base and dm["significant"])

    # G2 — regime robustness (augmented <= baseline in both calm and turbulent).
    oos_regime = regime_full[base_oos["idx"]]
    per_regime: dict[str, Any] = {}
    g2_parts: list[bool] = []
    for rname, mask in (("calm", ~oos_regime), ("turbulent", oos_regime)):
        if mask.any():
            mb, ma = float(np.mean(ql_base[mask])), float(np.mean(ql_aug[mask]))
            ok = bool(ma <= mb)
            per_regime[rname] = {
                "n": int(mask.sum()),
                "qlike_base": round(mb, 6),
                "qlike_aug": round(ma, 6),
                "aug_le_base": ok,
            }
            g2_parts.append(ok)
        else:
            per_regime[rname] = {
                "n": 0,
                "qlike_base": None,
                "qlike_aug": None,
                "aug_le_base": False,
            }
            g2_parts.append(False)
    g2 = bool(len(g2_parts) == 2 and all(g2_parts))

    # G3 — temporal stability (augmented beats baseline in >= min_folds folds).
    fold_id = base_oos["fold_id"]
    uniq = np.unique(fold_id)
    beat = sum(
        1
        for f in uniq
        if float(np.mean(ql_aug[fold_id == f])) < float(np.mean(ql_base[fold_id == f]))
    )
    g3cfg = p24.gates.g3_temporal_stability
    g3 = bool(beat >= g3cfg.min_folds_beating_baseline)

    candidate = bool(g1 and g2 and g3)
    shap = treatment_shap(mtm_aug, aug_cfg) if interpret else None
    block_share = (shap or {}).get("block_shares", {}).get(arm.block) if shap else None

    logger.info(
        f"[{symbol}] arm {arm.key}: QLIKE base={mean_base:.6f} aug={mean_aug:.6f} "
        f"(Δ={mean_base - mean_aug:+.6f}) DM_p={dm['p_value']} cov={precise_cov:.3f} "
        f"G1={g1} G2={g2} G3={g3}({beat}/{uniq.size}) -> candidate={candidate}"
    )
    return {
        "key": arm.key,
        "block": arm.block,
        "status": "run",
        "candidate": candidate,
        "oos_coverage": round(precise_cov, 4),
        "n_oos": int(y.size),
        "qlike_baseline": round(mean_base, 6),
        "qlike_augmented": round(mean_aug, 6),
        "qlike_improvement": round(mean_base - mean_aug, 6),
        "dm_stat_hln": dm["dm_stat_hln"],
        "dm_p_value": dm["p_value"],
        "dm_significant": dm["significant"],
        "regime": per_regime,
        "fold_stability": {
            "n_folds": int(uniq.size),
            "aug_beats_base_folds": beat,
            "min_folds_beating_baseline": g3cfg.min_folds_beating_baseline,
        },
        "gates": {
            "g1_incremental_accuracy": g1,
            "g2_regime_robustness": g2,
            "g3_temporal_stability": g3,
        },
        "block_shap_share": block_share,
    }


# --------------------------------------------------------------------------- #
# Per-symbol run
# --------------------------------------------------------------------------- #
def run_symbol_fd(
    symbol: str, p24: Phase24Config, *, store: DuckStore, interpret: bool = True
) -> dict[str, Any]:
    """Build the confirmed baseline once, then evaluate each free-data arm against it."""
    baseline_cfg = p24.baseline_core()
    base_b = build_daily_base(symbol, baseline_cfg, store=store)
    mtm_base = make_matrix(base_b, p24.core.horizons.primary)
    regime_full = regime_labels(mtm_base.rv, p24.core.regime.trailing_days)
    base_oos = treat_oos(mtm_base, baseline_cfg)
    qlike_baseline = float(np.mean(qlike_from_log(base_oos["y_true"], base_oos["fcast"])))

    arms = {
        a.key: evaluate_arm(
            a, symbol, p24, base_oos, mtm_base, regime_full, store=store, interpret=interpret
        )
        for a in p24.arms
    }
    return {
        "symbol": symbol,
        "phase": "24",
        "baseline": "confirmed Phase-23 h=1 model (existing features)",
        "primary_horizon": p24.core.horizons.primary,
        "n_oos": int(base_oos["y_true"].size),
        "qlike_baseline": round(qlike_baseline, 6),
        "arms": arms,
        "cost_disclaimer": "forecast-skill verdict only (incremental QLIKE vs the confirmed "
        "baseline at h=1); forecast skill is not tradeable money — no strategy/backtest/live.",
    }


# --------------------------------------------------------------------------- #
# Decision + persistence
# --------------------------------------------------------------------------- #
def decide_fd(symbol_results: dict[str, dict[str, Any]], p24: Phase24Config) -> dict[str, Any]:
    """Per block: both symbols pass → adds value; one → fragile; both deferred → deferred."""
    per_block: dict[str, Any] = {}
    syms = list(symbol_results)
    for arm in p24.arms:
        statuses = [symbol_results[s]["arms"][arm.key] for s in syms]
        if all(a["status"] == "deferred_coverage" for a in statuses):
            verdict = "deferred_coverage"
        else:
            passes = [s for s, a in zip(syms, statuses, strict=True) if a.get("candidate")]
            if len(passes) == len(syms) and syms:
                verdict = "adds_incremental_value"
            elif passes:
                verdict = "fragile_single_symbol"
            else:
                verdict = "no_incremental_value"
        per_block[arm.key] = {
            "block": arm.block,
            "verdict": verdict,
            "per_symbol": {
                s: ("deferred" if a["status"] != "run" else ("pass" if a["candidate"] else "fail"))
                for s, a in zip(syms, statuses, strict=True)
            },
        }
    return {"per_block": per_block}


def save_symbol_run(result: dict[str, Any]) -> None:
    d = _runs_dir()
    d.mkdir(parents=True, exist_ok=True)
    with (d / f"{result['symbol']}.json").open("w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, default=str)


def save_verdict(verdict: dict[str, Any]) -> None:
    d = _runs_dir()
    d.mkdir(parents=True, exist_ok=True)
    with (d / "verdict.json").open("w", encoding="utf-8") as fh:
        json.dump(verdict, fh, indent=2, default=str)


def _log_mlflow(result: dict[str, Any], p24: Phase24Config) -> None:
    try:
        import os

        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
        import mlflow
    except Exception as exc:  # noqa: BLE001 - tracking is optional
        logger.warning(f"mlflow unavailable ({exc}); skipping tracking")
        return
    mlflow.set_tracking_uri((Settings().data_dir / "mlruns").as_uri())
    mlflow.set_experiment(p24.artifacts.get("mlflow_experiment", "volatility-freedata-fd1"))
    with mlflow.start_run(run_name=f"{result['symbol']}-{p24.freedata_version}"):
        mlflow.log_param("symbol", result["symbol"])
        mlflow.log_metric("qlike_baseline", float(result["qlike_baseline"]))
        for key, a in result["arms"].items():
            if a["status"] == "run":
                mlflow.log_metric(f"qlike_aug_{key}", float(a["qlike_augmented"]))
                mlflow.log_metric(f"dm_p_{key}", float(a["dm_p_value"] or 1.0))
                mlflow.set_tag(f"candidate_{key}", str(a["candidate"]))
            else:
                mlflow.set_tag(f"status_{key}", a["status"])
        mlflow.log_dict(result, "result.json")


def run_freedata(
    symbols: list[str] | None = None,
    *,
    interpret: bool = True,
    log_mlflow: bool = True,
    save: bool = True,
) -> dict[str, Any]:
    """Run the Phase-24 incremental verdict for all symbols → combined verdict (persisted)."""
    p24 = Phase24Config.load()
    syms = list(dict.fromkeys(symbols or p24.core.symbols))
    is_full = set(syms) == set(p24.core.symbols)
    if save and not is_full:
        raise ValueError(
            f"Phase-24 verdict is pre-registered over {list(p24.core.symbols)}; refusing to save a "
            f"canonical decision over {syms}. Re-run with the exact set, or pass save=False."
        )

    store = DuckStore()
    symbol_results: dict[str, dict[str, Any]] = {}
    try:
        for symbol in syms:
            symbol_results[symbol] = run_symbol_fd(symbol, p24, store=store, interpret=interpret)
    finally:
        store.close()

    decision = decide_fd(symbol_results, p24)
    verdict = {
        "phase": "24",
        "freedata_version": p24.freedata_version,
        "preregistration": "docs/PHASE24_PREREGISTRATION.md",
        "study": "free-data incremental value over the confirmed Phase-23 h=1 baseline",
        "baseline": "confirmed Phase-23 h=1 model (existing features)",
        "primary_horizon": p24.core.horizons.primary,
        "coverage_threshold": p24.coverage.min_oos_fraction,
        "symbols": symbol_results,
        "decision": decision,
        "partial": not is_full,
        "cost_disclaimer": "forecast-skill verdict only; forecast skill is not tradeable money; "
        "authorizes no strategy/backtest/risk/execution/live trading.",
    }
    if save:
        for r in symbol_results.values():
            save_symbol_run(r)
        save_verdict(verdict)
    if log_mlflow:
        for r in symbol_results.values():
            _log_mlflow(r, p24)
    return verdict


def _print_verdict(verdict: dict[str, Any]) -> None:
    logger.info("=== Phase-24 free-data incremental verdict ===")
    for sym, r in verdict["symbols"].items():
        for key, a in r["arms"].items():
            if a["status"] == "run":
                g = a["gates"]
                logger.info(
                    f"[{sym}] {key}: base={a['qlike_baseline']} aug={a['qlike_augmented']} "
                    f"DM_p={a['dm_p_value']} G1={g['g1_incremental_accuracy']} "
                    f"G2={g['g2_regime_robustness']} G3={g['g3_temporal_stability']} "
                    f"-> {'PASS' if a['candidate'] else 'fail'}"
                )
            else:
                logger.info(f"[{sym}] {key}: {a['status']} (coverage {a['oos_coverage']})")
    for key, b in verdict["decision"]["per_block"].items():
        logger.info(f"BLOCK {key} ({b['block']}): {b['verdict']} — {b['per_symbol']}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="volatility.run_freedata", description=__doc__)
    p.add_argument("--symbols", nargs="+", default=None, help="default: config symbols")
    p.add_argument("--no-mlflow", action="store_true", help="skip MLflow logging")
    p.add_argument("--no-interpret", action="store_true", help="skip SHAP interpretation")
    args = p.parse_args(argv)
    verdict = run_freedata(
        symbols=args.symbols, interpret=not args.no_interpret, log_mlflow=not args.no_mlflow
    )
    _print_verdict(verdict)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
