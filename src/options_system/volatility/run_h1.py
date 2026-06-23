"""Phase-23 — 1-day RV-forecast CONFIRMATION verdict (hardened benchmark battery).

Frozen contract: ``docs/PHASE23_PREREGISTRATION.md`` / ``config/phase23_vol_h1.yaml``. Run::

    uv run python -m options_system.volatility.run_h1 --symbols MES MNQ

This reuses the **identical** Phase-21 pipeline (RV target, the single fixed LightGBM treatment, the
existing leak-safe feature set, the anchored expanding walk-forward, the regime split, the QLIKE +
Diebold-Mariano machinery) and re-asks the question at the **gated h = 1** under a harder bar: the
treatment must beat a four-benchmark battery {HAR-RV, random walk, EWMA(0.94), GARCH(1,1)} and clear
four gates per symbol —

* **G1** accuracy vs HAR (QLIKE + significant one-sided DM),
* **G2** regime robustness (treat ≤ HAR in both calm and turbulent),
* **G3** benchmark hardness (beat each challenger with a significant DM),
* **G4** temporal stability (beat HAR and RW in ≥ ``min_folds`` of the walk-forward folds).

Forecast-skill verdict only — forecast skill is not tradeable money; authorizes no strategy /
backtest / risk / execution / live trading. Reads only local lakes; no Databento/IBKR/network/spend.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from typing import Any

import numpy as np
import polars as pl

from config.settings import Settings

from ..common.logging import get_logger
from ..data.store import DuckStore
from ..features.config import FeatureConfig
from ..validation._purge import train_indices
from ..validation.forecast_stats import diebold_mariano, qlike_from_log
from .benchmarks import (
    ewma_forecast_log,
    fit_garch11,
    garch11_forecast_log,
    riskmetrics_params,
    rw_forecast_log,
)
from .config import VolatilityConfig
from .config_h1 import Phase23Config
from .dataset import DailyBase, VolatilityMatrix, build_daily_base, make_matrix
from .har import fit_har, predict_har
from .lgbm import build_vol_estimator, fit_vol_fold
from .realized import daily_rth_log_return
from .run import _MIN_TRAIN, evaluate_horizon, regime_labels, treatment_shap

logger = get_logger(__name__)

VERDICT_CONFIRMED = "confirmed 1-day RV forecast skill"
VERDICT_NONE = "no skill"

_WIDE_START = datetime(2000, 1, 1, tzinfo=UTC)
_WIDE_END = datetime(2100, 1, 1, tzinfo=UTC)


def _runs_dir():
    return Settings().data_dir / "volatility" / "runs_h1"


# --------------------------------------------------------------------------- #
# Daily return series aligned to the matrix rows (for the GARCH benchmark)
# --------------------------------------------------------------------------- #
def aligned_returns(symbol: str, mtm: VolatilityMatrix, scfg, store: DuckStore) -> np.ndarray:
    """Within-session RTH returns aligned 1:1 to ``mtm``'s (session-ordered) rows.

    GARCH needs a return series the RV target does not carry. Same RTH sessionization and window as
    the RV estimator (excludes overnight), joined on ``session_date``, so each matrix row gets its
    own self-contained session return.
    """
    bars = store.get_bars(symbol, _WIDE_START, _WIDE_END, freq="1m", continuous=True)
    ret_df = daily_rth_log_return(bars, scfg).with_columns(pl.col("session_date").cast(pl.Date))
    left = pl.DataFrame({"session_date": mtm.session_date}).with_columns(
        pl.col("session_date").cast(pl.Date)
    )
    joined = left.join(ret_df, on="session_date", how="left")
    return joined["ret"].to_numpy().astype(float)


# --------------------------------------------------------------------------- #
# Anchored expanding-window OOS forecasts: treatment + the full benchmark battery
# --------------------------------------------------------------------------- #
def anchored_oos_h1(
    mtm: VolatilityMatrix, vcfg: VolatilityConfig, returns: np.ndarray, *, ewma_lambda: float
) -> dict[str, Any]:
    """Pooled OOS forecasts for the treatment and all four benchmarks, with per-row fold ids.

    HAR + LightGBM are refit per anchored-expanding fold (the frozen primitives). GARCH(1,1) is
    refit per fold on that fold's training returns and rolled forward through the fold's OOS days
    with frozen parameters (disclosed non-convergence fallbacks). Random-walk and EWMA are
    parameter-free causal transforms of the RV series.
    """
    n = mtm.n
    t0, t1 = mtm.t0, mtm.t1
    oos_start = np.datetime64(vcfg.walk_forward.oos_start)
    oos_mask = t0 >= oos_start
    if not oos_mask.any():
        raise ValueError(f"[{mtm.symbol}] no OOS rows at/after {vcfg.walk_forward.oos_start}")
    oos_all = np.flatnonzero(oos_mask)
    folds = [f for f in np.array_split(oos_all, vcfg.walk_forward.n_steps) if f.size]
    embargo = mtm.horizon - 1  # 0 at h = 1 (no forward-target overlap)

    pred_har = np.full(n, np.nan)
    pred_treat = np.full(n, np.nan)
    pred_garch = np.full(n, np.nan)
    fold_id = np.full(n, -1, dtype=int)
    scored = np.zeros(n, dtype=bool)
    garch_diag = {"folds": 0, "converged": 0, "fallback_carry": 0, "fallback_riskmetrics": 0}
    prev_params: dict[str, float] | None = None

    for fi, test_idx in enumerate(folds):
        first, last = int(test_idx[0]), int(test_idx[-1])
        candidates = np.arange(0, first)  # anchored: everything strictly before the fold
        train_idx = train_indices(t0, t1, test_idx, n, embargo, candidates=candidates)
        if train_idx.size < _MIN_TRAIN:
            continue
        # Arm B — HAR-RV OLS.
        coef = fit_har(mtm.x_har[train_idx], mtm.y[train_idx])
        pred_har[test_idx] = predict_har(coef, mtm.x_har[test_idx])
        # Arm T — fixed LightGBM (purged inner-val early stopping).
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
        pred_treat[test_idx] = est.predict(mtm.x_treat[test_idx])
        # GARCH(1,1) — refit on contiguous train returns, rolled forward (frozen params).
        r_train = returns[:first]
        params = fit_garch11(r_train)
        garch_diag["folds"] += 1
        if params["converged"]:
            garch_diag["converged"] += 1
            prev_params = params
        elif prev_params is not None:
            params = prev_params
            garch_diag["fallback_carry"] += 1
        else:
            params = riskmetrics_params(r_train)
            garch_diag["fallback_riskmetrics"] += 1
        garch_log = garch11_forecast_log(returns[: last + 1], params)
        pred_garch[test_idx] = garch_log[test_idx]

        fold_id[test_idx] = fi
        scored[test_idx] = True

    s = scored
    # Parameter-free causal benchmarks over the whole RV series, then sliced to scored rows.
    rw_log = rw_forecast_log(mtm.rv)
    ewma_log = ewma_forecast_log(mtm.rv, ewma_lambda)
    return {
        "idx": np.flatnonzero(s),
        "y_true": mtm.y[s],
        "fcast_treat": pred_treat[s],
        "fcast_har": pred_har[s],
        "fcast_rw": rw_log[s],
        "fcast_ewma": ewma_log[s],
        "fcast_garch": pred_garch[s],
        "fold_id": fold_id[s],
        "rv": mtm.rv[s],
        "t0": mtm.t0[s],
        "garch_diag": garch_diag,
    }


# --------------------------------------------------------------------------- #
# Primary (gated) horizon evaluation: the four-benchmark battery + G1..G4
# --------------------------------------------------------------------------- #
def evaluate_primary_h1(
    mtm: VolatilityMatrix,
    vcfg: VolatilityConfig,
    p23: Phase23Config,
    returns: np.ndarray,
    *,
    regime_full: np.ndarray,
) -> dict[str, Any]:
    """QLIKE/DM vs each benchmark, the regime split, and the four pre-registered gates at h = 1."""
    h = mtm.horizon
    oos = anchored_oos_h1(mtm, vcfg, returns, ewma_lambda=p23.benchmarks.ewma.lam)
    y = oos["y_true"]
    ql_treat = qlike_from_log(y, oos["fcast_treat"])
    mean_treat = float(np.mean(ql_treat))

    # --- per-benchmark QLIKE + one-sided DM (treatment better than the benchmark) ---
    bench_keys = {
        "har": "fcast_har",
        "random_walk": "fcast_rw",
        "ewma": "fcast_ewma",
        "garch": "fcast_garch",
    }
    benches: dict[str, Any] = {}
    for name, key in bench_keys.items():
        ql_b = qlike_from_log(y, oos[key])
        dm = diebold_mariano(ql_b, ql_treat, horizon=h, alpha=vcfg.dm.alpha)
        mean_b = float(np.mean(ql_b))
        benches[name] = {
            "qlike_bench": round(mean_b, 6),
            "qlike_treat": round(mean_treat, 6),
            "qlike_improvement": round(mean_b - mean_treat, 6),
            "dm_stat_hln": dm["dm_stat_hln"],
            "dm_p_value": dm["p_value"],
            "dm_significant": dm["significant"],
            "treat_beats": bool(mean_treat < mean_b and dm["significant"]),
        }

    # --- G1: accuracy vs HAR ---
    g1 = bool(benches["har"]["treat_beats"])

    # --- G2: regime robustness vs HAR (treat <= HAR in BOTH calm and turbulent) ---
    ql_har = qlike_from_log(y, oos["fcast_har"])
    oos_regime = regime_full[oos["idx"]]
    per_regime: dict[str, Any] = {}
    g2_parts: list[bool] = []
    for rname, mask in (("calm", ~oos_regime), ("turbulent", oos_regime)):
        if mask.any():
            mh, mt = float(np.mean(ql_har[mask])), float(np.mean(ql_treat[mask]))
            consistent = bool(mt <= mh)
            per_regime[rname] = {
                "n": int(mask.sum()),
                "qlike_har": round(mh, 6),
                "qlike_treat": round(mt, 6),
                "treat_le_har": consistent,
            }
            g2_parts.append(consistent)
        else:
            per_regime[rname] = {
                "n": 0,
                "qlike_har": None,
                "qlike_treat": None,
                "treat_le_har": False,
            }
            g2_parts.append(False)
    g2 = bool(len(g2_parts) == 2 and all(g2_parts))

    # --- G3: benchmark hardness (beat EACH configured challenger with a significant DM) ---
    challengers = list(p23.gates.g3_benchmark_hardness.challengers)
    g3 = bool(challengers) and all(benches[c]["treat_beats"] for c in challengers)

    # --- G4: temporal stability (beat HAR and RW in >= min_folds walk-forward folds) ---
    fold_id = oos["fold_id"]
    ql_rw = qlike_from_log(y, oos["fcast_rw"])
    uniq = np.unique(fold_id)
    beat_har = beat_rw = 0
    for f in uniq:
        m = fold_id == f
        if float(np.mean(ql_treat[m])) < float(np.mean(ql_har[m])):
            beat_har += 1
        if float(np.mean(ql_treat[m])) < float(np.mean(ql_rw[m])):
            beat_rw += 1
    g4cfg = p23.gates.g4_temporal_stability
    g4 = bool(beat_har >= g4cfg.min_folds_beating_har and beat_rw >= g4cfg.min_folds_beating_rw)

    candidate = bool(g1 and g2 and g3 and g4)
    return {
        "horizon": h,
        "n_oos": int(y.size),
        "qlike_treat": round(mean_treat, 6),
        "benchmarks": benches,
        "regime": per_regime,
        "fold_stability": {
            "n_folds": int(uniq.size),
            "treat_beats_har_folds": beat_har,
            "treat_beats_rw_folds": beat_rw,
            "min_folds_beating_har": g4cfg.min_folds_beating_har,
            "min_folds_beating_rw": g4cfg.min_folds_beating_rw,
        },
        "garch_diagnostics": oos["garch_diag"],
        "gates": {
            "g1_accuracy": g1,
            "g2_regime_robustness": g2,
            "g3_benchmark_hardness": g3,
            "g4_temporal_stability": g4,
        },
        "candidate": candidate,
    }


# --------------------------------------------------------------------------- #
# Per-symbol run
# --------------------------------------------------------------------------- #
def run_symbol_h1(
    symbol: str, p23: Phase23Config, *, store: DuckStore, interpret: bool = True
) -> dict[str, Any]:
    """Build the base; run the gated h=1 battery + the reported (non-gated) diagnostic horizons."""
    vcfg = p23.core
    fcfg = FeatureConfig.load()
    base: DailyBase = build_daily_base(symbol, vcfg, store=store)
    primary = vcfg.horizons.primary

    mtm = make_matrix(base, primary)
    returns = aligned_returns(symbol, mtm, fcfg.session, store)
    regime_full = regime_labels(mtm.rv, vcfg.regime.trailing_days)
    primary_res = evaluate_primary_h1(mtm, vcfg, p23, returns, regime_full=regime_full)
    primary_shap = treatment_shap(mtm, vcfg) if interpret else None

    g = primary_res["gates"]
    b = primary_res["benchmarks"]
    logger.info(
        f"[{symbol}] h={primary} n={primary_res['n_oos']} "
        f"QLIKE treat={primary_res['qlike_treat']} | beats "
        f"HAR={b['har']['treat_beats']} RW={b['random_walk']['treat_beats']} "
        f"EWMA={b['ewma']['treat_beats']} GARCH={b['garch']['treat_beats']} | "
        f"G1={g['g1_accuracy']} G2={g['g2_regime_robustness']} "
        f"G3={g['g3_benchmark_hardness']} G4={g['g4_temporal_stability']} "
        f"-> candidate={primary_res['candidate']}"
    )

    # Diagnostic horizons (reported, NOT gated) — reuse the Phase-21 HAR-vs-treat evaluator.
    diagnostics: dict[str, Any] = {}
    for hd in vcfg.horizons.diagnostic:
        mtd = make_matrix(base, hd)
        rf = regime_labels(mtd.rv, vcfg.regime.trailing_days)
        diagnostics[str(hd)] = evaluate_horizon(mtd, vcfg, regime_full=rf)

    return {
        "symbol": symbol,
        "phase": "23",
        "objective_used": vcfg.lgbm.objective,
        "sentiment_coverage": round(base.sentiment_coverage, 4),
        "n_incomplete_sessions": base.n_incomplete_sessions,
        "feature_blocks": base.feature_blocks,
        "primary_horizon": primary,
        "primary": primary_res,
        "candidate": primary_res["candidate"],
        "verdict": VERDICT_CONFIRMED if primary_res["candidate"] else VERDICT_NONE,
        "diagnostics_ungated": diagnostics,
        "treatment_shap": primary_shap,
        "cost_disclaimer": "forecast-skill verdict only (QLIKE accuracy vs a hardened benchmark "
        "battery at h=1); forecast skill is not tradeable money — authorizes no strategy/backtest/"
        "live trading.",
    }


# --------------------------------------------------------------------------- #
# Decision + persistence
# --------------------------------------------------------------------------- #
def decide_h1(symbol_results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Frozen rule: both symbols clear G1-G4 → confirmed; single → fragile; else honest null."""
    candidates = [s for s, r in symbol_results.items() if r["candidate"]]
    n = len(symbol_results)
    if n > 0 and len(candidates) == n:
        overall = "confirmed_1day_rv_forecast_skill"
        note = (
            "both symbols beat HAR-RV, random-walk, EWMA and GARCH(1,1) at h=1 (QLIKE, "
            "DM-significant), robust across regimes and walk-forward folds → authorizes ONLY the "
            "Phase-24 free-data incremental study; never live trading."
        )
    elif candidates:
        overall = "confirmed_1day_rv_forecast_skill_fragile"
        note = (
            f"single-symbol confirmation ({', '.join(candidates)}); the other failed at least one "
            "gate — flagged FRAGILE; no skill promoted."
        )
    else:
        overall = "no_significant_skill"
        note = (
            "the treatment did not clear the hardened h=1 battery on both symbols — an honest "
            "null. Re-scoped only by deliberate operator decision."
        )
    return {
        "per_symbol": {s: r["verdict"] for s, r in symbol_results.items()},
        "candidates": candidates,
        "fragile": bool(candidates) and len(candidates) < n,
        "overall": overall,
        "note": note,
    }


def _symbol_view(result: dict[str, Any]) -> dict[str, Any]:
    pr = result["primary"]
    return {
        "verdict": result["verdict"],
        "gates": pr["gates"],
        "n_oos": pr["n_oos"],
        "qlike_treat": pr["qlike_treat"],
        "benchmarks": pr["benchmarks"],
        "fold_stability": pr["fold_stability"],
        "garch_diagnostics": pr["garch_diagnostics"],
        "objective_used": result["objective_used"],
        "sentiment_coverage": result["sentiment_coverage"],
        "feature_blocks": result["feature_blocks"],
        "shap_top": (result.get("treatment_shap") or {}).get("top_features", [])[:8],
        "shap_block_shares": (result.get("treatment_shap") or {}).get("block_shares"),
    }


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


def _log_mlflow(result: dict[str, Any], p23: Phase23Config) -> None:
    """Best-effort MLflow logging (never fails the run)."""
    try:
        import os

        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
        import mlflow
    except Exception as exc:  # noqa: BLE001 - tracking is optional
        logger.warning(f"mlflow unavailable ({exc}); skipping tracking")
        return
    mlflow.set_tracking_uri((Settings().data_dir / "mlruns").as_uri())
    mlflow.set_experiment(p23.artifacts.get("mlflow_experiment", "volatility-forecast-h1"))
    pr = result["primary"]
    with mlflow.start_run(run_name=f"{result['symbol']}-{p23.core.volatility_version}"):
        mlflow.log_params(
            {
                "symbol": result["symbol"],
                "volatility_version": p23.core.volatility_version,
                "primary_horizon": result["primary_horizon"],
                "objective_used": result["objective_used"],
            }
        )
        mlflow.log_metrics(
            {
                "qlike_treat": float(pr["qlike_treat"]),
                **{f"qlike_{k}": float(v["qlike_bench"]) for k, v in pr["benchmarks"].items()},
                **{f"dm_p_{k}": float(v["dm_p_value"] or 1.0) for k, v in pr["benchmarks"].items()},
            }
        )
        for gate, val in pr["gates"].items():
            mlflow.set_tag(gate, str(val))
        mlflow.set_tag("verdict", result["verdict"])
        mlflow.log_dict(result, "result.json")


def run_volatility_h1(
    symbols: list[str] | None = None,
    *,
    interpret: bool = True,
    log_mlflow: bool = True,
    save: bool = True,
) -> dict[str, Any]:
    """Run the full Phase-23 confirmation verdict for all symbols → combined verdict (persisted)."""
    p23 = Phase23Config.load()
    vcfg = p23.core
    syms = list(dict.fromkeys(symbols or vcfg.symbols))
    is_full = set(syms) == set(vcfg.symbols)
    if save and not is_full:
        raise ValueError(
            f"Phase-23's combined verdict is pre-registered over {list(vcfg.symbols)}; refusing to "
            f"save a canonical decision over {syms}. Re-run with the exact set or pass save=False."
        )

    # Compute ALL symbols first; persist only once every symbol succeeds, so a mid-run failure can
    # never leave the canonical runs_h1/ dir with a fresh symbol JSON beside a stale verdict.json.
    store = DuckStore()
    symbol_results: dict[str, dict[str, Any]] = {}
    try:
        for symbol in syms:
            symbol_results[symbol] = run_symbol_h1(symbol, p23, store=store, interpret=interpret)
    finally:
        store.close()

    decision = decide_h1(symbol_results)
    verdict = {
        "phase": "23",
        "volatility_version": vcfg.volatility_version,
        "preregistration": "docs/PHASE23_PREREGISTRATION.md",
        "study": "1-day RV-forecast confirmation under a hardened benchmark battery",
        "benchmarks": [
            "HAR-RV",
            "random_walk",
            f"EWMA(lambda={p23.benchmarks.ewma.lam})",
            "GARCH(1,1)",
        ],
        "primary_horizon": vcfg.horizons.primary,
        "dm_alpha": vcfg.dm.alpha,
        "oos_start": vcfg.walk_forward.oos_start,
        "symbols": {s: _symbol_view(r) for s, r in symbol_results.items()},
        "decision": decision,
        "partial": not is_full,
        "cost_disclaimer": "forecast-skill verdict only; forecast skill is not tradeable money; "
        "authorizes no strategy/backtest/risk/execution/live trading.",
    }
    if save:
        # All-or-nothing canonical write: per-symbol files then the combined verdict, only after
        # the full run succeeded above.
        for r in symbol_results.values():
            save_symbol_run(r)
        save_verdict(verdict)
    if log_mlflow:
        for r in symbol_results.values():
            _log_mlflow(r, p23)
    return verdict


def _print_verdict(verdict: dict[str, Any]) -> None:
    logger.info("=== Phase-23 1-day RV-forecast confirmation verdict ===")
    for sym, v in verdict["symbols"].items():
        g = v["gates"]
        b = v["benchmarks"]
        logger.info(
            f"[{sym}] n={v['n_oos']} QLIKE treat={v['qlike_treat']} "
            f"(HAR={b['har']['qlike_bench']} RW={b['random_walk']['qlike_bench']} "
            f"EWMA={b['ewma']['qlike_bench']} GARCH={b['garch']['qlike_bench']}) "
            f"G1={g['g1_accuracy']} G2={g['g2_regime_robustness']} "
            f"G3={g['g3_benchmark_hardness']} G4={g['g4_temporal_stability']} -> {v['verdict']}"
        )
    d = verdict["decision"]
    logger.info(f"DECISION: {d['overall']} (candidates={d['candidates'] or 'none'}) — {d['note']}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="volatility.run_h1", description=__doc__)
    p.add_argument("--symbols", nargs="+", default=None, help="default: config symbols")
    p.add_argument("--no-mlflow", action="store_true", help="skip MLflow logging")
    p.add_argument("--no-interpret", action="store_true", help="skip SHAP interpretation")
    args = p.parse_args(argv)
    verdict = run_volatility_h1(
        symbols=args.symbols, interpret=not args.no_interpret, log_mlflow=not args.no_mlflow
    )
    _print_verdict(verdict)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
