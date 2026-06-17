"""Phase-21 volatility-forecast skill verdict — orchestrator.

For each symbol (MES, MNQ separately) and each horizon, this runs the pre-registered comparison::

    uv run python -m options_system.volatility.run --symbols MES MNQ

A fixed **HAR-RV benchmark** (arm B) and a single fixed regularized **LightGBM** (arm T) each
forecast the forward log-RV target through an **anchored expanding-window walk-forward** (train
anchors from history start; OOS from ``oos_start``; the ``h``-day target overlap is purged +
embargoed via the shared ``train_indices`` primitive — WalkForward's equal-block API cannot express
the date-anchored OOS start, so the leak-safe primitive it delegates to is used directly). The
verdict (primary horizon ``h = 5``, per symbol) is whether arm T's OOS **QLIKE** is significantly
lower than arm B's by a one-sided **Diebold-Mariano** test (HAC + HLN), robust across the calm/
turbulent regimes.

It is a **forecast-skill verdict only** — forecast skill is not tradeable money. It builds no
strategy, no economic backtest, no risk/execution path, and authorizes no live trading. Reads only
local lakes — no Databento, no IBKR, no network, no spend.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np

from config.settings import Settings

from ..common.logging import get_logger
from ..validation._purge import train_indices
from ..validation.forecast_stats import diebold_mariano, qlike_from_log
from .config import VolatilityConfig
from .dataset import DailyBase, VolatilityMatrix, build_daily_base, make_matrix
from .har import fit_har, predict_har
from .lgbm import build_vol_estimator, fit_vol_fold

logger = get_logger(__name__)

VERDICT_CANDIDATE = "skill candidate"
VERDICT_NONE = "no skill"
_MIN_TRAIN = 100  # minimum training rows for a walk-forward step to be scored


def _runs_dir():
    return Settings().data_dir / "volatility" / "runs"


# --------------------------------------------------------------------------- #
# Anchored expanding-window OOS forecasts (both arms)
# --------------------------------------------------------------------------- #
def anchored_oos_forecasts(mtm: VolatilityMatrix, vcfg: VolatilityConfig) -> dict[str, np.ndarray]:
    """Pooled OOS forecasts for arm B (HAR) and arm T (LightGBM) over the date-anchored WF.

    OOS rows are those with ``t0 >= oos_start``; they are split into ``n_steps`` contiguous folds,
    each refit on everything strictly before it (anchored expanding). Purge+embargo (``h-1`` days)
    on each fold's ``t1`` remove the forward-target overlap. Returns aligned OOS arrays.
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

    pred_har = np.full(n, np.nan)
    pred_treat = np.full(n, np.nan)
    scored = np.zeros(n, dtype=bool)
    for test_idx in folds:
        first = int(test_idx[0])
        candidates = np.arange(0, first)  # anchored: everything strictly before the fold
        train_idx = train_indices(t0, t1, test_idx, n, embargo, candidates=candidates)
        if train_idx.size < _MIN_TRAIN:
            continue
        # Arm B — HAR-RV OLS.
        coef = fit_har(mtm.x_har[train_idx], mtm.y[train_idx])
        pred_har[test_idx] = predict_har(coef, mtm.x_har[test_idx])
        # Arm T — fixed LightGBM with purged inner-val early stopping.
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
        scored[test_idx] = True

    s = scored
    return {
        "idx": np.flatnonzero(s),
        "y_true": mtm.y[s],
        "fcast_har": pred_har[s],
        "fcast_treat": pred_treat[s],
        "rv": mtm.rv[s],
        "t0": mtm.t0[s],
    }


# --------------------------------------------------------------------------- #
# Regime split (causal, leak-safe)
# --------------------------------------------------------------------------- #
def _trailing_mean(x: np.ndarray, window: int) -> np.ndarray:
    out = np.full(x.size, np.nan)
    csum = np.concatenate([[0.0], np.cumsum(x)])
    for i in range(x.size):
        if i + 1 >= window:
            out[i] = (csum[i + 1] - csum[i + 1 - window]) / window
    return out


def _causal_expanding_median(x: np.ndarray) -> np.ndarray:
    """Median of the non-NaN prefix x[:i+1] at each i (leak-safe; NaN until any value exists)."""
    out = np.full(x.size, np.nan)
    for i in range(x.size):
        prefix = x[: i + 1]
        valid = prefix[np.isfinite(prefix)]
        if valid.size:
            out[i] = float(np.median(valid))
    return out


def regime_labels(rv: np.ndarray, trailing_days: int) -> np.ndarray:
    """Turbulent (True) if the trailing-``trailing_days`` mean RV exceeds the causal expanding
    median of that trailing mean, else calm. Computed over the full (session-ordered) matrix rv."""
    tr = _trailing_mean(np.asarray(rv, dtype=float), trailing_days)
    med = _causal_expanding_median(tr)
    return np.asarray(tr > med) & np.isfinite(tr) & np.isfinite(med)


# --------------------------------------------------------------------------- #
# Per-horizon evaluation
# --------------------------------------------------------------------------- #
def _ols_calibration(y_true: np.ndarray, fcast: np.ndarray) -> dict[str, float]:
    """Mincer-Zarnowitz: regress realized on forecast → intercept + slope (unbiased ⇒ 0, 1)."""
    a = np.column_stack([np.ones(fcast.size), fcast])
    coef, _r, _rk, _sv = np.linalg.lstsq(a, y_true, rcond=None)
    return {"intercept": round(float(coef[0]), 5), "slope": round(float(coef[1]), 5)}


def evaluate_horizon(
    mtm: VolatilityMatrix, vcfg: VolatilityConfig, *, regime_full: np.ndarray
) -> dict[str, Any]:
    """Run the WF, compute QLIKE / DM / regime robustness / calibration for one horizon."""
    oos = anchored_oos_forecasts(mtm, vcfg)
    y_true, fc_har, fc_treat = oos["y_true"], oos["fcast_har"], oos["fcast_treat"]
    ql_har = qlike_from_log(y_true, fc_har)
    ql_treat = qlike_from_log(y_true, fc_treat)
    dm = diebold_mariano(ql_har, ql_treat, horizon=mtm.horizon, alpha=vcfg.dm.alpha)

    mean_ql_har = float(np.mean(ql_har))
    mean_ql_treat = float(np.mean(ql_treat))
    g1 = bool(mean_ql_treat < mean_ql_har and dm["significant"])

    # Regime robustness (G2): treatment <= HAR in BOTH calm and turbulent OOS sub-samples.
    oos_regime = regime_full[oos["idx"]]
    per_regime: dict[str, Any] = {}
    g2_parts: list[bool] = []
    for name, mask in (("calm", ~oos_regime), ("turbulent", oos_regime)):
        if mask.any():
            mh, mt = float(np.mean(ql_har[mask])), float(np.mean(ql_treat[mask]))
            consistent = bool(mt <= mh)
            per_regime[name] = {
                "n": int(mask.sum()),
                "qlike_har": round(mh, 6),
                "qlike_treat": round(mt, 6),
                "treat_le_har": consistent,
            }
            g2_parts.append(consistent)
        else:
            per_regime[name] = {
                "n": 0,
                "qlike_har": None,
                "qlike_treat": None,
                "treat_le_har": False,
            }
            g2_parts.append(False)  # an empty regime cannot confirm robustness
    g2 = bool(len(g2_parts) == 2 and all(g2_parts))

    rmse_har = float(np.sqrt(np.mean((y_true - fc_har) ** 2)))
    rmse_treat = float(np.sqrt(np.mean((y_true - fc_treat) ** 2)))
    mae_har = float(np.mean(np.abs(y_true - fc_har)))
    mae_treat = float(np.mean(np.abs(y_true - fc_treat)))

    return {
        "horizon": mtm.horizon,
        "n_oos": int(y_true.size),
        "qlike_har": round(mean_ql_har, 6),
        "qlike_treat": round(mean_ql_treat, 6),
        "qlike_improvement": round(mean_ql_har - mean_ql_treat, 6),
        "dm": dm,
        "g1_accuracy": g1,
        "g2_regime_robustness": g2,
        "regime": per_regime,
        "calibration_treat": _ols_calibration(y_true, fc_treat),
        "calibration_har": _ols_calibration(y_true, fc_har),
        "rmse_har": round(rmse_har, 6),
        "rmse_treat": round(rmse_treat, 6),
        "mae_har": round(mae_har, 6),
        "mae_treat": round(mae_treat, 6),
    }


# --------------------------------------------------------------------------- #
# SHAP for arm T (interpretation only)
# --------------------------------------------------------------------------- #
def treatment_shap(mtm: VolatilityMatrix, vcfg: VolatilityConfig) -> dict[str, Any]:
    """Global SHAP for the full-data arm-T fit (interpretation only; never a performance number).

    Reports per-feature mean ``|SHAP|`` and the share contributed by each feature block (HAR vs
    price vs macro vs sentiment), so the report can say whether the extra blocks add over the HAR
    lags. NEVER raises: any failure is surfaced as ``{"available": False, ...}``.
    """
    try:
        import shap

        est = build_vol_estimator(vcfg)
        est.early_stopping_rounds = None  # full-data fit, no inner split
        est.fit(mtm.x_treat, mtm.y)
        rng = np.random.default_rng(vcfg.seed)
        n = mtm.n
        sel = np.arange(n) if n <= 2000 else np.sort(rng.choice(n, 2000, replace=False))
        x_bg = mtm.x_treat[sel]
        values = shap.TreeExplainer(est._model).shap_values(x_bg)
        arr = np.asarray(values)
        mean_abs = np.abs(arr).mean(axis=0) if arr.ndim == 2 else np.abs(arr).mean(axis=(0, 2))
        total = float(mean_abs.sum()) or 1.0
        cols = mtm.treat_cols
        order = np.argsort(mean_abs)[::-1]
        importances = [
            {
                "feature": cols[i],
                "mean_abs_shap": round(float(mean_abs[i]), 6),
                "share": round(float(mean_abs[i] / total), 4),
            }
            for i in order
        ]

        def _block(prefixes: tuple[str, ...], exact: tuple[str, ...] = ()) -> float:
            return round(
                float(
                    sum(
                        mean_abs[i] / total
                        for i in range(len(cols))
                        if cols[i] in exact or cols[i].startswith(prefixes)
                    )
                ),
                4,
            )

        har_share = _block((), exact=tuple(mtm.har_cols))
        sent_share = _block(("sent_",))
        mkt_share = _block(("mkt_",))  # Phase-22 x1 cross-asset block (0 when off)
        gkg_share = _block(("gkgtone_",))  # Phase-22 s3 GKG-tone block (0 when off)
        accounted = har_share + sent_share + mkt_share + gkg_share
        block_shares = {
            "har": har_share,
            "sentiment": sent_share,
            "marketdata": mkt_share,
            "gkg": gkg_share,
            "other": round(max(0.0, 1.0 - accounted), 4),
        }
        return {
            "available": True,
            "n_background": int(x_bg.shape[0]),
            "top_features": [d["feature"] for d in importances[:12]],
            "importances": importances[:30],
            "block_shares": block_shares,
        }
    except Exception as exc:  # noqa: BLE001 - interpretation is optional, never fail the run
        logger.warning(f"volatility SHAP unavailable ({exc}); continuing without it")
        return {"available": False, "reason": str(exc)}


# --------------------------------------------------------------------------- #
# Per-symbol run
# --------------------------------------------------------------------------- #
def run_symbol(
    symbol: str, vcfg: VolatilityConfig, *, interpret: bool = True, rebuild_cache: bool = False
) -> dict[str, Any]:
    """Build the daily base, run every horizon, and assemble the per-symbol result + gates."""
    base: DailyBase = build_daily_base(symbol, vcfg)
    primary = vcfg.horizons.primary
    horizons = vcfg.horizons.all

    per_horizon: dict[str, Any] = {}
    primary_shap: dict[str, Any] | None = None
    for h in horizons:
        mtm = make_matrix(base, h)
        regime_full = regime_labels(mtm.rv, vcfg.regime.trailing_days)
        res = evaluate_horizon(mtm, vcfg, regime_full=regime_full)
        per_horizon[str(h)] = res
        logger.info(
            f"[{symbol}] h={h} n_oos={res['n_oos']} QLIKE har={res['qlike_har']} "
            f"treat={res['qlike_treat']} DM_p={res['dm']['p_value']} "
            f"G1={res['g1_accuracy']} G2={res['g2_regime_robustness']}"
        )
        if h == primary and interpret:
            primary_shap = treatment_shap(mtm, vcfg)

    pr = per_horizon[str(primary)]
    g1 = bool(pr["g1_accuracy"])
    g2 = bool(pr["g2_regime_robustness"])
    candidate = g1 and g2
    return {
        "symbol": symbol,
        "objective_used": vcfg.lgbm.objective,
        "sentiment_coverage": round(base.sentiment_coverage, 4),
        "n_incomplete_sessions": base.n_incomplete_sessions,
        "feature_blocks": base.feature_blocks,
        "primary_horizon": primary,
        "gates": {"g1_accuracy": g1, "g2_regime_robustness": g2},
        "verdict": VERDICT_CANDIDATE if candidate else VERDICT_NONE,
        "candidate": candidate,
        "horizons": per_horizon,
        "treatment_shap": primary_shap,
        "cost_disclaimer": "forecast-skill verdict only (QLIKE accuracy vs HAR-RV); forecast skill "
        "is not tradeable money — authorizes no strategy/backtest/live trading.",
    }


# --------------------------------------------------------------------------- #
# Decision + persistence
# --------------------------------------------------------------------------- #
def decide(symbol_results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Frozen decision rule: both symbols clear G1+G2 → candidate (Phase 22 study only); single →
    fragile; either fails → no candidate (honest null)."""
    candidates = [s for s, r in symbol_results.items() if r["candidate"]]
    n = len(symbol_results)
    fragile = bool(candidates) and len(candidates) < n
    if len(candidates) == n and n > 0:
        overall = "volatility_forecast_skill_candidate"
        note = (
            "both symbols beat HAR-RV (QLIKE, DM-significant, regime-robust) → authorizes ONLY a "
            "future Phase 22 economic-value study (realistic costs); never live trading."
        )
    elif candidates:
        overall = "volatility_forecast_skill_candidate_fragile"
        note = (
            f"single-symbol candidate ({', '.join(candidates)}) with the other failing — flagged "
            "FRAGILE; authorizes ONLY a Phase 22 economic-value study for the passing symbol."
        )
    else:
        overall = "no_significant_skill"
        note = (
            "ML did not beat HAR-RV on these instruments/features — an honest, informative null "
            "(the benchmark is hard to beat here). Re-scoped only by deliberate operator decision."
        )
    return {
        "per_symbol": {s: r["verdict"] for s, r in symbol_results.items()},
        "candidates": candidates,
        "fragile": fragile,
        "overall": overall,
        "note": note,
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


def read_verdict() -> dict[str, Any] | None:
    path = _runs_dir() / "verdict.json"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _symbol_view(result: dict[str, Any]) -> dict[str, Any]:
    """Compact per-symbol slice for the combined verdict file."""
    pr = result["horizons"][str(result["primary_horizon"])]
    return {
        "verdict": result["verdict"],
        "gates": result["gates"],
        "objective_used": result["objective_used"],
        "sentiment_coverage": result["sentiment_coverage"],
        "feature_blocks": result["feature_blocks"],
        "primary": {
            "horizon": pr["horizon"],
            "n_oos": pr["n_oos"],
            "qlike_har": pr["qlike_har"],
            "qlike_treat": pr["qlike_treat"],
            "dm_stat_hln": pr["dm"]["dm_stat_hln"],
            "dm_p_value": pr["dm"]["p_value"],
            "regime": pr["regime"],
            "calibration_treat": pr["calibration_treat"],
        },
        "shap_top": (result.get("treatment_shap") or {}).get("top_features", [])[:8],
        "shap_block_shares": (result.get("treatment_shap") or {}).get("block_shares"),
    }


def run_volatility(
    symbols: list[str] | None = None,
    *,
    interpret: bool = True,
    rebuild_cache: bool = False,
    log_mlflow: bool = True,
    save: bool = True,
) -> dict[str, Any]:
    """Run the full Phase-21 verdict for all symbols → the combined verdict (and persist it)."""
    vcfg = VolatilityConfig.load()
    syms = list(dict.fromkeys(symbols or vcfg.symbols))
    is_full = set(syms) == set(vcfg.symbols)
    if save and not is_full:
        raise ValueError(
            f"Phase-21's combined verdict is pre-registered over {list(vcfg.symbols)}; refusing to "
            f"save a canonical decision over {syms}. Re-run with the exact pre-registered set, or "
            "pass save=False for a diagnostic (non-canonical) run."
        )

    symbol_results: dict[str, dict[str, Any]] = {}
    for symbol in syms:
        result = run_symbol(symbol, vcfg, interpret=interpret, rebuild_cache=rebuild_cache)
        symbol_results[symbol] = result
        if save:
            save_symbol_run(result)
        if log_mlflow:
            _log_mlflow(result, vcfg)

    decision = decide(symbol_results)
    verdict = {
        "phase": "21",
        "volatility_version": vcfg.volatility_version,
        "preregistration": "docs/PHASE21_PREREGISTRATION.md",
        "benchmark": "HAR-RV (Corsi 2009) on log-RV",
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
        save_verdict(verdict)
    return verdict


def _log_mlflow(result: dict[str, Any], vcfg: VolatilityConfig) -> None:
    """Best-effort MLflow logging to a local file store (never fails the run)."""
    try:
        import os

        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
        import mlflow
    except Exception as exc:  # noqa: BLE001 - tracking is optional
        logger.warning(f"mlflow unavailable ({exc}); skipping tracking")
        return
    mlflow.set_tracking_uri((Settings().data_dir / "mlruns").as_uri())
    mlflow.set_experiment("volatility-forecast")
    pr = result["horizons"][str(result["primary_horizon"])]
    with mlflow.start_run(run_name=f"{result['symbol']}-{vcfg.volatility_version}"):
        mlflow.log_params(
            {
                "symbol": result["symbol"],
                "volatility_version": vcfg.volatility_version,
                "objective_used": result["objective_used"],
                "primary_horizon": result["primary_horizon"],
                "oos_start": vcfg.walk_forward.oos_start,
            }
        )
        mlflow.log_metrics(
            {
                "qlike_har": float(pr["qlike_har"]),
                "qlike_treat": float(pr["qlike_treat"]),
                "dm_p_value": float(pr["dm"]["p_value"]),
                "sentiment_coverage": float(result["sentiment_coverage"]),
            }
        )
        mlflow.set_tag("verdict", result["verdict"])
        mlflow.log_dict(result, "result.json")


def _print_verdict(verdict: dict[str, Any]) -> None:
    logger.info("=== Phase-21 volatility-forecast verdict ===")
    for sym, v in verdict["symbols"].items():
        p, g = v["primary"], v["gates"]
        logger.info(
            f"[{sym}] h={p['horizon']} n={p['n_oos']} QLIKE har={p['qlike_har']} "
            f"treat={p['qlike_treat']} DM_p={p['dm_p_value']} "
            f"G1={g['g1_accuracy']} G2={g['g2_regime_robustness']} -> {v['verdict']}"
        )
    d = verdict["decision"]
    logger.info(f"DECISION: {d['overall']} (candidates={d['candidates'] or 'none'}) — {d['note']}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="volatility.run", description=__doc__)
    p.add_argument("--symbols", nargs="+", default=None, help="default: config volatility.symbols")
    p.add_argument("--no-mlflow", action="store_true", help="skip MLflow logging")
    p.add_argument("--no-interpret", action="store_true", help="skip SHAP interpretation")
    p.add_argument("--rebuild-cache", action="store_true", help="(reserved) rebuild caches")
    args = p.parse_args(argv)
    verdict = run_volatility(
        symbols=args.symbols,
        interpret=not args.no_interpret,
        rebuild_cache=args.rebuild_cache,
        log_mlflow=not args.no_mlflow,
    )
    _print_verdict(verdict)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
