"""Phase-25 — economic-value verdict for the confirmed 1-day RV forecast (offline / paper only).

Frozen contract: ``docs/PHASE25_PREREGISTRATION.md`` / ``config/phase25_econ.yaml``. Run::

    uv run python -m options_system.volatility.run_econ --symbols MES MNQ

Does the **confirmed Phase-23 h = 1 volatility forecast** translate into **economically-meaningful,
cost-robust, direction-free value** — net of realistic *and* 3×-stressed costs — over a no-timing
baseline and every econometric benchmark, per symbol? The forecasts are reused **verbatim**
(deterministically regenerated under the frozen Phase-23 config + seed 7, fingerprint/QLIKE-verified
fail-closed, never refit); the economic layer adds **zero fitted parameters**.

Two co-required legs: a **return-free volatility-targeting-error** leg (G1/G2) and the
**Fleming-Kirby-Ostdiek performance-fee** money leg (G3-G8), the latter **leverage-matched** to the
no-timing ``STATIC`` baseline so a win cannot be a larger-average-long risk-premium bet; an
**E9 directional-leakage VOID gate** on the active overweight closes the back door to the program's
six confirmed directional nulls. This is the **integrity core** of the study.

Economic-value verdict in offline simulation only — authorizes NO live trading and NO automated
paper-execution deployment, regardless of outcome. Reads only local lakes; no Databento / IBKR /
network / spend (``OPTIONS_DATABENTO_SPEND_OK`` stays unset).
"""

from __future__ import annotations

import argparse
import hashlib
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
from . import econ
from .benchmarks import (
    ewma_forecast_log,
    fit_garch11,
    garch11_forecast_log,
    riskmetrics_params,
    rw_forecast_log,
)
from .config import VolatilityConfig
from .config_econ import Phase25Config
from .dataset import DailyBase, VolatilityMatrix, build_daily_base, make_matrix
from .har import fit_har, predict_har
from .lgbm import build_vol_estimator, fit_vol_fold
from .realized import daily_rth_log_return
from .run import _MIN_TRAIN, regime_labels
from .run_h1 import aligned_returns

logger = get_logger(__name__)

_WIDE_START = datetime(2000, 1, 1, tzinfo=UTC)
_WIDE_END = datetime(2100, 1, 1, tzinfo=UTC)

# Arm taxonomy (frozen).
FCAST_ARMS = ("treat", "har", "rw", "ewma", "garch")  # the five forecast sources
BENCH_ARMS = ("har", "rw", "ewma", "garch")  # the four econometric benchmarks
GATED_VS = ("static", "har", "rw", "ewma", "garch")  # arms TREAT is compared against
_FCAST_KEY = {
    "treat": "fcast_treat",
    "har": "fcast_har",
    "rw": "fcast_rw",
    "ewma": "fcast_ewma",
    "garch": "fcast_garch",
}

VERDICT_CONFIRMED = "confirmed_offline_economic_value_1day_vol_forecast"
VERDICT_FRAGILE = "fragile_single_symbol"
VERDICT_NULL = "no_costed_direction_free_economic_value"
VERDICT_VOID = "voided_directional_leakage"

# The FROZEN Phase-23 OOS-frame shape — the structural reproduction anchors the per-arm QLIKE pins
# do NOT cover (fold-partition count, frame length, the GARCH-fallback structure). These are facts of
# the already-committed Phase-23 pipeline (``data/volatility/runs_h1/verdict.json``, the 18-fold
# anchored walk-forward of ``docs/PHASE23_PREREGISTRATION.md``), fixed BEFORE any Phase-25 economic
# computation — determinism anchors, not snooped Phase-25 results. They are enforced on EVERY run
# (including the first), so a QLIKE-preserving drift in the fold ids / GARCH fallback / frame length
# cannot be silently blessed and written as a new baseline.
_EXPECTED_N_FOLDS = 18
_FROZEN_FRAME: dict[str, dict[str, Any]] = {
    "MES": {
        "n_oos": 1139,
        "garch_diag": {
            "folds": 18,
            "converged": 17,
            "fallback_carry": 1,
            "fallback_riskmetrics": 0,
        },
    },
    "MNQ": {
        "n_oos": 1139,
        "garch_diag": {
            "folds": 18,
            "converged": 18,
            "fallback_carry": 0,
            "fallback_riskmetrics": 0,
        },
    },
}


def _runs_dir():
    return Settings().data_dir / "volatility" / "runs_ev"


# --------------------------------------------------------------------------- #
# Forecast regeneration (mirrors run_h1.anchored_oos_h1) + per-fold training weights
# --------------------------------------------------------------------------- #
def _econ_forecasts(
    mtm: VolatilityMatrix,
    vcfg: VolatilityConfig,
    returns: np.ndarray,
    *,
    ewma_lambda: float,
    gamma: float,
    w_cap: float,
    mu_floor: float,
) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    """Regenerate the frozen Phase-23 OOS forecasts AND the per-fold training-window diagnostics.

    The anchored-expanding walk-forward, the leak-safe purge, the per-fold HAR/LightGBM/GARCH fits,
    and the parameter-free RW/EWMA transforms are **identical** to ``run_h1.anchored_oos_h1`` (so
    OOS forecast arrays reproduce byte-for-byte — verified by the QLIKE pins). The only addition is
    that, at each fold, this also evaluates every arm's **in-sample (training-row)** forecast to
    form the causal per-fold ``μ_bar`` / ``σ²_bar`` / ``STATIC`` weight and the matching constants
    (matching each arm's training-window mean MV weight to ``STATIC``'s).
    """
    n = mtm.n
    t0, t1 = mtm.t0, mtm.t1
    oos_start = np.datetime64(vcfg.walk_forward.oos_start)
    oos_mask = t0 >= oos_start
    if not oos_mask.any():
        raise ValueError(f"[{mtm.symbol}] no OOS rows at/after {vcfg.walk_forward.oos_start}")
    oos_all = np.flatnonzero(oos_mask)
    folds = [f for f in np.array_split(oos_all, vcfg.walk_forward.n_steps) if f.size]
    embargo = mtm.horizon - 1  # 0 at h = 1

    pred_har = np.full(n, np.nan)
    pred_treat = np.full(n, np.nan)
    pred_garch = np.full(n, np.nan)
    fold_id = np.full(n, -1, dtype=int)
    scored = np.zeros(n, dtype=bool)
    garch_diag = {"folds": 0, "converged": 0, "fallback_carry": 0, "fallback_riskmetrics": 0}
    prev_params: dict[str, float] | None = None

    # parameter-free causal benchmarks over the whole RV series (identical to anchored_oos_h1).
    rw_log = rw_forecast_log(mtm.rv)
    ewma_log = ewma_forecast_log(mtm.rv, ewma_lambda)

    fold_meta: dict[int, dict[str, Any]] = {}

    for fi, test_idx in enumerate(folds):
        first, last = int(test_idx[0]), int(test_idx[-1])
        candidates = np.arange(0, first)
        train_idx = train_indices(t0, t1, test_idx, n, embargo, candidates=candidates)
        if train_idx.size < _MIN_TRAIN:
            continue

        coef = fit_har(mtm.x_har[train_idx], mtm.y[train_idx])
        pred_har[test_idx] = predict_har(coef, mtm.x_har[test_idx])
        har_train = predict_har(coef, mtm.x_har[train_idx])

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
        treat_train = est.predict(mtm.x_treat[train_idx])

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
        garch_train = garch_log[train_idx]

        # --- per-fold causal scalars (training rows only) ---
        train_ret = returns[train_idx]  # same-session RTH return r_t on training rows
        m_train = float(np.mean(train_ret)) if train_ret.size else 0.0
        mu_bar = m_train if m_train > 0.0 else mu_floor
        sigma2_bar = float(np.mean(mtm.rv[train_idx]))
        w_static_mv = float(np.clip(mu_bar / (gamma * sigma2_bar), 0.0, w_cap))

        # leverage-matching constant per arm: c = w_static / mean(training-window MV weight).
        train_f = {
            "treat": treat_train,
            "har": har_train,
            "rw": rw_log[train_idx],
            "ewma": ewma_log[train_idx],
            "garch": garch_train,
        }
        c_match: dict[str, float] = {"static": 1.0}
        wbar_train: dict[str, float] = {"static": w_static_mv}
        for arm, f_tr in train_f.items():
            w_tr = econ.mean_variance_weight(mu_bar, gamma, np.exp(f_tr), w_cap)
            wb = float(np.mean(w_tr)) if w_tr.size else 0.0
            wbar_train[arm] = wb
            c_match[arm] = econ.leverage_match_constant(wb, w_static_mv)

        fold_meta[fi] = {
            "mu_bar": mu_bar,
            "mu_floored": bool(m_train <= 0.0),
            "sigma2_bar": sigma2_bar,
            "w_static_mv": w_static_mv,
            "c_match": c_match,
            "wbar_train": wbar_train,
            "n_train": int(train_idx.size),
        }
        fold_id[test_idx] = fi
        scored[test_idx] = True

    s = scored
    oos = {
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
    return oos, fold_meta


# --------------------------------------------------------------------------- #
# Fail-closed reproduction guard (the frozen anchor)
# --------------------------------------------------------------------------- #
def _assert_frozen_frame_shape(
    symbol: str, n_oos: int, n_folds: int, garch_diag: dict[str, Any]
) -> None:
    """Fail closed unless the regenerated frame's SHAPE matches the frozen Phase-23 anchors.

    Pins the dimensions the per-arm QLIKE comparison does **not** cover — the frame length, the fold
    count, and the GARCH-fallback structure — so a QLIKE-preserving drift in any of them cannot be
    silently accepted on the first (or any) run. Enforced before the fingerprint is ever written.
    """
    exp = _FROZEN_FRAME.get(symbol)
    if exp is None:
        raise ValueError(f"[{symbol}] no frozen Phase-23 frame anchor — cannot verify reproduction")
    if n_oos != exp["n_oos"]:
        raise ValueError(
            f"[{symbol}] reproduction guard FAILED: n_oos {n_oos} != frozen {exp['n_oos']} — "
            "the regenerated OOS frame does not match the frozen Phase-23 frame; fail closed."
        )
    if n_folds != _EXPECTED_N_FOLDS:
        raise ValueError(
            f"[{symbol}] reproduction guard FAILED: {n_folds} folds != frozen {_EXPECTED_N_FOLDS}."
        )
    if garch_diag != exp["garch_diag"]:
        raise ValueError(
            f"[{symbol}] reproduction guard FAILED: GARCH diagnostics {garch_diag} != frozen "
            f"{exp['garch_diag']} — a GARCH-fallback drift; fail closed."
        )


def _verify_reproduction(symbol: str, oos: dict[str, Any], p25: Phase25Config) -> dict[str, Any]:
    """Assert the regenerated forecasts reproduce the frozen Phase-23 result; fail closed otherwise.

    Two fail-closed layers, both enforced on EVERY run (including the first, before any artifact is
    written): (1) the per-arm OOS QLIKE pins from ``reproduction_guard.expected_qlike`` — treatment
    byte-exact (6 dp), each benchmark to 3 dp; (2) the structural anchors the QLIKE pins do not cover
    — the frame length, the 18-fold partition, and the GARCH-fallback diagnostics (``_FROZEN_FRAME``).
    Treatment QLIKE alone is insufficient (a benchmark / fold / GARCH-fallback / return drift could
    flip the verdict while leaving treatment QLIKE unchanged), so both layers pin every dimension that
    has a frozen Phase-23 reference; the per-day ``next_rth_log_return`` (no frozen scalar reference)
    is guarded by the forward-shift alignment test, the ≥0.99 coverage assert, and the persisted
    cross-run frame fingerprint. Any mismatch raises.
    """
    y = oos["y_true"]
    exp = p25.reproduction_guard.expected_qlike
    observed: dict[str, float] = {}
    for arm in FCAST_ARMS:
        ql = float(np.mean(qlike_from_log(y, oos[_FCAST_KEY[arm]])))
        observed[arm] = round(ql, 6)
        want = exp[arm][symbol]
        # treatment pinned byte-exact (6 dp); benchmarks to the contract's 3-dp precision.
        ndp = 6 if arm == "treat" else 3
        if round(ql, ndp) != round(float(want), ndp):
            raise ValueError(
                f"[{symbol}] reproduction guard FAILED: {arm} QLIKE {ql:.6f} != pinned "
                f"{want} (to {ndp} dp) — refusing to run Phase-25 on drifted forecasts."
            )
    n_folds = int(np.unique(oos["fold_id"]).size)
    _assert_frozen_frame_shape(symbol, int(y.size), n_folds, oos["garch_diag"])
    return {
        "qlike": observed,
        "garch_diagnostics": oos["garch_diag"],
        "n_oos": int(y.size),
        "n_folds": n_folds,
    }


def _frame_fingerprint(oos: dict[str, Any], r_next: np.ndarray) -> str:
    """SHA-256 over the full regenerated OOS frame — auditable, and a cross-run drift guard.

    Covers every ``fcast_*`` series, the per-row ``fold_id`` / ``t0`` / ``rv`` / ``y``, the aligned
    ``next_rth_log_return``, and the GARCH diagnostics. Persisted so a later re-run that drifts in
    *any* of these (not just treatment QLIKE) is caught by comparison against the stored value.
    """
    h = hashlib.sha256()
    for key in (
        "fcast_treat",
        "fcast_har",
        "fcast_rw",
        "fcast_ewma",
        "fcast_garch",
        "y_true",
        "rv",
    ):
        h.update(np.round(np.asarray(oos[key], dtype=float), 10).tobytes())
    h.update(np.asarray(oos["fold_id"], dtype=np.int64).tobytes())
    h.update(np.asarray(oos["t0"]).astype("datetime64[ns]").astype(np.int64).tobytes())
    h.update(np.round(np.asarray(r_next, dtype=float), 10).tobytes())
    h.update(json.dumps(oos["garch_diag"], sort_keys=True).encode())
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# Next-session RTH return (the t+1 alignment) — the held-period P&L return
# --------------------------------------------------------------------------- #
def _next_rth_returns(
    symbol: str, base: DailyBase, mtm: VolatilityMatrix, scfg: Any, store: DuckStore
) -> np.ndarray:
    """``r_{t+1}`` for every matrix row: the RTH return of the session **after** the forecast row.

    Built on the FULL daily session universe (``base.frame``, which includes the trailing session a
    matrix row's forward target dropped) so the last forecast row maps to its real ``t+1`` return.
    A one-session forward shift of the same-session RTH return, mapped to the matrix rows by session
    date. The close-set weight on row ``t`` earns this ``t+1`` return — not the same-session ``r_t``
    (that would multiply a close-set weight by an already-realized return / look-ahead leak).
    """
    bars = store.get_bars(symbol, _WIDE_START, _WIDE_END, freq="1m", continuous=True)
    ret_df = daily_rth_log_return(bars, scfg).with_columns(pl.col("session_date").cast(pl.Date))
    full = (
        base.frame.select(pl.col("session_date").cast(pl.Date))
        .sort("session_date")
        .join(ret_df, on="session_date", how="left")
        .sort("session_date")
    )
    return _map_next_returns(full["session_date"], full["ret"], pl.Series(mtm.session_date))


def _map_next_returns(
    full_sessions: pl.Series, same_session_ret: pl.Series, matrix_sessions: pl.Series
) -> np.ndarray:
    """Pure session→t+1 mapping: each matrix row gets the NEXT full-session's same-session RTH return.

    Forward-shifts the same-session returns over the full (sorted) session universe, then maps each
    matrix row's session date to its successor's return by a left join (left-order preserved via a
    row index). A matrix session whose successor is absent (the trailing session) yields NaN. This is
    the integration of :func:`econ.forward_shift_one` over the real per-symbol session sequence — it
    is what proves forecast row ``t`` is scored against session ``t+1`` only, never the same-session
    ``r_t``.
    """
    full = pl.DataFrame(
        {"session_date": full_sessions.cast(pl.Date), "ret": same_session_ret}
    ).sort("session_date")
    next_full = econ.forward_shift_one(full["ret"].to_numpy().astype(float))  # next[i] = same[i+1]
    mapper = pl.DataFrame({"session_date": full["session_date"], "next_ret": pl.Series(next_full)})
    rows = (
        pl.DataFrame({"session_date": matrix_sessions.cast(pl.Date)})
        .with_row_index("_i")
        .join(mapper, on="session_date", how="left")
        .sort("_i")
    )
    return rows["next_ret"].to_numpy().astype(float)


# --------------------------------------------------------------------------- #
# Weight assembly (per OOS row, every arm, both legs)
# --------------------------------------------------------------------------- #
def _assemble_weights(
    oos: dict[str, Any], fold_meta: dict[int, dict[str, Any]], p25: Phase25Config
) -> dict[str, Any]:
    """Per-arm OOS weights: raw MV, leverage-matched MV (money leg), and vol-target (VTE leg)."""
    o = p25.overlay
    gamma, w_cap, w_floor = o.gamma, o.w_cap, o.w_floor
    sig_td = o.sigma_target_daily
    fid = oos["fold_id"]

    # per-row causal scalars from the fold each OOS row belongs to.
    mu_bar = np.array([fold_meta[f]["mu_bar"] for f in fid], dtype=float)
    sigma2_bar = np.array([fold_meta[f]["sigma2_bar"] for f in fid], dtype=float)

    # per-row forecast variance for every arm (STATIC = the causal per-fold unconditional variance).
    sigma2: dict[str, np.ndarray] = {"static": sigma2_bar}
    for arm in FCAST_ARMS:
        sigma2[arm] = np.exp(np.asarray(oos[_FCAST_KEY[arm]], dtype=float))

    mv_raw: dict[str, np.ndarray] = {}
    mv_matched: dict[str, np.ndarray] = {}
    vt: dict[str, np.ndarray] = {}
    for arm in ("static", *FCAST_ARMS):
        s2 = sigma2[arm]
        mv_raw[arm] = econ.mean_variance_weight(mu_bar, gamma, s2, w_cap)
        c = np.array([fold_meta[f]["c_match"][arm] for f in fid], dtype=float)
        mv_matched[arm] = np.clip(c * mv_raw[arm], 0.0, w_cap)
        vt[arm] = econ.vol_target_weight(sig_td, np.sqrt(s2), w_floor, w_cap)
    return {
        "mv_raw": mv_raw,
        "mv_matched": mv_matched,
        "vt": vt,
        "mu_bar": mu_bar,
        "sigma2_bar": sigma2_bar,
    }


# --------------------------------------------------------------------------- #
# Money-leg net returns + utilities at a given cost stress
# --------------------------------------------------------------------------- #
def _net_returns(
    weight: np.ndarray, r_next: np.ndarray, c_side: float, stress: float
) -> np.ndarray:
    """``net = w·r_{t+1} − 2·c_side·w·stress`` — full daily round-trip cost on the held position."""
    w = np.asarray(weight, dtype=float)
    gross = w * np.asarray(r_next, dtype=float)
    cost = 2.0 * c_side * w * stress
    return gross - cost


def _utilities(
    mv_matched: dict[str, np.ndarray],
    r_next: np.ndarray,
    c_side: float,
    stress: float,
    gamma: float,
) -> dict[str, np.ndarray]:
    """Per-day net realized utilities for every arm at one cost stress (matched money weights)."""
    return {
        arm: econ.realized_utility(_net_returns(mv_matched[arm], r_next, c_side, stress), gamma)
        for arm in ("static", *FCAST_ARMS)
    }


def _fee_vs(
    mv_matched: dict[str, np.ndarray],
    r_next: np.ndarray,
    c_side: float,
    stress: float,
    gamma: float,
    k: str,
    ann: int,
) -> float | None:
    """Annualized FKO fee (bps/yr) TREAT vs arm ``k`` at one stress; ``None`` on non-convergence."""
    net_t = _net_returns(mv_matched["treat"], r_next, c_side, stress)
    net_k = _net_returns(mv_matched[k], r_next, c_side, stress)
    try:
        return econ.fee_bps_per_year(econ.performance_fee(net_t, net_k, gamma), ann)
    except econ.FeeNonConvergence:
        return None


# --------------------------------------------------------------------------- #
# The nine gates (per symbol)
# --------------------------------------------------------------------------- #
def _evaluate_gates(
    oos: dict[str, Any],
    weights: dict[str, Any],
    r_next: np.ndarray,
    regime_oos: np.ndarray,
    *,
    p25: Phase25Config,
    c_side: float,
) -> dict[str, Any]:
    """Compute G1-G8 + the E9 void on the assembled weights/returns for one symbol."""
    o = p25.overlay
    g = p25.gates
    sig = p25.significance
    gamma, ann = o.gamma, o.ann_factor
    sig_td = o.sigma_target_daily
    mvm = weights["mv_matched"]
    vt = weights["vt"]
    fid = oos["fold_id"]
    folds = np.unique(fid)
    base_cost, stress_cost = p25.costs.base_and_stress()

    # ---------------- LEG 1: volatility-targeting error (return-free) ----------------
    vte_arm = {arm: econ.vte(vt[arm], r_next, sig_td) for arm in ("static", *FCAST_ARMS)}
    g1 = bool(
        vte_arm["treat"] < vte_arm["static"]
        and all(vte_arm["treat"] < vte_arm[b] for b in BENCH_ARMS)
    )
    # G2 — per-fold VTE stability (treat < static in >= min_folds of n_folds).
    vte_beat = 0
    for f in folds:
        msk = fid == f
        if econ.vte(vt["treat"][msk], r_next[msk], sig_td) < econ.vte(
            vt["static"][msk], r_next[msk], sig_td
        ):
            vte_beat += 1
    g2 = bool(vte_beat >= g.vte_stability_min_folds)

    # ---------------- LEG 2: FKO performance fee (money), leverage-matched ----------------
    seed = p25.core.seed
    sidx = 0  # deterministic per-comparison bootstrap seed offset

    def _boot(u_treat: np.ndarray, u_k: np.ndarray) -> dict[str, Any]:
        nonlocal sidx
        sidx += 1
        return econ.stationary_bootstrap_pvalue(
            u_treat - u_k,
            n_resamples=sig.n_resamples,
            expected_block_length=sig.expected_block_length,
            seed=seed * 100000 + sidx,
        )

    def _hac_p(u_treat: np.ndarray, u_k: np.ndarray) -> float:
        # DM-style HAC asymptotic cross-check (higher utility better -> loss = -U), one-sided.
        return float(diebold_mariano(-u_k, -u_treat, horizon=1, alpha=sig.alpha)["p_value"])

    fees: dict[str, Any] = {}
    for stress, tag in ((base_cost, "base"), (stress_cost, "stress3x")):
        u = _utilities(mvm, r_next, c_side, stress, gamma)
        per: dict[str, Any] = {}
        for k in GATED_VS:
            fee = _fee_vs(mvm, r_next, c_side, stress, gamma, k, ann)
            boot = _boot(u["treat"], u[k])
            per[k] = {
                "fee_bps_yr": None if fee is None else round(fee, 4),
                "bootstrap_p": boot["p_value"],
                "hac_p": round(_hac_p(u["treat"], u[k]), 6),
                "mean_util_diff": boot["mean_diff"],
            }
        fees[tag] = per

    def _passes(entry: dict[str, Any], min_fee: float) -> bool:
        fee = entry["fee_bps_yr"]
        return bool(fee is not None and fee > min_fee and entry["bootstrap_p"] < sig.alpha)

    g3 = _passes(fees["base"]["static"], g.min_effect_floor_bps_per_year)
    g4 = all(_passes(fees["base"][b], 0.0) for b in BENCH_ARMS)
    # G5 — stressed cost: G3/G4 hold IN SIGN (floor relaxes to >0) with p<0.05 at 3x.
    g5 = _passes(fees["stress3x"]["static"], 0.0) and all(
        _passes(fees["stress3x"][b], 0.0) for b in BENCH_ARMS
    )

    # G6 — regime robustness: fee(treat vs static) >= 0 in BOTH calm and turbulent (base cost).
    regime_fees: dict[str, float | None] = {}
    g6_parts: list[bool] = []
    for rname, rmask in (("calm", ~regime_oos), ("turbulent", regime_oos)):
        if rmask.sum() >= 3:
            net_t = _net_returns(mvm["treat"][rmask], r_next[rmask], c_side, base_cost)
            net_s = _net_returns(mvm["static"][rmask], r_next[rmask], c_side, base_cost)
            try:
                rf = econ.fee_bps_per_year(econ.performance_fee(net_t, net_s, gamma), ann)
            except econ.FeeNonConvergence:
                rf = None
            regime_fees[rname] = None if rf is None else round(rf, 4)
            g6_parts.append(rf is not None and rf >= 0.0)
        else:
            regime_fees[rname] = None
            g6_parts.append(False)
    g6 = bool(len(g6_parts) == 2 and all(g6_parts))

    # G7 — temporal stability of value (treat beats static AND rw in >= min_folds folds, base cost).
    u_base = _utilities(mvm, r_next, c_side, base_cost, gamma)
    beat_static = beat_rw = 0
    for f in folds:
        msk = fid == f
        if float(np.mean(u_base["treat"][msk])) > float(np.mean(u_base["static"][msk])):
            beat_static += 1
        if float(np.mean(u_base["treat"][msk])) > float(np.mean(u_base["rw"][msk])):
            beat_rw += 1
    g7 = bool(beat_static >= g.value_stability_min_folds and beat_rw >= g.value_stability_min_folds)

    # G8 — exposure neutrality (post-match mean weight) + cost-erosion (net@3x / gross fee).
    wbar_t = float(np.mean(mvm["treat"]))
    wbar_s = float(np.mean(mvm["static"]))
    mean_dev = abs(wbar_t - wbar_s)
    g8_match = bool(
        wbar_s > 0 and mean_dev <= g.g8_max_post_match_mean_weight_dev_vs_static * wbar_s
    )
    gross_fee = _fee_vs(mvm, r_next, c_side, 0.0, gamma, "static", ann)
    net3x_fee = fees["stress3x"]["static"]["fee_bps_yr"]
    if gross_fee is not None and gross_fee > 0 and net3x_fee is not None:
        net_to_gross = net3x_fee / gross_fee
    else:
        net_to_gross = 0.0
    g8_cost = bool(net_to_gross >= g.g8_min_net_to_gross_fee_frac_at_3x)
    g8 = bool(g8_match and g8_cost)

    # E9 — directional-leakage VOID on the ACTIVE (overweight) exposure (matched MV weights).
    e9_corrs: dict[str, float] = {}
    for k in GATED_VS:
        e9_corrs[k] = round(econ.active_exposure_corr(mvm["treat"], mvm[k], r_next), 6)
    e9_void = bool(any(abs(c) >= g.e9_max_abs_corr for c in e9_corrs.values()))

    gates = {
        "g1_vte_intersection": g1,
        "g2_vte_temporal_stability": g2,
        "g3_econ_value_vs_static": g3,
        "g4_econ_value_vs_each_benchmark": g4,
        "g5_stressed_cost_robustness": g5,
        "g6_regime_robustness": g6,
        "g7_value_temporal_stability": g7,
        "g8_exposure_neutrality_costerosion": g8,
    }
    passed = all(gates.values()) and not e9_void
    return {
        "vte": {k: (None if not np.isfinite(v) else round(v, 6)) for k, v in vte_arm.items()},
        "vte_fold_stability": {
            "treat_beats_static_folds": vte_beat,
            "n_folds": int(folds.size),
            "min_folds": g.vte_stability_min_folds,
        },
        "fees": fees,
        "regime_fees": regime_fees,
        "value_fold_stability": {
            "treat_beats_static_folds": beat_static,
            "treat_beats_rw_folds": beat_rw,
            "n_folds": int(folds.size),
            "min_folds": g.value_stability_min_folds,
        },
        "g8_detail": {
            "wbar_treat": round(wbar_t, 6),
            "wbar_static": round(wbar_s, 6),
            "mean_weight_dev": round(mean_dev, 6),
            "match_ok": g8_match,
            "gross_fee_bps_yr": None if gross_fee is None else round(gross_fee, 4),
            "net3x_fee_bps_yr": net3x_fee,
            "net_to_gross_frac": round(net_to_gross, 4),
            "cost_erosion_ok": g8_cost,
        },
        "e9": {
            "active_exposure_corr": e9_corrs,
            "max_abs_corr_threshold": g.e9_max_abs_corr,
            "voided": e9_void,
        },
        "gates": gates,
        "passed": bool(passed),
    }


# --------------------------------------------------------------------------- #
# Reported-not-gated diagnostics
# --------------------------------------------------------------------------- #
def _reported_diagnostics(
    weights: dict[str, Any], r_next: np.ndarray, *, p25: Phase25Config, c_side: float
) -> dict[str, Any]:
    """Diagnostics surfaced but NEVER used to claim a pass (raw fee, γ/cost sensitivities…)."""
    o = p25.overlay
    ann = o.ann_factor
    mvm = weights["mv_matched"]
    mvr = weights["mv_raw"]
    base_cost = 1.0

    def _fee(mv: dict[str, np.ndarray], k: str, stress: float, gamma: float) -> float | None:
        net_t = _net_returns(mv["treat"], r_next, c_side, stress)
        net_k = _net_returns(mv[k], r_next, c_side, stress)
        try:
            return round(econ.fee_bps_per_year(econ.performance_fee(net_t, net_k, gamma), ann), 4)
        except econ.FeeNonConvergence:
            return None

    # γ sensitivity (reported only) — matched, base cost, vs static.
    gamma_fees = {
        f"gamma_{gv:g}": _fee(mvm, "static", base_cost, gv) for gv in o.gamma_reported_only
    }
    # cost sensitivity (reported only) — matched, vs static, headline γ.
    cost_fees = {
        f"cost_{m:g}x": _fee(mvm, "static", m, o.gamma) for m in p25.costs.reported_only_multipliers
    }
    # treat net Sharpe (annualized), turnover-free flat-RTH P&L.
    net_t = _net_returns(mvm["treat"], r_next, c_side, base_cost)
    sd = float(np.std(net_t, ddof=1))
    sharpe = float(np.mean(net_t) / sd * np.sqrt(ann)) if sd > 0 else None

    return {
        "raw_unmatched_fee_vs_static_bps_yr": _fee(mvr, "static", base_cost, o.gamma),
        "gamma_sensitivity_fees_vs_static_bps_yr": gamma_fees,
        "cost_sensitivity_fees_vs_static_bps_yr": cost_fees,
        "per_arm_weight_return_corr": {
            arm: round(econ.weight_return_corr(mvm[arm], r_next), 6)
            for arm in ("static", *FCAST_ARMS)
        },
        "mean_matched_mv_weight": {
            arm: round(float(np.mean(mvm[arm])), 6) for arm in ("static", *FCAST_ARMS)
        },
        "treat_net_sharpe_annualized": None if sharpe is None else round(sharpe, 4),
    }


# --------------------------------------------------------------------------- #
# Per-symbol run
# --------------------------------------------------------------------------- #
def run_symbol_ev(
    symbol: str, p25: Phase25Config, *, store: DuckStore
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Regenerate forecasts (verified), assemble the overlay, and evaluate the nine gates.

    Returns ``(result, persist_payload)``. The drift guard (compare against any stored fingerprint)
    runs here read-only on every call; the actual artifact WRITE is deferred to the caller's
    all-or-nothing save block, so a ``save=False`` / dry run mutates nothing on disk.
    """
    if symbol not in p25.costs.per_symbol:
        raise ValueError(f"[{symbol}] no frozen cost block in config/phase25_econ.yaml")
    vcfg = p25.core
    fcfg = FeatureConfig.load()
    base = build_daily_base(symbol, vcfg, store=store)
    mtm = make_matrix(base, vcfg.horizons.primary)
    returns = aligned_returns(symbol, mtm, fcfg.session, store)

    oos, fold_meta = _econ_forecasts(
        mtm,
        vcfg,
        returns,
        ewma_lambda=p25.ewma_lambda,
        gamma=p25.overlay.gamma,
        w_cap=p25.overlay.w_cap,
        mu_floor=p25.drift.mu_floor_per_day,
    )
    repro = _verify_reproduction(symbol, oos, p25)

    r_next_full = _next_rth_returns(symbol, base, mtm, fcfg.session, store)
    r_next = r_next_full[oos["idx"]]
    finite = np.isfinite(r_next)
    if not finite.all():
        n_drop = int((~finite).sum())
        logger.warning(f"[{symbol}] dropping {n_drop} OOS row(s) with no t+1 RTH return")
        for key in ("idx", "y_true", "fold_id", "rv", "t0", *_FCAST_KEY.values()):
            oos[key] = oos[key][finite]
        r_next = r_next[finite]
    cov = float(np.mean(finite))
    if cov < 0.99:
        raise ValueError(f"[{symbol}] next-RTH-return coverage {cov:.3f} < 0.99 — alignment broken")

    fingerprint = _frame_fingerprint(oos, r_next)
    _check_fingerprint_drift(symbol, fingerprint)  # read-only fail-closed; no write here

    weights = _assemble_weights(oos, fold_meta, p25)
    regime_full = regime_labels(mtm.rv, vcfg.regime.trailing_days)
    regime_oos = regime_full[oos["idx"]]

    c_side = p25.costs.per_symbol[symbol].c_side()
    gates = _evaluate_gates(oos, weights, r_next, regime_oos, p25=p25, c_side=c_side)
    diagnostics = _reported_diagnostics(weights, r_next, p25=p25, c_side=c_side)

    passed, void = gates["passed"], gates["e9"]["voided"]
    verdict = VERDICT_VOID if void else (VERDICT_CONFIRMED if passed else VERDICT_NULL)
    g = gates["gates"]
    logger.info(
        f"[{symbol}] n={len(r_next)} c_side={c_side:.3e} | "
        f"G1={g['g1_vte_intersection']} G2={g['g2_vte_temporal_stability']} "
        f"G3={g['g3_econ_value_vs_static']} G4={g['g4_econ_value_vs_each_benchmark']} "
        f"G5={g['g5_stressed_cost_robustness']} G6={g['g6_regime_robustness']} "
        f"G7={g['g7_value_temporal_stability']} G8={g['g8_exposure_neutrality_costerosion']} "
        f"E9_void={void} -> {'PASS' if (passed and not void) else verdict}"
    )

    result = {
        "symbol": symbol,
        "phase": "25",
        "econvalue_version": p25.econvalue_version,
        "n_oos": int(len(r_next)),
        "reproduction": repro,
        "fingerprint": fingerprint,
        "c_side": c_side,
        "cost_stress_gated": list(p25.costs.base_and_stress()),
        "mu_floored_folds": int(sum(m["mu_floored"] for m in fold_meta.values())),
        **{
            k: gates[k]
            for k in (
                "vte",
                "vte_fold_stability",
                "fees",
                "regime_fees",
                "value_fold_stability",
                "g8_detail",
                "e9",
                "gates",
            )
        },
        "passed": bool(passed and not void),
        "verdict": verdict,
        "reported_diagnostics": diagnostics,
        "cost_disclaimer": "economic-value verdict in offline simulation only — authorizes NO live "
        "trading and NO automated paper-execution deployment, regardless of outcome.",
    }
    payload = {"oos": oos, "r_next": r_next, "fingerprint": fingerprint, "repro": repro}
    return result, payload


# --------------------------------------------------------------------------- #
# Decision + persistence
# --------------------------------------------------------------------------- #
def decide_ev(symbol_results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Frozen rule: both PASS → confirmed; one → fragile; any VOID / neither → honest null."""
    n = len(symbol_results)
    voids = [s for s, r in symbol_results.items() if r["verdict"] == VERDICT_VOID]
    passes = [s for s, r in symbol_results.items() if r["passed"]]
    if voids:
        overall = VERDICT_VOID
        note = (
            f"directional-leakage VOID on {', '.join(voids)} (E9) — the active overweight is "
            "directionally correlated; verdict voided regardless of G1-G8. Honest null."
        )
    elif n > 0 and len(passes) == n:
        overall = VERDICT_CONFIRMED
        note = (
            "both symbols clear G1-G8 with E9 not voided → confirmed OFFLINE/SIMULATED/PAPER "
            "economic value of the 1-day volatility forecast (net of realistic and 3×-stressed "
            "costs). Authorizes ONLY designing a separate, future, pre-registered paper-trading "
            "prototype (or options/variance-swap valuation); NEVER live trading or deployment."
        )
    elif passes:
        overall = VERDICT_FRAGILE
        note = (
            f"single-symbol pass ({', '.join(passes)}); the other failed a gate — FRAGILE; "
            "nothing promoted."
        )
    else:
        overall = VERDICT_NULL
        note = (
            "the confirmed accuracy edge did not translate into economically-meaningful, "
            "cost-robust, direction-free value — an honest, acceptable, likely-modal null. "
            "Re-scoped only by deliberate operator decision."
        )
    return {
        "per_symbol": {s: r["verdict"] for s, r in symbol_results.items()},
        "passes": passes,
        "voids": voids,
        # a VOID is an honest null, not a fragile single-symbol pass — don't co-report both.
        "fragile": (not voids) and bool(passes) and len(passes) < n,
        "overall": overall,
        "note": note,
    }


def _check_fingerprint_drift(symbol: str, fingerprint: str) -> None:
    """Read-only fail-closed: if a fingerprint from a prior saved run exists, it MUST match.

    Catches any cross-run drift (fold ids, return series, GARCH fallback) the QLIKE pins alone
    cannot. Never writes — safe to call on a dry / ``save=False`` run.
    """
    fp_path = _runs_dir() / f"{symbol}_fingerprint.json"
    if not fp_path.exists():
        return
    prior = json.loads(fp_path.read_text(encoding="utf-8")).get("fingerprint")
    if prior and prior != fingerprint:
        raise ValueError(
            f"[{symbol}] OOS-frame fingerprint drift: stored {prior} != regenerated "
            f"{fingerprint}. The reused Phase-23 forecasts are NOT reproducing — fail closed."
        )


def _write_oos_frame(
    symbol: str, oos: dict[str, Any], r_next: np.ndarray, fingerprint: str, repro: dict[str, Any]
) -> None:
    """Persist the regenerated OOS frame + fingerprint (gitignored) so the audit trail is reproducible.

    Only called from the canonical save path, after every symbol has succeeded (all-or-nothing), so
    a mid-run failure can never leave a partial fingerprint that poisons a future run's drift guard.
    """
    d = _runs_dir()
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        d / f"{symbol}_oos_frame.npz",
        fcast_treat=oos["fcast_treat"],
        fcast_har=oos["fcast_har"],
        fcast_rw=oos["fcast_rw"],
        fcast_ewma=oos["fcast_ewma"],
        fcast_garch=oos["fcast_garch"],
        fold_id=oos["fold_id"],
        t0=np.asarray(oos["t0"]).astype("datetime64[ns]"),
        rv=oos["rv"],
        y_true=oos["y_true"],
        next_rth_log_return=r_next,
    )
    (d / f"{symbol}_fingerprint.json").write_text(
        json.dumps({"fingerprint": fingerprint, **repro}, indent=2, default=str), encoding="utf-8"
    )


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


def _log_mlflow(result: dict[str, Any], p25: Phase25Config) -> None:
    """Best-effort MLflow logging (never fails the run)."""
    try:
        import os

        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
        import mlflow
    except Exception as exc:  # noqa: BLE001 - tracking is optional
        logger.warning(f"mlflow unavailable ({exc}); skipping tracking")
        return
    mlflow.set_tracking_uri((Settings().data_dir / "mlruns").as_uri())
    mlflow.set_experiment(p25.artifacts.get("mlflow_experiment", "volatility-econvalue-ev1"))
    with mlflow.start_run(run_name=f"{result['symbol']}-{p25.econvalue_version}"):
        mlflow.log_param("symbol", result["symbol"])
        mlflow.log_param("n_oos", result["n_oos"])
        base = result["fees"]["base"]
        mlflow.log_metric("fee_vs_static_bps_yr", float(base["static"]["fee_bps_yr"] or 0.0))
        mlflow.log_metric("bootstrap_p_vs_static", float(base["static"]["bootstrap_p"] or 1.0))
        for gate, val in result["gates"].items():
            mlflow.set_tag(gate, str(val))
        mlflow.set_tag("e9_voided", str(result["e9"]["voided"]))
        mlflow.set_tag("verdict", result["verdict"])
        mlflow.log_dict(result, "result.json")


def run_econvalue(
    symbols: list[str] | None = None, *, log_mlflow: bool = True, save: bool = True
) -> dict[str, Any]:
    """Run the Phase-25 economic-value verdict for all symbols → combined verdict (persisted)."""
    p25 = Phase25Config.load()
    syms = list(dict.fromkeys(symbols or p25.symbols))
    is_full = set(syms) == set(p25.symbols)
    if save and not is_full:
        raise ValueError(
            f"Phase-25's combined verdict is pre-registered over {list(p25.symbols)}; refusing to "
            f"save a canonical decision over {syms}. Re-run with the exact set or pass save=False."
        )

    # Compute ALL symbols first; persist only after every symbol succeeds, so a mid-run failure can
    # never leave a fresh OOS-frame / fingerprint / symbol JSON beside a stale verdict (and a
    # save=False / dry run writes nothing at all).
    store = DuckStore()
    symbol_results: dict[str, dict[str, Any]] = {}
    payloads: dict[str, dict[str, Any]] = {}
    try:
        for symbol in syms:
            symbol_results[symbol], payloads[symbol] = run_symbol_ev(symbol, p25, store=store)
    finally:
        store.close()

    decision = decide_ev(symbol_results)
    verdict = {
        "phase": "25",
        "econvalue_version": p25.econvalue_version,
        "preregistration": "docs/PHASE25_PREREGISTRATION.md",
        "study": "economic value of the confirmed 1-day volatility forecast (FKO performance fee + "
        "return-free vol-targeting error), offline/paper, net of realistic & stressed costs",
        "framework": "fko_performance_fee_plus_returnfree_vol_targeting",
        "gamma": p25.overlay.gamma,
        "w_cap": p25.overlay.w_cap,
        "min_effect_floor_bps_yr": p25.gates.min_effect_floor_bps_per_year,
        "cost_stress_gated": list(p25.costs.base_and_stress()),
        "bootstrap": {
            "test": "stationary_bootstrap_politis_romano_1994",
            "n_resamples": p25.significance.n_resamples,
            "expected_block_length": p25.significance.expected_block_length,
            "alpha": p25.significance.alpha,
        },
        "symbols": symbol_results,
        "decision": decision,
        "partial": not is_full,
        "cost_disclaimer": "economic-value verdict in offline simulation only — like forecast "
        "skill before it, it is NOT a trading authorization. Authorizes no strategy/backtest/risk/"
        "execution/live trading; at most a separate future pre-registered paper-prototype design.",
    }
    if save:
        for symbol, pl in payloads.items():
            _write_oos_frame(symbol, pl["oos"], pl["r_next"], pl["fingerprint"], pl["repro"])
        for r in symbol_results.values():
            save_symbol_run(r)
        save_verdict(verdict)
    if log_mlflow:
        for r in symbol_results.values():
            _log_mlflow(r, p25)
    return verdict


def _print_verdict(verdict: dict[str, Any]) -> None:
    logger.info("=== Phase-25 economic-value verdict ===")
    for sym, r in verdict["symbols"].items():
        g = r["gates"]
        base = r["fees"]["base"]["static"]
        logger.info(
            f"[{sym}] n={r['n_oos']} fee_vs_static={base['fee_bps_yr']} bps/yr "
            f"(p={base['bootstrap_p']}) gates={sum(g.values())}/8 E9_void={r['e9']['voided']} "
            f"-> {r['verdict']}"
        )
    d = verdict["decision"]
    logger.info(f"DECISION: {d['overall']} (passes={d['passes'] or 'none'}) — {d['note']}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="volatility.run_econ", description=__doc__)
    p.add_argument("--symbols", nargs="+", default=None, help="default: config symbols")
    p.add_argument("--no-mlflow", action="store_true", help="skip MLflow logging")
    p.add_argument("--no-save", action="store_true", help="compute but do not persist the verdict")
    args = p.parse_args(argv)
    verdict = run_econvalue(
        symbols=args.symbols, log_mlflow=not args.no_mlflow, save=not args.no_save
    )
    _print_verdict(verdict)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
