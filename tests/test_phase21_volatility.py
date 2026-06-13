"""Phase-21 volatility-forecast: RV estimator, HAR predictors + forward target causality, QLIKE,
the HAC/HLN Diebold-Mariano test, the regime split, config-driven gates, determinism, and the
disclosed L2 fallback.

Pure/offline tests on the new primitives + a small synthetic LightGBM fit (no lake, no network, no
spend; the heavy walk-forward is exercised by the real run).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from options_system.features.config import FeatureConfig
from options_system.validation.forecast_stats import (
    diebold_mariano,
    qlike_from_log,
    qlike_from_variance,
)
from options_system.volatility.config import VolatilityConfig
from options_system.volatility.har import fit_har, predict_har
from options_system.volatility.lgbm import (
    VolatilityLGBM,
    build_vol_estimator,
    qlike_eval,
    qlike_objective,
)
from options_system.volatility.realized import (
    daily_realized_variance,
    forward_log_rv,
    har_predictors,
)
from options_system.volatility.run import regime_labels

SCFG = FeatureConfig.load().session
VCFG = VolatilityConfig.load()


# --------------------------------------------------------------------------- #
# RV estimator
# --------------------------------------------------------------------------- #
def _session_bars(day: datetime, n_min: int, seed: int) -> pl.DataFrame:
    """A synthetic RTH session: ``n_min`` 1-minute bars starting 14:30 UTC (= 09:30 ET winter)."""
    rng = np.random.default_rng(seed)
    steps = rng.normal(0, 0.0005, size=n_min)
    close = 5000.0 * np.exp(np.cumsum(steps))
    ts = [day.replace(hour=14, minute=30) + timedelta(minutes=i) for i in range(n_min)]
    return pl.DataFrame({"ts_event": ts, "close": close}).with_columns(
        pl.col("ts_event").dt.cast_time_unit("us").dt.replace_time_zone("UTC")
    )


def _bars(specs: list[tuple[datetime, int, int]]) -> pl.DataFrame:
    return pl.concat([_session_bars(d, n, s) for d, n, s in specs])


def test_rv_estimator_sessions_positive_and_drops_incomplete():
    # Two full sessions (390 RTH minutes) + one stub (10 minutes < min_5min_returns).
    bars = _bars(
        [
            (datetime(2022, 1, 4, tzinfo=UTC), 390, 1),
            (datetime(2022, 1, 5, tzinfo=UTC), 390, 2),
            (datetime(2022, 1, 6, tzinfo=UTC), 10, 3),
        ]
    )
    frame, dropped = daily_realized_variance(bars, SCFG, VCFG.rv)
    assert frame.height == 2  # the 10-minute session is dropped as incomplete
    assert dropped == 1
    assert (frame["rv"].to_numpy() > 0).all()
    assert frame["t_close"].dtype == pl.Datetime("us", "UTC")


def test_rv_estimator_is_deterministic():
    bars = _bars([(datetime(2022, 1, 4, tzinfo=UTC), 390, 7)])
    a, _ = daily_realized_variance(bars, SCFG, VCFG.rv)
    b, _ = daily_realized_variance(bars, SCFG, VCFG.rv)
    assert np.allclose(a["rv"].to_numpy(), b["rv"].to_numpy())


def test_rv_is_five_grid_average():
    """The reported RV equals the mean of the five offset-grid RVs (the estimator's definition)."""
    bars = _session_bars(datetime(2022, 1, 4, tzinfo=UTC), 390, 11)
    frame, _ = daily_realized_variance(bars, SCFG, VCFG.rv)
    logc = np.log(bars["close"].to_numpy())
    grids = []
    for off in range(VCFG.rv.n_subsample_grids):
        idx = np.arange(off, logc.size, VCFG.rv.sampling_minutes)
        r = np.diff(logc[idx])
        grids.append(float(np.dot(r, r)))
    assert frame["rv"].to_numpy()[0] == pytest.approx(float(np.mean(grids)), rel=1e-9)


# --------------------------------------------------------------------------- #
# HAR predictors + forward target: causality
# --------------------------------------------------------------------------- #
def test_har_predictors_are_causal_and_warmed_up():
    rv = np.linspace(1e-4, 5e-4, 40)
    har = har_predictors(rv, (1, 5, 22))
    assert set(har) == {"har_log_rv_d1", "har_log_rv_w5", "har_log_rv_m22"}
    # daily = log(rv_t) exactly.
    assert np.allclose(har["har_log_rv_d1"], np.log(rv))
    # monthly NaN until 22-day warmup, defined after.
    assert np.isnan(har["har_log_rv_m22"][:21]).all()
    assert np.isfinite(har["har_log_rv_m22"][21:]).all()
    # Causality: changing a FUTURE rv does not change an earlier predictor.
    rv2 = rv.copy()
    rv2[30:] = 9e-4
    har2 = har_predictors(rv2, (1, 5, 22))
    assert np.allclose(har["har_log_rv_w5"][:25], har2["har_log_rv_w5"][:25], equal_nan=True)


def test_forward_target_is_strictly_future():
    rv = np.arange(1, 11, dtype=float)  # 10 days
    y = forward_log_rv(rv, 5)
    # y_t = log(mean rv[t+1..t+5]); defined for t<=4, NaN for the last 5.
    assert np.isnan(y[5:]).all()
    assert y[0] == pytest.approx(np.log(np.mean(rv[1:6])))
    # Changing rv at/before t does not change y_t (target reads only the future).
    rv2 = rv.copy()
    rv2[0] = 100.0
    y2 = forward_log_rv(rv2, 5)
    assert y2[0] == pytest.approx(y[0])  # y_0 uses rv[1..5], not rv[0]


# --------------------------------------------------------------------------- #
# QLIKE
# --------------------------------------------------------------------------- #
def test_qlike_zero_at_perfect_forecast_and_positive_otherwise():
    rv = np.array([1e-4, 2e-4, 3e-4])
    assert np.allclose(qlike_from_variance(rv, rv), 0.0)
    assert (qlike_from_variance(rv, rv * 2.0) > 0).all()
    assert (qlike_from_variance(rv, rv * 0.5) > 0).all()


def test_qlike_log_matches_variance_form():
    y_log = np.log(np.array([1e-4, 2e-4]))
    f_log = np.log(np.array([1.5e-4, 2e-4]))
    assert np.allclose(
        qlike_from_log(y_log, f_log), qlike_from_variance(np.exp(y_log), np.exp(f_log))
    )


def test_qlike_objective_grad_hess_finite_difference():
    y = np.array([0.3, -0.2, 1.0])
    pred = np.array([0.1, 0.1, 0.5])
    grad, hess = qlike_objective(y, pred)
    # At d=0 (y==pred): grad=0, hess=1.
    g0, h0 = qlike_objective(np.array([0.5]), np.array([0.5]))
    assert g0[0] == pytest.approx(0.0) and h0[0] == pytest.approx(1.0)
    # Finite-difference check of grad = dL/dpred for L = exp(d) - d - 1, d=y-pred.
    eps = 1e-6

    def loss(p):
        d = y - p
        return np.exp(d) - d - 1.0

    fd = (loss(pred + eps) - loss(pred - eps)) / (2 * eps)
    assert np.allclose(grad, fd, atol=1e-4)
    assert (hess > 0).all()
    name, val, higher = qlike_eval(y, pred)
    assert name == "qlike" and higher is False and val >= 0


# --------------------------------------------------------------------------- #
# HAR OLS
# --------------------------------------------------------------------------- #
def test_har_ols_recovers_linear_coefficients():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(500, 3))
    true = np.array([0.5, 1.0, -0.3, 0.2])
    y = true[0] + x @ true[1:]
    coef = fit_har(x, y)
    assert np.allclose(coef, true, atol=1e-8)
    assert np.allclose(predict_har(coef, x), y, atol=1e-8)


# --------------------------------------------------------------------------- #
# Diebold-Mariano: HAC widens vs naive; HLN shrinks the stat
# --------------------------------------------------------------------------- #
def test_dm_hac_is_more_conservative_than_naive_under_autocorrelation():
    rng = np.random.default_rng(3)
    e = rng.normal(0, 1.0, size=400)
    # Positively-autocorrelated loss differential with a small positive mean (B slightly better).
    d = 0.05 + e + np.roll(e, 1)
    loss_a = d  # treat loss_b = 0 ⇒ differential = d
    loss_b = np.zeros_like(d)
    naive = diebold_mariano(loss_a, loss_b, horizon=1, alpha=0.05)  # lag 0
    hac = diebold_mariano(loss_a, loss_b, horizon=5, alpha=0.05)  # lag 4
    assert naive["lag"] == 0 and hac["lag"] == 4
    # HAC captures the autocorrelation ⇒ larger LRV ⇒ smaller stat ⇒ larger p-value.
    assert hac["lrv"] > naive["lrv"]
    assert hac["p_value"] > naive["p_value"]


def test_dm_hln_factor_uses_horizon_not_lag():
    """The HLN small-sample factor must use h (shrinks, <1), not h-1. Verified against the
    published formula sqrt((n+1-2h+h(h-1)/n)/n) at h=1 and h=5."""
    rng = np.random.default_rng(9)
    d = 0.1 + rng.normal(0, 1.0, size=300)
    loss_a, loss_b = d, np.zeros_like(d)
    for h in (1, 5):
        out = diebold_mariano(loss_a, loss_b, horizon=h, alpha=0.05)
        n = out["n"]
        expected = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
        assert expected < 1.0  # the correction always SHRINKS (never inflates)
        ratio = out["dm_stat_hln"] / out["dm_stat"]
        assert ratio == pytest.approx(expected, rel=1e-4)
    # h=1 must shrink — the old (lag-based) code inflated it via sqrt((n+1)/n) > 1.
    o1 = diebold_mariano(loss_a, loss_b, horizon=1, alpha=0.05)
    assert abs(o1["dm_stat_hln"]) < abs(o1["dm_stat"])


def test_dm_degenerate_is_not_significant():
    x = np.ones(50)
    out = diebold_mariano(x, x, horizon=5, alpha=0.05)  # zero differential
    assert out["significant"] is False and out["p_value"] == 1.0


# --------------------------------------------------------------------------- #
# Regime split (causal)
# --------------------------------------------------------------------------- #
def test_regime_split_is_causal_and_deterministic():
    rng = np.random.default_rng(5)
    rv = np.abs(rng.normal(3e-4, 1e-4, size=300))
    r1 = regime_labels(rv, 22)
    r2 = regime_labels(rv, 22)
    assert np.array_equal(r1, r2)
    # Changing a FUTURE rv must not change an earlier regime label (expanding median is causal).
    rv2 = rv.copy()
    rv2[200:] = 9e-4
    r3 = regime_labels(rv2, 22)
    assert np.array_equal(r1[:150], r3[:150])


# --------------------------------------------------------------------------- #
# Config-driven gates + the fixed model
# --------------------------------------------------------------------------- #
def test_config_holds_pre_registered_knobs():
    assert VCFG.horizons.primary == 5
    assert VCFG.horizons.all[0] == 5  # primary first
    assert VCFG.dm.alpha == 0.05
    assert VCFG.har.lags == [1, 5, 22]
    assert VCFG.lgbm.objective in {"qlike", "l2_log_rv"}
    assert set(VCFG.symbols) == {"MES", "MNQ"}


def _xy(seed: int = 0, n: int = 600):
    # n is comfortably above min_child_samples so the heavily-regularized model can split
    # (a ~200-row synthetic fold trips LightGBM's feature pre-filter; real folds are ~1,500).
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 4))
    y = np.log(np.abs(rng.normal(3e-4, 1e-4, size=n)))  # log-RV-like target (~ -8)
    return x, y


def test_vol_lgbm_is_deterministic_seed7():
    x, y = _xy()
    a = build_vol_estimator(VCFG).fit(x, y).predict(x)
    b = build_vol_estimator(VCFG).fit(x, y).predict(x)
    assert np.allclose(a, b)


def test_vol_lgbm_centers_target_and_predicts_near_level():
    """Target centering puts predictions near the true log-RV level (not stranded at 0)."""
    x, y = _xy(seed=2)
    pred = build_vol_estimator(VCFG).fit(x, y).predict(x)
    # Predictions live near the target's level (~ -8), not near raw 0.
    assert abs(float(np.mean(pred)) - float(np.mean(y))) < 1.0


def test_disclosed_l2_fallback_objective_runs():
    cfg = VCFG.model_copy(update={"lgbm": VCFG.lgbm.model_copy(update={"objective": "l2_log_rv"})})
    x, y = _xy(seed=4)
    est = build_vol_estimator(cfg)
    assert est.objective_mode == "l2_log_rv"
    pred = est.fit(x, y).predict(x)
    assert pred.shape == y.shape and np.isfinite(pred).all()


def test_vol_lgbm_qlike_objective_is_default():
    est = build_vol_estimator(VCFG)
    assert isinstance(est, VolatilityLGBM)
    assert est.objective_mode == "qlike"
