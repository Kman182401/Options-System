"""Phase-23 unit tests — the hardened benchmark battery + frozen-contract loader.

Covers the NEW pure logic (random-walk / EWMA / GARCH(1,1) forecasts, their leak-safety and
determinism) and that ``config/phase23_vol_h1.yaml`` parses to the pre-registered knobs. The
end-to-end verdict is exercised by the real ``run_h1`` run; these guard the building blocks.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from options_system.volatility.benchmarks import (
    RISKMETRICS_LAMBDA,
    ewma_forecast_log,
    fit_garch11,
    garch11_forecast_log,
    riskmetrics_params,
    rw_forecast_log,
)
from options_system.volatility.config_h1 import Phase23Config


# --------------------------------------------------------------------------- #
# Random walk
# --------------------------------------------------------------------------- #
def test_rw_is_log_of_today_rv():
    rv = np.array([1e-4, 4e-4, 9e-4])
    np.testing.assert_allclose(rw_forecast_log(rv), np.log(rv))


def test_rw_rejects_nonpositive():
    with pytest.raises(ValueError):
        rw_forecast_log(np.array([1e-4, 0.0, 9e-4]))


# --------------------------------------------------------------------------- #
# EWMA / RiskMetrics
# --------------------------------------------------------------------------- #
def test_ewma_matches_manual_recursion():
    rv = np.array([2.0, 6.0, 1.0, 5.0])
    lam = 0.9
    e = np.empty_like(rv)
    e[0] = rv[0]
    for i in range(1, rv.size):
        e[i] = lam * e[i - 1] + (1 - lam) * rv[i]
    np.testing.assert_allclose(ewma_forecast_log(rv, lam), np.log(e))


def test_ewma_is_causal():
    """Forecast at row i must not depend on rv after i (leak-safety)."""
    rng = np.random.default_rng(0)
    rv = np.abs(rng.normal(1.0, 0.3, size=50)) + 0.1
    base = ewma_forecast_log(rv, 0.94)
    perturbed = rv.copy()
    perturbed[-1] = perturbed[-1] * 10.0  # change only the last day
    after = ewma_forecast_log(perturbed, 0.94)
    np.testing.assert_allclose(base[:-1], after[:-1])  # all earlier forecasts unchanged


def test_ewma_lambda_must_be_in_unit_interval():
    with pytest.raises(ValueError):
        ewma_forecast_log(np.array([1.0, 2.0]), 1.0)


# --------------------------------------------------------------------------- #
# GARCH(1,1) QMLE
# --------------------------------------------------------------------------- #
def _simulate_garch(n, omega, alpha, beta, seed=0):
    rng = np.random.default_rng(seed)
    r = np.empty(n)
    s2 = omega / (1 - alpha - beta)
    for t in range(n):
        r[t] = np.sqrt(s2) * rng.normal()
        s2 = omega + alpha * r[t] ** 2 + beta * s2
    return r


def test_garch_fit_recovers_stationary_params():
    r = _simulate_garch(4000, omega=2e-6, alpha=0.08, beta=0.90, seed=7)
    p = fit_garch11(r)
    assert p["converged"] is True
    assert p["omega"] > 0.0
    assert 0.0 <= p["alpha"] < 0.5
    assert 0.0 <= p["beta"] < 1.0
    assert p["alpha"] + p["beta"] < 1.0  # covariance-stationary


def test_garch_fit_is_deterministic():
    r = _simulate_garch(1500, 2e-6, 0.07, 0.9, seed=3)
    a = fit_garch11(r)
    b = fit_garch11(r)
    assert a == b


def test_garch_fit_too_short_returns_unconverged():
    p = fit_garch11(np.array([0.001, -0.002, 0.0005]))
    assert p["converged"] is False


def test_garch_forecast_is_finite_and_log_of_positive_variance():
    r = _simulate_garch(2000, 2e-6, 0.08, 0.9, seed=11)
    p = fit_garch11(r)
    fc = garch11_forecast_log(r, p)
    assert fc.shape == r.shape
    assert np.all(np.isfinite(fc))
    assert np.all(np.exp(fc) > 0.0)


def test_riskmetrics_params_are_integrated():
    p = riskmetrics_params(np.array([0.01, -0.02, 0.005, -0.001]))
    assert p["omega"] == 0.0
    assert p["alpha"] == pytest.approx(1.0 - RISKMETRICS_LAMBDA)
    assert p["beta"] == pytest.approx(RISKMETRICS_LAMBDA)


# --------------------------------------------------------------------------- #
# Frozen contract loads to the pre-registered knobs
# --------------------------------------------------------------------------- #
def test_phase23_config_matches_preregistration():
    p23 = Phase23Config.load()
    # core (identical to Phase 21, except the gated horizon)
    assert p23.core.volatility_version == "vf2"
    assert p23.core.symbols == ["MES", "MNQ"]
    assert p23.core.horizons.primary == 1
    assert set(p23.core.horizons.diagnostic) == {5, 22}
    assert p23.core.dm.alpha == 0.05
    # confirm-first: the Phase-22 free-data blocks stay OFF
    assert p23.core.features.with_marketdata is False
    assert p23.core.features.with_gkg is False
    # the hardened battery
    assert p23.benchmarks.har is True
    assert p23.benchmarks.random_walk is True
    assert p23.benchmarks.ewma.lam == 0.94
    assert p23.benchmarks.garch.enabled is True
    # gate thresholds
    assert p23.gates.g3_benchmark_hardness.challengers == ["random_walk", "ewma", "garch"]
    assert p23.gates.g4_temporal_stability.min_folds_beating_har == 13
    assert p23.gates.g4_temporal_stability.min_folds_beating_rw == 13


# --------------------------------------------------------------------------- #
# GARCH return interval — must be RTH-internal (exclude the overnight gap)
# --------------------------------------------------------------------------- #
def test_daily_rth_return_excludes_overnight_gap():
    """The GARCH return is within-session (open-to-close), not close-to-close across the gap.

    Two RTH sessions with a 100→200 overnight jump: a close-to-close return would be ~log(2)≈0.69;
    the correct within-session return for day 2 is ``log(last/first)`` inside that session only.
    """
    from datetime import date, datetime, timedelta
    from zoneinfo import ZoneInfo

    from options_system.features.config import FeatureConfig
    from options_system.volatility.realized import daily_rth_log_return

    scfg = FeatureConfig.load().session
    tz = ZoneInfo(scfg.tz)
    h, m = divmod(scfg.rth_open_min, 60)

    def _session(d: date, base: float) -> list[dict]:
        start = datetime(d.year, d.month, d.day, h, m, tzinfo=tz)
        return [
            {
                "ts_event": (start + timedelta(minutes=k)).astimezone(ZoneInfo("UTC")),
                "close": base + k,
            }
            for k in range(5)  # closes base..base+4, all inside RTH
        ]

    d1, d2 = date(2023, 1, 3), date(2023, 1, 4)  # Tue, Wed (weekdays)
    bars = pl.DataFrame(_session(d1, 100.0) + _session(d2, 200.0)).with_columns(
        pl.col("ts_event").cast(pl.Datetime("us", "UTC"))
    )
    out = daily_rth_log_return(bars, scfg)
    ret_d2 = float(out.filter(pl.col("session_date") == d2)["ret"][0])
    assert ret_d2 == pytest.approx(math.log(204.0 / 200.0))  # within-session only
    assert ret_d2 < 0.1  # excludes the ~0.69 overnight jump
