"""Phase-25 unit tests — the frozen economic-overlay contract + the pure economic primitives.

Covers the NEW Phase-25 logic in isolation: the frozen config loads to the right knobs; the
long-only weight rules; the FKO performance-fee solver (identical arms → 0; a constant edge → that
edge exactly; a no-real-root case → FAIL); the return-free VTE; the Politis-Romano stationary
bootstrap (determinism + one-sided significance); leverage matching; the E9 active-exposure
correlation; the ``t+1`` forward-shift alignment; the decision rule; and the canonical-set save
guard. The end-to-end economic verdict is exercised by the real ``run_econ`` run on the lakes.
"""

from __future__ import annotations

import json
import math
from datetime import date

import numpy as np
import polars as pl
import pytest

from options_system.volatility import econ
from options_system.volatility.config_econ import Phase25Config
from options_system.volatility.run_econ import VERDICT_CONFIRMED, VERDICT_VOID, decide_ev


# --------------------------------------------------------------------------- #
# Frozen contract
# --------------------------------------------------------------------------- #
def test_config_loads_frozen_knobs():
    p25 = Phase25Config.load()
    assert p25.econvalue_version == "ev1"
    assert p25.symbols == ["MES", "MNQ"]
    # inherited Phase-23 core (h=1, seed 7, the confirmed model)
    assert p25.core.horizons.primary == 1
    assert p25.core.seed == 7
    assert p25.core.walk_forward.oos_start == "2022-01-01"
    # overlay knobs
    assert p25.overlay.gamma == 5.0
    assert p25.overlay.w_cap == 1.5
    assert p25.overlay.w_floor == 0.1
    assert p25.overlay.sigma_target_annual == 0.10
    assert p25.overlay.ann_factor == 252
    assert math.isclose(p25.overlay.sigma_target_daily, 0.10 / math.sqrt(252), rel_tol=1e-9)
    # drift firewall
    assert p25.drift.mu_floor_per_day == 0.00012
    assert p25.drift.vol_target_leg_drift == 0.0
    assert p25.drift.rf == 0.0
    # bootstrap
    assert p25.significance.n_resamples == 10000
    assert p25.significance.expected_block_length == 10
    assert p25.significance.alpha == 0.05
    # gates
    assert p25.gates.min_effect_floor_bps_per_year == 25
    assert p25.gates.e9_max_abs_corr == 0.10
    assert p25.gates.g8_min_net_to_gross_fee_frac_at_3x == 0.50


def test_config_cost_fractions_match_frozen_derivation():
    p25 = Phase25Config.load()
    mes = p25.costs.per_symbol["MES"].c_side()
    mnq = p25.costs.per_symbol["MNQ"].c_side()
    # (0.62 + 0.625) / 25000 ; (0.62 + 0.25) / 35000
    assert math.isclose(mes, 4.98e-5, rel_tol=5e-3)
    assert math.isclose(mnq, 2.49e-5, rel_tol=5e-3)


def test_config_gated_cost_levels_are_frozen():
    p25 = Phase25Config.load()
    assert p25.costs.base_and_stress() == (1.0, 3.0)


def test_expected_qlike_pins_present():
    p25 = Phase25Config.load()
    exp = p25.reproduction_guard.expected_qlike
    assert exp["treat"]["MES"] == 0.246401
    assert exp["treat"]["MNQ"] == 0.224083
    assert set(exp) == {"treat", "har", "rw", "ewma", "garch"}


# --------------------------------------------------------------------------- #
# Weight rules
# --------------------------------------------------------------------------- #
def test_mean_variance_weight_long_only_and_capped():
    sigma2 = np.array([1e-4, 1e-6, 1e-2])
    w = econ.mean_variance_weight(mu_bar=0.0005, gamma=5.0, sigma2=sigma2, w_cap=1.5)
    assert np.all(w >= 0.0) and np.all(w <= 1.5)
    # tiny variance -> hits the cap; large variance -> small weight
    assert w[1] == 1.5
    assert w[2] < w[0] < w[1]


def test_mean_variance_weight_never_shorts():
    # a negative mu would imply a short; the long-only clamp forces >= 0 (mu_bar is >= 0 upstream)
    w = econ.mean_variance_weight(mu_bar=-0.01, gamma=5.0, sigma2=np.array([1e-4]), w_cap=1.5)
    assert w[0] == 0.0


def test_vol_target_weight_floored_and_capped():
    sigma = np.array([0.001, 0.05, 0.5])
    wv = econ.vol_target_weight(sigma_target_daily=0.0063, sigma=sigma, w_floor=0.1, w_cap=1.5)
    assert np.all(wv >= 0.1) and np.all(wv <= 1.5)
    assert wv[0] == 1.5  # tiny forecast vol -> cap
    assert wv[2] == 0.1  # huge forecast vol -> floor


# --------------------------------------------------------------------------- #
# FKO performance fee
# --------------------------------------------------------------------------- #
def test_fee_zero_for_identical_arms():
    rng = np.random.default_rng(0)
    net = rng.normal(0.0, 0.01, size=500)
    phi = econ.performance_fee(net, net, gamma=5.0)
    assert abs(phi) < 1e-12


def test_fee_equals_constant_additive_edge():
    # net_treat = net_k + delta (constant) => the admissible fee is exactly delta.
    rng = np.random.default_rng(1)
    net_k = rng.normal(0.0, 0.01, size=800)
    delta = 0.0007
    phi = econ.performance_fee(net_k + delta, net_k, gamma=5.0)
    assert phi == pytest.approx(delta, abs=1e-9)
    # a positive edge => positive fee => investor pays to switch to treatment
    assert phi > 0


def test_fee_non_convergence_raises():
    # net_k pinned at the utility-maximizing R (=1/gamma per day) and a high-variance treat:
    # arm k achieves higher total utility than ANY constant-shifted treat -> no real root.
    net_k = np.full(400, 1.0 / 5.0)  # R_k = 1.2 = 1/a, the utility peak
    net_treat = np.tile([0.5, -0.1], 200)  # high spread, mean far below peak
    with pytest.raises(econ.FeeNonConvergence):
        econ.performance_fee(net_treat, net_k, gamma=5.0)


def test_fee_bps_annualization():
    assert econ.fee_bps_per_year(0.0001, ann_factor=252) == pytest.approx(0.0001 * 252 * 1e4)


def test_utility_coefficient():
    assert econ.utility_coefficient(5.0) == pytest.approx(5.0 / 6.0)


# --------------------------------------------------------------------------- #
# Return-free VTE
# --------------------------------------------------------------------------- #
def test_vte_zero_when_realized_vol_hits_target():
    rng = np.random.default_rng(2)
    r = rng.normal(0.0, 0.02, size=2000)
    target = 0.0063
    wv = np.full(r.size, target / float(np.std(r, ddof=1)))  # scale so realized vol == target
    assert econ.vte(wv, r, target) == pytest.approx(0.0, abs=1e-12)


def test_vte_positive_when_off_target():
    rng = np.random.default_rng(3)
    r = rng.normal(0.0, 0.02, size=2000)
    wv = np.full(r.size, 1.0)
    assert econ.vte(wv, r, 0.0063) > 0.0


# --------------------------------------------------------------------------- #
# Politis-Romano stationary bootstrap
# --------------------------------------------------------------------------- #
def test_bootstrap_deterministic():
    rng = np.random.default_rng(4)
    diff = rng.normal(0.001, 0.01, size=300)
    a = econ.stationary_bootstrap_pvalue(diff, n_resamples=500, expected_block_length=10, seed=7)
    b = econ.stationary_bootstrap_pvalue(diff, n_resamples=500, expected_block_length=10, seed=7)
    assert a["p_value"] == b["p_value"]


def test_bootstrap_strong_positive_is_significant():
    diff = np.full(300, 0.002) + np.random.default_rng(5).normal(0, 1e-5, 300)
    res = econ.stationary_bootstrap_pvalue(diff, n_resamples=1000, expected_block_length=10, seed=7)
    assert res["p_value"] < 0.05


def test_bootstrap_zero_centered_not_significant():
    diff = np.random.default_rng(6).normal(0.0, 0.01, size=400)
    res = econ.stationary_bootstrap_pvalue(diff, n_resamples=1000, expected_block_length=10, seed=7)
    assert res["p_value"] > 0.05


# --------------------------------------------------------------------------- #
# Leverage matching + E9
# --------------------------------------------------------------------------- #
def test_leverage_match_constant():
    assert econ.leverage_match_constant(0.5, 1.0) == pytest.approx(2.0)
    assert econ.leverage_match_constant(0.0, 1.0) == 1.0  # degenerate -> no rescale


def test_active_exposure_corr_detects_directional_overweight():
    rng = np.random.default_rng(7)
    r = rng.normal(0.0, 0.02, size=500)
    w_k = np.full(500, 0.5)
    w_treat = 0.5 + 3.0 * r  # active overweight perfectly tracks the return -> directional leak
    c = econ.active_exposure_corr(w_treat, w_k, r)
    assert abs(c) >= 0.10


def test_active_exposure_corr_zero_for_constant_active():
    r = np.random.default_rng(8).normal(0.0, 0.02, size=500)
    w_treat = np.full(500, 0.7)
    w_k = np.full(500, 0.4)  # constant active overweight -> zero correlation
    assert econ.active_exposure_corr(w_treat, w_k, r) == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------- #
# t+1 alignment
# --------------------------------------------------------------------------- #
def test_forward_shift_maps_t_to_t_plus_1():
    same = np.array([10.0, 20.0, 30.0, 40.0])
    nxt = econ.forward_shift_one(same)
    assert nxt[0] == 20.0  # session 0's held return is session 1's RTH return
    assert nxt[1] == 30.0
    assert nxt[2] == 40.0
    assert math.isnan(nxt[3])  # last session has no t+1


# --------------------------------------------------------------------------- #
# Decision rule
# --------------------------------------------------------------------------- #
def _sym(passed: bool, verdict: str) -> dict:
    return {"passed": passed, "verdict": verdict}


def test_decide_both_pass_is_confirmed():
    res = {"MES": _sym(True, VERDICT_CONFIRMED), "MNQ": _sym(True, VERDICT_CONFIRMED)}
    d = decide_ev(res)
    assert d["overall"] == VERDICT_CONFIRMED
    assert set(d["passes"]) == {"MES", "MNQ"}


def test_decide_one_pass_is_fragile():
    res = {"MES": _sym(True, VERDICT_CONFIRMED), "MNQ": _sym(False, "no_costed")}
    d = decide_ev(res)
    assert d["overall"] == "fragile_single_symbol"
    assert d["fragile"] is True


def test_decide_void_overrides_everything():
    res = {"MES": _sym(False, VERDICT_VOID), "MNQ": _sym(True, VERDICT_CONFIRMED)}
    d = decide_ev(res)
    assert d["overall"] == VERDICT_VOID
    assert d["voids"] == ["MES"]


def test_decide_neither_pass_is_null():
    res = {"MES": _sym(False, "no_costed"), "MNQ": _sym(False, "no_costed")}
    d = decide_ev(res)
    assert d["overall"].startswith("no_costed")


# --------------------------------------------------------------------------- #
# Reproduction guard — the structural (frame-shape) fail-closed pins
# --------------------------------------------------------------------------- #
def test_frozen_frame_shape_pins():
    from options_system.volatility.run_econ import _FROZEN_FRAME, _assert_frozen_frame_shape

    gd = _FROZEN_FRAME["MES"]["garch_diag"]
    _assert_frozen_frame_shape("MES", 1139, 18, gd)  # exact frozen shape -> ok
    with pytest.raises(ValueError, match="n_oos"):
        _assert_frozen_frame_shape("MES", 1138, 18, gd)
    with pytest.raises(ValueError, match="folds"):
        _assert_frozen_frame_shape("MES", 1139, 17, gd)
    with pytest.raises(ValueError, match="GARCH"):  # a GARCH-fallback drift the QLIKE pins miss
        _assert_frozen_frame_shape("MES", 1139, 18, {**gd, "converged": 16})
    with pytest.raises(ValueError, match="frozen Phase-23 frame anchor"):
        _assert_frozen_frame_shape("FOO", 1139, 18, gd)


# --------------------------------------------------------------------------- #
# Fingerprint cross-run drift guard (read-only)
# --------------------------------------------------------------------------- #
def test_fingerprint_drift_guard(tmp_path, monkeypatch):
    import options_system.volatility.run_econ as run_econ

    monkeypatch.setattr(run_econ, "_runs_dir", lambda: tmp_path)
    run_econ._check_fingerprint_drift("MES", "abc")  # no prior file -> no raise
    (tmp_path / "MES_fingerprint.json").write_text(json.dumps({"fingerprint": "abc"}))
    run_econ._check_fingerprint_drift("MES", "abc")  # matching prior -> no raise
    with pytest.raises(ValueError, match="fingerprint drift"):
        run_econ._check_fingerprint_drift("MES", "different")


# --------------------------------------------------------------------------- #
# Per-symbol t+1 alignment integration (the real session→t+1 mapping path)
# --------------------------------------------------------------------------- #
def test_map_next_returns_scores_t_against_t_plus_1():
    from options_system.volatility.run_econ import _map_next_returns

    full_sd = pl.Series([date(2022, 1, d) for d in (3, 4, 5, 6, 7)])
    same = pl.Series([1.0, 2.0, 3.0, 4.0, 5.0])  # same-session r_t per full session
    # the matrix drops the trailing session (its forward RV target is incomplete)
    matrix_sd = pl.Series([date(2022, 1, d) for d in (3, 4, 5, 6)])
    nxt = _map_next_returns(full_sd, same, matrix_sd)
    # forecast row t earns session t+1's return — never the same-session r_t (no look-ahead)
    assert list(nxt) == [2.0, 3.0, 4.0, 5.0]
    # the last forecast row (1/6) maps to the DROPPED trailing session's (1/7) return, not NaN
    assert nxt[-1] == 5.0


# --------------------------------------------------------------------------- #
# Canonical-set save guard (Phase-20/23 discipline)
# --------------------------------------------------------------------------- #
def test_canonical_save_refuses_subset():
    from options_system.volatility.run_econ import run_econvalue

    with pytest.raises(ValueError, match="pre-registered over"):
        run_econvalue(symbols=["MES"], save=True, log_mlflow=False)
