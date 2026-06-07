"""Overfitting statistics checked against closed-form reference values.

Reference values are derived in comments from the source formulas (Bailey & López
de Prado 2012/2014; Bailey, Borwein, López de Prado & Zhu 2015) so a reviewer can
verify them without running anything.
"""

from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np
import pytest

from options_system.validation import stats as S

_PHI = NormalDist()


def test_psr_full_form_normal_case():
    # SR=0.1, SR*=0, n=100, skew=0, raw_kurt=3:
    #   denom = sqrt(1 + 0.5*0.1**2) = sqrt(1.005); z = 0.1*sqrt(99)/sqrt(1.005)
    #   PSR = Phi(z) ≈ 0.83955
    psr = S.probabilistic_sharpe_ratio(0.1, 0.0, 100, 0.0, 3.0)
    assert abs(psr - 0.83955) < 1e-3


def test_psr_is_exactly_half_at_the_benchmark():
    # SR_hat == SR* ⇒ z == 0 ⇒ Phi(0) == 0.5 exactly, under any skew/kurt.
    assert abs(S.probabilistic_sharpe_ratio(0.3, 0.3, 500, -0.5, 6.0) - 0.5) < 1e-12


def test_psr_normal_denominator_uses_raw_kurtosis():
    # The Normal case must reproduce denom = sqrt(1 + 0.5*SR^2) (Lo 2002): a back-door
    # check that kurtosis is treated as RAW (=3), not excess (=0).
    sr, n = 0.4, 250
    z = S.probabilistic_sharpe_ratio(sr, 0.0, n, 0.0, 3.0)
    expected = _PHI.cdf(sr * math.sqrt(n - 1) / math.sqrt(1 + 0.5 * sr**2))
    assert abs(z - expected) < 1e-12


def test_min_track_record_length_reference():
    # SR=0.1, SR*=0, p=0.95, skew=0, kurt=3:
    #   minTRL = 1 + 1.005 * (Phi^-1(0.95)/0.1)^2 = 1 + 1.005*(16.448536)^2 ≈ 272.907
    mtrl = S.min_track_record_length(0.1, 0.0, 0.0, 3.0, 0.95)
    assert abs(mtrl - 272.907) < 1e-2


def test_min_track_record_length_undefined_below_benchmark():
    with pytest.raises(ValueError, match="must exceed"):
        S.min_track_record_length(0.0, 0.1)


def test_expected_max_sharpe_reference():
    # V=0.04 (std 0.2), N=10 trials → SR0 ≈ 0.3150 (Gumbel two-term, Euler–Mascheroni).
    sr0 = S.expected_max_sharpe(0.04, 10)
    assert abs(sr0 - 0.3150) < 2e-3


def test_expected_max_sharpe_single_trial_is_zero():
    assert S.expected_max_sharpe(0.04, 1) == 0.0  # no selection bias with one trial


def test_deflated_sharpe_below_psr_when_trials_vary():
    # DSR deflates by the spread of trial Sharpes ⇒ DSR <= PSR(0) for the same SR.
    trials = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
    dsr = S.deflated_sharpe_ratio(0.4, trials, 500, 0.0, 3.0)
    psr0 = S.probabilistic_sharpe_ratio(0.4, 0.0, 500, 0.0, 3.0)
    assert 0.0 <= dsr <= psr0


def test_return_moments_normal_sample():
    r = np.random.default_rng(11).normal(0.0, 1.0, 20000)
    n, sr, skew, kurt = S.return_moments(r)
    assert n == 20000
    assert abs(skew) < 0.1
    assert abs(kurt - 3.0) < 0.2  # raw kurtosis of a Normal ≈ 3


def test_average_ranks_handles_ties():
    # values [10, 30, 20, 30] → ranks [1, 3.5, 2, 3.5] (the two 30s share rank (3+4)/2)
    ranks = S._average_ranks(np.array([10.0, 30.0, 20.0, 30.0]))
    assert ranks.tolist() == [1.0, 3.5, 2.0, 3.5]


def test_pbo_is_zero_when_is_best_is_always_oos_best():
    # config j has a uniformly larger mean than j-1 across ALL periods ⇒ the IS winner is
    # always the OOS winner ⇒ never overfit ⇒ PBO = 0.
    rng = np.random.default_rng(3)
    t, n = 240, 8
    M = rng.normal(0, 1, (t, n)) + np.arange(n)[None, :] * 2.0
    res = S.probability_of_backtest_overfitting(M, n_partitions=10)
    assert res["pbo"] == 0.0


def test_pbo_is_one_when_is_best_is_always_oos_worst():
    # Two perfectly anti-correlated configs over 2 chunks: config 0 wins chunk 0 and loses
    # chunk 1; config 1 is its mirror. With n_partitions=2 the only in-sample subsets are the
    # single chunks, so the in-sample winner is unambiguously the out-of-sample loser ⇒ PBO = 1.
    t = 200
    rows = np.arange(t)
    chunk = rows // (t // 2)  # 2 contiguous chunks
    wiggle = 0.001 * np.sin(rows)  # identical for both configs ⇒ keeps std>0 within a chunk
    first = (chunk == 0).astype(float) * 2.0 - 1.0  # +1 on chunk 0, -1 on chunk 1
    M = np.column_stack([first + wiggle, -first + wiggle])
    res = S.probability_of_backtest_overfitting(M, n_partitions=2)
    assert res["pbo"] == 1.0


def test_pbo_requires_two_configs():
    with pytest.raises(ValueError, match=">= 2 config"):
        S.probability_of_backtest_overfitting(np.zeros((100, 1)))
