"""Sample weights: concurrency, average uniqueness, decay vs hand-computed (Task 4)."""

from __future__ import annotations

import numpy as np

from options_system.labeling.weights import (
    average_uniqueness,
    concurrency,
    sample_weights,
    time_decay_factors,
)


def test_concurrency_two_overlapping_labels():
    # label0 spans bars [0,4], label1 spans [2,6]
    starts = np.array([0, 2])
    ends = np.array([4, 6])
    conc = concurrency(starts, ends, 7)
    # bars 0,1 -> 1 ; bars 2,3,4 -> 2 ; bars 5,6 -> 1
    assert conc.tolist() == [1, 1, 2, 2, 2, 1, 1]


def test_average_uniqueness_matches_hand_reference():
    starts = np.array([0, 2])
    ends = np.array([4, 6])
    conc = concurrency(starts, ends, 7)
    au = average_uniqueness(starts, ends, conc)
    # label0 over bars0-4: (1+1+.5+.5+.5)/5 = 0.7 ; label1 over bars2-6: (.5+.5+.5+1+1)/5 = 0.7
    assert np.allclose(au, [0.7, 0.7])


def test_non_overlapping_labels_are_fully_unique():
    starts = np.array([0, 5])
    ends = np.array([2, 7])
    conc = concurrency(starts, ends, 8)
    au = average_uniqueness(starts, ends, conc)
    assert np.allclose(au, [1.0, 1.0])


def test_three_way_overlap_uniqueness():
    # all three cover bar 2; only their own elsewhere
    starts = np.array([0, 1, 2])
    ends = np.array([2, 3, 4])
    conc = concurrency(starts, ends, 5)
    assert conc.tolist() == [1, 2, 3, 2, 1]
    au = average_uniqueness(starts, ends, conc)
    # label0 bars0-2: (1 + 1/2 + 1/3)/3 = (11/6)/3 = 0.6111...
    # label1 bars1-3: (1/2 + 1/3 + 1/2)/3 = (4/3)/3 = 0.4444...
    # label2 bars2-4: (1/3 + 1/2 + 1)/3 = (11/6)/3 = 0.6111...
    assert np.allclose(au, [11 / 18, 4 / 9, 11 / 18])


def test_sample_weights_normalize_to_mean_one():
    starts = np.array([0, 2, 10])
    ends = np.array([4, 6, 14])
    out = sample_weights(starts, ends)
    assert np.isclose(out["weight"].mean(), 1.0)
    assert (out["weight"] > 0).all()


def test_uniqueness_return_scaling():
    starts = np.array([0, 5])
    ends = np.array([2, 7])  # fully unique -> avg_uniq = 1,1
    returns = np.array([0.01, 0.03])
    out = sample_weights(starts, ends, returns=returns, scheme="uniqueness_return")
    # weights proportional to |return|: ratio 1:3, normalized to mean 1 -> [0.5, 1.5]
    assert np.allclose(out["weight"], [0.5, 1.5])


def test_time_decay_no_decay_when_c_is_one():
    au = np.array([0.7, 0.7, 0.5])
    ends = np.array([4, 6, 14])
    assert np.allclose(time_decay_factors(au, ends, 1.0), [1.0, 1.0, 1.0])


def test_time_decay_downweights_older_labels():
    au = np.array([1.0, 1.0, 1.0])
    ends = np.array([0, 1, 2])  # oldest -> newest
    decay = time_decay_factors(au, ends, 0.0)
    # c=0: linear from near 0 (oldest) to 1 (newest), strictly increasing
    assert decay[0] < decay[1] < decay[2]
    assert np.isclose(decay[2], 1.0)


def test_empty_input_safe():
    out = sample_weights(np.array([], dtype=int), np.array([], dtype=int))
    assert out["weight"].size == 0 and out["avg_uniqueness"].size == 0
