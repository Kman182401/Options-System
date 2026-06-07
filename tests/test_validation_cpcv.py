"""Combinatorial Purged CV: split/path counts, path completeness, leak-safety."""

from __future__ import annotations

import math

import numpy as np

from options_system.validation.cpcv import CombinatorialPurgedCV


def _times(n, span):
    t0 = np.arange(n)
    t1 = t0 + span
    return t0, t1


def test_split_and_path_counts_match_combinatorics():
    t0, t1 = _times(600, 10)
    cv = CombinatorialPurgedCV(n_groups=6, test_groups=2, t0=t0, t1=t1, embargo_bars=5)
    assert cv.get_n_splits() == math.comb(6, 2) == 15
    assert cv.n_paths == math.comb(5, 1) == 5


def test_purged_kfold_special_case_one_test_group():
    t0, t1 = _times(500, 8)
    cv = CombinatorialPurgedCV(n_groups=5, test_groups=1, t0=t0, t1=t1)
    assert cv.get_n_splits() == 5
    assert cv.n_paths == 1  # C(4,0)


def test_path_assignment_is_complete_and_consistent():
    t0, t1 = _times(600, 10)
    cv = CombinatorialPurgedCV(n_groups=6, test_groups=2, t0=t0, t1=t1)
    n_paths, mapping = cv.assign_paths()
    assert n_paths == 5
    # every (split, group) test block maps to a path; total entries = k * C(N,k)
    assert len(mapping) == cv.test_groups * cv.get_n_splits()
    # each path collects each group exactly once → a full-length OOS series
    for path_j in range(n_paths):
        groups_in_path = sorted(g for (_s, g), p in mapping.items() if p == path_j)
        assert groups_in_path == list(range(cv.n_groups))
    # each group is tested in exactly n_paths splits
    for g in range(cv.n_groups):
        assert sum(1 for (_s, gg) in mapping if gg == g) == n_paths


def test_splits_are_leak_safe_and_deterministic():
    t0, t1 = _times(600, 12)
    cv = CombinatorialPurgedCV(n_groups=6, test_groups=2, t0=t0, t1=t1, embargo_bars=6)
    first = [(tuple(tr), tuple(te)) for tr, te in cv.split()]
    second = [(tuple(tr), tuple(te)) for tr, te in cv.split()]
    assert first == second  # deterministic
    # leak-safety holds per test GROUP (test groups may be non-adjacent): no training
    # sample's [t0,t1] overlaps any selected test group's interval.
    for split_index, (train_idx, test_idx) in enumerate(cv.split()):
        assert not (set(train_idx) & set(test_idx))
        for g in cv.combos[split_index]:
            gi = cv.groups[g]
            gs, ge = t0[gi].min(), t1[gi].max()
            assert all(not (t0[i] <= ge and t1[i] >= gs) for i in train_idx)
