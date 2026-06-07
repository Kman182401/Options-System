"""Purge + embargo core, and the K-fold / walk-forward splitters built on it.

The purge/embargo arithmetic is checked against a hand-computed reference; the
splitters are checked for the structural invariants that make them leak-safe (no
test index in its own train; chronological ordering for walk-forward).
"""

from __future__ import annotations

import numpy as np

from options_system.validation._purge import (
    embargo_bars_from_pct,
    purge_embargo_counts,
    train_indices,
)
from options_system.validation.purged_kfold import PurgedKFold
from options_system.validation.walk_forward import WalkForward


def _times(n, span):
    """t0 = 0..n-1, t1 = t0 + span (each label resolves `span` bars later)."""
    t0 = np.arange(n)
    t1 = t0 + span
    return t0, t1


def test_purge_matches_hand_computed_reference():
    # n=10, labels span 1 bar: t0=[0..9], t1=[1..10]. Test fold = positions {4,5}.
    t0, t1 = _times(10, span=1)
    test_idx = np.array([4, 5])
    # Test interval bounds: seg_start = t0[4] = 4, seg_end = max(t1[4], t1[5]) = 6.
    # Purge train i with (t0_i <= 6) and (t1_i >= 4):
    #   i=3 (3,4): 4>=4 ✓ → purge;  i=6 (6,7): 6<=6, 7>=4 ✓ → purge.
    #   i∈{0,1,2,7,8,9} survive the purge.
    train_no_embargo = train_indices(t0, t1, test_idx, n=10, embargo_bars=0)
    assert train_no_embargo.tolist() == [0, 1, 2, 7, 8, 9]

    # Embargo of 2 bars after the block (positions 6,7): 6 already purged, 7 newly dropped.
    train_embargo = train_indices(t0, t1, test_idx, n=10, embargo_bars=2)
    assert train_embargo.tolist() == [0, 1, 2, 8, 9]

    counts = purge_embargo_counts(t0, t1, test_idx, n=10, embargo_bars=2)
    assert counts == {"n_train": 5, "n_test": 2, "n_purged": 2, "n_embargoed": 1}


def test_no_overlap_no_purge():
    # Zero-length labels (t1 == t0) never overlap a neighbouring fold ⇒ nothing purged.
    t0 = np.arange(20)
    t1 = t0.copy()
    test_idx = np.array([8, 9, 10])
    train = train_indices(t0, t1, test_idx, n=20, embargo_bars=0)
    assert set(train) == set(range(20)) - {8, 9, 10}


def test_embargo_bars_from_pct():
    assert embargo_bars_from_pct(0.0, 1000) == 0
    assert embargo_bars_from_pct(0.01, 1000) == 10
    assert embargo_bars_from_pct(0.011, 1000) == 11  # ceil


def test_purged_kfold_no_test_index_in_train_and_full_coverage():
    n, span = 600, 20
    t0, t1 = _times(n, span)
    cv = PurgedKFold(n_splits=6, t0=t0, t1=t1, embargo_bars=10)
    assert cv.get_n_splits() == 6
    seen_test: set[int] = set()
    for train_idx, test_idx in cv.split():
        assert not (set(train_idx) & set(test_idx))  # disjoint: never train on the test fold
        # purge removed every train sample whose window overlaps the test interval
        seg_start, seg_end = t0[test_idx].min(), t1[test_idx].max()
        for i in train_idx:
            assert not (t0[i] <= seg_end and t1[i] >= seg_start)
        seen_test |= set(test_idx)
    assert seen_test == set(range(n))  # every sample is tested exactly once across folds


def test_purged_kfold_reports_purged_counts():
    t0, t1 = _times(300, span=30)
    cv = PurgedKFold(n_splits=5, t0=t0, t1=t1, embargo_bars=5)
    report = cv.split_report()
    assert len(report) == 5
    # overlapping labels (span 30) ⇒ each interior fold purges a non-trivial number of rows
    assert all(r["n_purged"] > 0 for r in report[1:-1])
    for r in report:
        assert r["n_train"] + r["n_test"] + r["n_purged"] + r["n_embargoed"] <= 300


def test_walk_forward_anchored_is_strictly_chronological():
    n, span = 700, 15
    t0, t1 = _times(n, span)
    wf = WalkForward(n_splits=5, t0=t0, t1=t1, scheme="anchored", embargo_bars=5)
    steps = list(wf.split())
    assert len(steps) >= 1
    for train_idx, test_idx in steps:
        # every training sample lies strictly before the test block in time (no future in train)
        assert t0[train_idx].max() < t0[test_idx].min()
        # purge guarantees no train label window bleeds into the test block
        assert t1[train_idx].max() < t0[test_idx].min()


def test_walk_forward_rolling_train_is_subset_of_preceding_block():
    n, span = 700, 10
    t0, t1 = _times(n, span)
    folds = [f for f in np.array_split(np.arange(n), 5 + 1) if f.size]
    wf = WalkForward(n_splits=5, t0=t0, t1=t1, scheme="rolling", embargo_bars=0)
    for i, (train_idx, _test) in enumerate(wf.split(), start=1):
        assert set(train_idx).issubset(set(folds[i - 1].tolist()))
