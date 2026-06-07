"""The teeth test — proof the framework actually catches leakage.

Two complementary assertions:

1. **Mechanism** — a training sample whose label window overlaps the test fold is
   *purged*, while a sample that resolves before the test is *retained*. This is the
   direct analogue of the feature-leakage teeth test (``test_features_leakage.py``):
   the leak path fails the check, the safe path passes.
2. **Skill collapse** — on a synthetic dataset whose labels are *forward returns of a
   random walk* (no global predictability whatsoever), a flexible model shows
   inflated out-of-sample skill WITHOUT purging — purely from train labels whose
   windows overlap the test fold — and that skill collapses to chance WITH
   purging + embargo. The fold size is set ≤ the label horizon so the leak reaches
   every test point, not just the boundary.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from options_system.validation._purge import train_indices
from options_system.validation.purged_kfold import PurgedKFold


def test_purge_removes_overlapping_train_sample_keeps_safe_one():
    # Minute grid. Test fold = positions {5,6}. Each label resolves 3 bars after its t0.
    t0 = np.arange(10)
    t1 = t0 + 3
    test_idx = np.array([5, 6])
    # Test interval = [t0[5], max(t1[5],t1[6])] = [5, 9].
    #   position 3 (window [3,6]) OVERLAPS the test interval → must be PURGED (leak path).
    #   position 1 (window [1,4]) resolves at 4 < 5 → no overlap → must be RETAINED (safe).
    train = set(train_indices(t0, t1, test_idx, n=10, embargo_bars=0).tolist())
    assert 3 not in train, "overlapping train sample leaked into the train set"
    assert 1 in train, "non-overlapping (safe) train sample was wrongly dropped"
    # A naive splitter that ignores t1 would keep position 3 — that is exactly the leak purge fixes.
    naive_train = set(range(10)) - set(test_idx.tolist())
    assert 3 in naive_train


def _random_walk_labels(n, horizon, seed):
    """Forward-return-sign labels of a random walk: unpredictable except via window overlap."""
    rng = np.random.default_rng(seed)
    t0 = datetime(2024, 1, 2, tzinfo=UTC)
    ts = np.array(
        [np.datetime64((t0 + timedelta(minutes=i)).replace(tzinfo=None), "us") for i in range(n)]
    )
    z = np.cumsum(rng.normal(0.0, 1.0, n + horizon))
    fwd = z[np.arange(n) + horizon] - z[np.arange(n)]
    # Median threshold ⇒ ~balanced classes, so KNN's chance level is 0.5 (not the majority rate).
    y = (fwd > np.median(fwd)).astype(int)
    x = np.arange(n, dtype=float).reshape(-1, 1)  # index feature ⇒ KNN neighbour = time neighbour
    t1 = ts[np.minimum(np.arange(n) + horizon, n - 1)]
    return ts, t1, x, y


def _knn_cv_accuracy(ts, t1, x, y, *, n_splits, embargo_bars):
    cv = PurgedKFold(n_splits, ts, t1, embargo_bars=embargo_bars)
    accs = []
    for train_idx, test_idx in cv.split():
        if train_idx.size == 0:
            continue
        knn = KNeighborsClassifier(n_neighbors=1).fit(x[train_idx], y[train_idx])
        accs.append(float((knn.predict(x[test_idx]) == y[test_idx]).mean()))
    return float(np.mean(accs))


def test_purging_collapses_leaked_skill_to_chance():
    # 2000 samples, label horizon 60 bars, 40 folds (fold size 50 ≤ horizon ⇒ overlap reaches
    # every test point). Random-walk forward returns ⇒ the ONLY signal is the overlap leak.
    n, horizon, n_splits, seed = 2000, 60, 40, 7
    ts, t1, x, y = _random_walk_labels(n, horizon, seed)
    base_rate = max(float(y.mean()), 1.0 - float(y.mean()))

    # WITHOUT purge: pretend labels are point-in-time (t1 == t0), no embargo ⇒ boundary
    # overlaps remain in train ⇒ inflated skill.
    no_purge = _knn_cv_accuracy(ts, ts.copy(), x, y, n_splits=n_splits, embargo_bars=0)
    # WITH purge: real overlapping t1 + an embargo of one horizon ⇒ leak severed ⇒ chance.
    with_purge = _knn_cv_accuracy(ts, t1, x, y, n_splits=n_splits, embargo_bars=horizon)

    assert base_rate < 0.55, "labels should be ~balanced so chance is ~0.5"
    assert no_purge > 0.70, f"expected inflated skill without purging, got {no_purge:.3f}"
    assert abs(with_purge - 0.5) < 0.07, (
        f"purged skill should collapse to chance (~0.5), got {with_purge:.3f}"
    )
    assert no_purge - with_purge > 0.20, "purging did not materially remove the leaked skill"
