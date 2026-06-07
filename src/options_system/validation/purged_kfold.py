"""Purged + embargoed K-fold cross-validation (López de Prado, AFML ch. 7).

An sklearn-compatible CV splitter for overlapping labels. Test folds are
contiguous blocks of the ``t0``-sorted samples; for each fold the training set
has every sample whose ``[t0, t1]`` window overlaps the test block **purged**, and
a forward **embargo** removes a further span of bars after the block. The purge
uses the label resolution times ``t1`` — the whole point is that plain K-fold,
which ignores ``t1``, leaks overlapping labels across the boundary and reports
fake skill.

Usage mirrors sklearn::

    cv = PurgedKFold(n_splits=6, t0=t0, t1=t1, embargo_bars=50)
    for train_idx, test_idx in cv.split(X):
        ...

``t0``/``t1`` are 1-D arrays aligned with ``X`` rows and assumed sorted ascending
by ``t0``. ``embargo_bars`` is an absolute bar count — translate a config
``embargo_pct`` with :func:`options_system.validation._purge.embargo_bars_from_pct`.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from ._purge import purge_embargo_counts, train_indices


class PurgedKFold:
    """sklearn-compatible purged + embargoed K-fold splitter."""

    def __init__(
        self,
        n_splits: int,
        t0: np.ndarray,
        t1: np.ndarray,
        embargo_bars: int = 0,
    ) -> None:
        if n_splits < 2:
            raise ValueError(f"n_splits must be >= 2, got {n_splits}")
        t0 = np.asarray(t0)
        t1 = np.asarray(t1)
        if t0.shape != t1.shape or t0.ndim != 1:
            raise ValueError("t0 and t1 must be 1-D arrays of equal length")
        if n_splits > t0.shape[0]:
            raise ValueError(f"n_splits ({n_splits}) > n_samples ({t0.shape[0]})")
        if embargo_bars < 0:
            raise ValueError(f"embargo_bars must be >= 0, got {embargo_bars}")
        self.n_splits = n_splits
        self.t0 = t0
        self.t1 = t1
        self.embargo_bars = embargo_bars

    def get_n_splits(self, X: object = None, y: object = None, groups: object = None) -> int:
        """Number of folds (sklearn API)."""
        return self.n_splits

    def _test_folds(self) -> list[np.ndarray]:
        """Contiguous, near-equal index blocks over the t0-sorted samples."""
        return [f for f in np.array_split(np.arange(self.t0.shape[0]), self.n_splits) if f.size]

    def split(
        self, X: object = None, y: object = None, groups: object = None
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield ``(train_idx, test_idx)`` pairs, purged + embargoed (sklearn API)."""
        n = self.t0.shape[0]
        for test_idx in self._test_folds():
            train_idx = train_indices(self.t0, self.t1, test_idx, n, self.embargo_bars)
            yield train_idx, test_idx

    def split_report(self) -> list[dict[str, int]]:
        """Per-fold drop accounting (train / test / purged / embargoed counts)."""
        n = self.t0.shape[0]
        return [
            purge_embargo_counts(self.t0, self.t1, test_idx, n, self.embargo_bars)
            for test_idx in self._test_folds()
        ]
