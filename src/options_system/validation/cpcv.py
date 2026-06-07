"""Combinatorial Purged Cross-Validation (López de Prado, AFML ch. 12).

Plain K-fold gives one out-of-sample path, and with overlapping, low-uniqueness
labels (avg uniqueness ≈ 0.23 here) that single number is high variance and easy
to fool. CPCV instead splits the ``t0``-sorted samples into ``n_groups``
contiguous groups and tests on **every** combination of ``test_groups`` of them.
That yields ``C(n_groups, test_groups)`` purged + embargoed train/test splits and,
by recombining the held-out groups, ``C(n_groups-1, test_groups-1)`` distinct
out-of-sample **paths** — a whole *distribution* of OOS performance to judge a
model honestly.

* Splits (sklearn-style) — :meth:`CombinatorialPurgedCV.split`.
* Path bookkeeping — :meth:`CombinatorialPurgedCV.assign_paths` maps each
  ``(split_index, group_id)`` test block to the path it belongs to, so the harness
  can stitch per-group OOS predictions into ``n_paths`` complete series.

Special case ``test_groups == 1`` reduces to purged K-fold: ``C(N,1) = N`` splits,
``C(N-1,0) = 1`` path.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from itertools import combinations

import numpy as np

from ._purge import train_indices


class CombinatorialPurgedCV:
    """Combinatorial Purged CV splitter with out-of-sample path reconstruction."""

    def __init__(
        self,
        n_groups: int,
        test_groups: int,
        t0: np.ndarray,
        t1: np.ndarray,
        embargo_bars: int = 0,
    ) -> None:
        if n_groups < 2:
            raise ValueError(f"n_groups must be >= 2, got {n_groups}")
        if not (0 < test_groups < n_groups):
            raise ValueError(f"test_groups must satisfy 0 < k < n_groups, got {test_groups}")
        t0 = np.asarray(t0)
        t1 = np.asarray(t1)
        if t0.shape != t1.shape or t0.ndim != 1:
            raise ValueError("t0 and t1 must be 1-D arrays of equal length")
        if n_groups > t0.shape[0]:
            raise ValueError(f"n_groups ({n_groups}) > n_samples ({t0.shape[0]})")
        if embargo_bars < 0:
            raise ValueError(f"embargo_bars must be >= 0, got {embargo_bars}")
        self.n_groups = n_groups
        self.test_groups = test_groups
        self.t0 = t0
        self.t1 = t1
        self.embargo_bars = embargo_bars
        # Contiguous, near-equal groups over the t0-sorted samples.
        self.groups: list[np.ndarray] = list(np.array_split(np.arange(t0.shape[0]), n_groups))
        # Deterministic combination order (used for both split() and path assignment).
        self.combos: list[tuple[int, ...]] = list(combinations(range(n_groups), test_groups))

    @property
    def n_paths(self) -> int:
        """Number of reconstructed out-of-sample paths = C(n_groups-1, test_groups-1)."""
        return math.comb(self.n_groups - 1, self.test_groups - 1)

    def get_n_splits(self, X: object = None, y: object = None, groups: object = None) -> int:
        """Number of train/test splits = C(n_groups, test_groups) (sklearn API)."""
        return len(self.combos)

    def split(
        self, X: object = None, y: object = None, groups: object = None
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield ``(train_idx, test_idx)`` for each group combination, purged + embargoed."""
        n = self.t0.shape[0]
        for combo in self.combos:
            test_idx = np.concatenate([self.groups[g] for g in combo])
            test_idx.sort()
            train_idx = train_indices(self.t0, self.t1, test_idx, n, self.embargo_bars)
            yield train_idx, test_idx

    def assign_paths(self) -> tuple[int, dict[tuple[int, int], int]]:
        """Map each tested ``(split_index, group_id)`` to its out-of-sample path index.

        Every group is a test group in exactly ``n_paths`` of the splits; the j-th
        such occurrence (in combination order) is assigned to path j. The harness
        gathers, for each path, the OOS prediction of every group from the split
        that path drew it from, then concatenates groups in time order to form one
        full-length OOS series. Returns ``(n_paths, mapping)``.
        """
        occurrence = dict.fromkeys(range(self.n_groups), 0)
        mapping: dict[tuple[int, int], int] = {}
        for split_index, combo in enumerate(self.combos):
            for g in combo:
                mapping[(split_index, g)] = occurrence[g]
                occurrence[g] += 1
        return self.n_paths, mapping
