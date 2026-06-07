"""Walk-forward validation — anchored and rolling (López de Prado, AFML ch. 7/11).

The realistic "train on the past, test on the next slice, roll forward" path. It
complements CPCV: CPCV gives a distribution over recombined paths, walk-forward
gives the single chronological path you would actually have lived through, which
is the honest picture of deployment-time performance.

Two schemes:

* ``anchored`` — the training window grows from the start of history; each step
  trains on *everything before* the test block.
* ``rolling`` — the training window is the single fold immediately preceding the
  test block (a fixed-ish lookback), so the model only ever sees recent data.

Either way the boundary is **purged**: training samples whose ``[t0, t1]`` label
window overlaps the test block are dropped, which also opens the realistic gap
between train-end and test-start. (A forward embargo is wired through for
symmetry with the other splitters; with strictly-past training it is a no-op,
because there are no train samples after the test block to embargo.)
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from ._purge import train_indices


class WalkForward:
    """Anchored or rolling sequential train→test splitter with purge + embargo."""

    def __init__(
        self,
        n_splits: int,
        t0: np.ndarray,
        t1: np.ndarray,
        scheme: str = "anchored",
        min_train_bars: int = 0,
        embargo_bars: int = 0,
    ) -> None:
        if n_splits < 2:
            raise ValueError(f"n_splits must be >= 2, got {n_splits}")
        scheme = scheme.strip().lower()
        if scheme not in {"anchored", "rolling"}:
            raise ValueError(f"scheme must be 'anchored' or 'rolling', got {scheme!r}")
        t0 = np.asarray(t0)
        t1 = np.asarray(t1)
        if t0.shape != t1.shape or t0.ndim != 1:
            raise ValueError("t0 and t1 must be 1-D arrays of equal length")
        if n_splits + 1 > t0.shape[0]:
            raise ValueError(f"n_splits+1 ({n_splits + 1}) > n_samples ({t0.shape[0]})")
        if min_train_bars < 0 or embargo_bars < 0:
            raise ValueError("min_train_bars and embargo_bars must be >= 0")
        self.n_splits = n_splits
        self.t0 = t0
        self.t1 = t1
        self.scheme = scheme
        self.min_train_bars = min_train_bars
        self.embargo_bars = embargo_bars

    def get_n_splits(self, X: object = None, y: object = None, groups: object = None) -> int:
        """Maximum number of sequential steps (sklearn API). Steps below the minimum
        training size are skipped at iteration time, so the realised count can be lower."""
        return self.n_splits

    def split(
        self, X: object = None, y: object = None, groups: object = None
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield chronological ``(train_idx, test_idx)`` steps, purged + embargoed."""
        n = self.t0.shape[0]
        # n_splits+1 contiguous blocks: block 0 seeds the first train window, blocks
        # 1..n_splits are the successive test folds.
        folds = [f for f in np.array_split(np.arange(n), self.n_splits + 1) if f.size]
        for i in range(1, len(folds)):
            test_idx = folds[i]
            # anchored: train on everything before the test fold; rolling: only the
            # single immediately-preceding fold (a fixed-ish lookback window).
            candidates = np.concatenate(folds[:i]) if self.scheme == "anchored" else folds[i - 1]
            train_idx = train_indices(
                self.t0, self.t1, test_idx, n, self.embargo_bars, candidates=candidates
            )
            if train_idx.size < self.min_train_bars:
                continue
            yield train_idx, test_idx
