"""Core purge + embargo primitive — the single source of truth for leak-free splits.

López de Prado, *Advances in Financial Machine Learning* ch. 7. Each sample ``i``
carries a label interval ``[t0_i, t1_i]`` (event time → barrier resolution time).
Given a set of TEST positions, this returns the TRAIN positions that cannot leak
into the test fold:

* **PURGE** — drop any training sample whose ``[t0, t1]`` overlaps a contiguous
  test segment's bounding interval ``[t0_min, t1_max]``. Two intervals ``[a, b]``
  and ``[c, d]`` overlap iff ``a <= d and c <= b``. This catches train labels that
  were still resolving while the test window was live (information bleed in either
  direction).
* **EMBARGO** — drop a further ``embargo_bars`` training samples positioned
  immediately *after* each contiguous test segment. Embargo is forward-only: the
  purge already handles the overlap/before side; the embargo kills residual
  serial-correlation leakage from the test block into the bars that follow it.

Samples are assumed sorted ascending by ``t0`` (the labeling and feature builders
emit ``t0``/``ts_event``-sorted tables; callers sort defensively). Every splitter
in this package — purged K-fold, CPCV, walk-forward — delegates here so the
leakage logic lives in exactly one place and is tested once.
"""

from __future__ import annotations

import math

import numpy as np


def embargo_bars_from_pct(embargo_pct: float, n: int) -> int:
    """Translate an ``embargo_pct`` fraction of total bars into an integer bar count."""
    if embargo_pct <= 0.0 or n <= 0:
        return 0
    return int(math.ceil(embargo_pct * n))


def _contiguous_runs(positions: np.ndarray) -> list[tuple[int, int]]:
    """Split sorted unique positions into (start, end) inclusive contiguous runs."""
    if positions.size == 0:
        return []
    runs: list[tuple[int, int]] = []
    run_start = int(positions[0])
    prev = run_start
    for raw in positions[1:]:
        p = int(raw)
        if p == prev + 1:
            prev = p
            continue
        runs.append((run_start, prev))
        run_start = prev = p
    runs.append((run_start, prev))
    return runs


def _keep_masks(
    t0: np.ndarray,
    t1: np.ndarray,
    test_idx: np.ndarray,
    n: int,
    embargo_bars: int,
    candidates: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (keep, purged, embargoed) boolean masks of length ``n``.

    ``keep`` = leak-free train positions. ``purged`` / ``embargoed`` are the
    candidate (non-test, eligible) positions removed by each mechanism, so callers
    can report exact drop accounting.
    """
    t0 = np.asarray(t0)
    t1 = np.asarray(t1)
    test_idx = np.unique(np.asarray(test_idx, dtype=np.int64))

    if candidates is None:
        eligible = np.ones(n, dtype=bool)
    else:
        eligible = np.zeros(n, dtype=bool)
        eligible[np.asarray(candidates, dtype=np.int64)] = True
    eligible[test_idx] = False  # test rows are never train

    keep = eligible.copy()
    purged = np.zeros(n, dtype=bool)
    embargoed = np.zeros(n, dtype=bool)

    for lo, hi in _contiguous_runs(test_idx):
        seg_start = t0[lo]  # earliest event start in the segment (t0-sorted ⇒ first position)
        seg_end = t1[lo : hi + 1].max()  # latest label resolution in the segment
        overlap = (t0 <= seg_end) & (t1 >= seg_start) & keep
        keep &= ~overlap
        purged |= overlap
        if embargo_bars > 0:
            emb_lo = hi + 1
            emb_hi = min(n, hi + 1 + embargo_bars)
            if emb_lo < emb_hi:
                window = np.zeros(n, dtype=bool)
                window[emb_lo:emb_hi] = True
                emb = window & keep
                keep &= ~emb
                embargoed |= emb

    return keep, purged, embargoed


def train_indices(
    t0: np.ndarray,
    t1: np.ndarray,
    test_idx: np.ndarray,
    n: int,
    embargo_bars: int = 0,
    candidates: np.ndarray | None = None,
) -> np.ndarray:
    """Leak-free TRAIN positions for a given ``test_idx`` (purged + embargoed).

    ``candidates`` restricts which positions are eligible to be train (used by
    walk-forward to forbid future folds); ``None`` means all non-test positions.
    """
    keep, _, _ = _keep_masks(t0, t1, test_idx, n, embargo_bars, candidates)
    return np.flatnonzero(keep)


def purge_embargo_counts(
    t0: np.ndarray,
    t1: np.ndarray,
    test_idx: np.ndarray,
    n: int,
    embargo_bars: int = 0,
    candidates: np.ndarray | None = None,
) -> dict[str, int]:
    """Exact drop accounting for one fold: train / test / purged / embargoed counts."""
    keep, purged, embargoed = _keep_masks(t0, t1, test_idx, n, embargo_bars, candidates)
    return {
        "n_train": int(keep.sum()),
        "n_test": int(np.unique(test_idx).size),
        "n_purged": int(purged.sum()),
        "n_embargoed": int(embargoed.sum()),
    }
