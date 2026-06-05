"""Sample weights from label overlap (López de Prado, AFML ch. 4).

Triple-barrier labels overlap in time: two events started a few bars apart share
most of their outcome window, so their labels are highly correlated. Training on
them as if they were independent over-counts the overlapping region. The fix is
to weight each label by how *unique* its window is.

Pipeline:

1. :func:`concurrency` — how many labels' ``[t0, t1]`` windows cover each bar.
2. :func:`average_uniqueness` — for label *i*, the mean of ``1 / concurrency``
   over the bars it spans (1.0 if it never overlaps anything, →0 as it piles up).
3. :func:`sample_weights` — average uniqueness, optionally scaled by the
   realized ``|return|`` (return attribution), then an optional linear
   **time-decay** (AFML 4.11), finally normalized so the weights average 1.

All positions are integer **bar indices** into the continuous series; the builder
maps each label's ``t0`` / ``t1`` to those indices. Pure NumPy, deterministic.
"""

from __future__ import annotations

import numpy as np


def concurrency(starts: np.ndarray, ends: np.ndarray, n: int) -> np.ndarray:
    """Number of label windows covering each bar in ``[0, n)``.

    ``starts`` / ``ends`` are inclusive 0-based bar positions (``ends >= starts``).
    Computed in O(n + labels) via a difference array + cumulative sum.
    """
    if starts.shape != ends.shape:
        raise ValueError("starts and ends must have the same shape")
    delta = np.zeros(n + 1, dtype=np.int64)
    np.add.at(delta, starts, 1)
    np.add.at(delta, ends + 1, -1)
    return np.cumsum(delta[:n])


def average_uniqueness(starts: np.ndarray, ends: np.ndarray, conc: np.ndarray) -> np.ndarray:
    """Average uniqueness per label = mean of ``1/concurrency`` over its bars.

    Uses a prefix sum of ``1/conc`` so each label is O(1). ``conc`` must be the
    concurrency array covering every bar any label spans (no zeros on those bars).
    """
    if np.any(conc[starts] <= 0):
        raise ValueError("concurrency must be >= 1 on every bar a label spans")
    # Bars between labels can have concurrency 0; they belong to no label's span,
    # so set their 1/conc contribution to 0 rather than letting inf poison cumsum.
    inv = np.zeros_like(conc, dtype=np.float64)
    nz = conc > 0
    inv[nz] = 1.0 / conc[nz]
    prefix = np.concatenate(([0.0], np.cumsum(inv)))
    sums = prefix[ends + 1] - prefix[starts]
    lengths = (ends - starts + 1).astype(np.float64)
    return sums / lengths


def time_decay_factors(avg_uniq: np.ndarray, ends: np.ndarray, c: float) -> np.ndarray:
    """Linear time-decay weights over cumulative uniqueness (AFML 4.11).

    Labels are ordered by ``ends`` (resolution time, oldest → newest). ``c == 1``
    means no decay (all 1.0); ``0 <= c < 1`` decays the oldest label to weight
    ``c``; ``c < 0`` zeroes out the oldest ``|c|`` fraction of cumulative
    uniqueness. Returns one factor per label in the *original* order.
    """
    if c == 1.0:
        return np.ones_like(avg_uniq)
    order = np.argsort(ends, kind="stable")
    cum = np.cumsum(avg_uniq[order])
    total = cum[-1] if cum.size else 0.0
    if total <= 0:
        return np.ones_like(avg_uniq)
    slope = (1.0 - c) / total if c >= 0 else 1.0 / ((c + 1.0) * total)
    const = 1.0 - slope * total
    decay_sorted = const + slope * cum
    decay_sorted[decay_sorted < 0] = 0.0
    decay = np.empty_like(decay_sorted)
    decay[order] = decay_sorted
    return decay


def sample_weights(
    starts: np.ndarray,
    ends: np.ndarray,
    *,
    returns: np.ndarray | None = None,
    scheme: str = "uniqueness",
    time_decay: float = 1.0,
) -> dict[str, np.ndarray]:
    """Compute ``avg_uniqueness`` and the final normalized ``weight`` per label.

    ``scheme='uniqueness_return'`` multiplies uniqueness by ``|returns|`` (return
    attribution). ``time_decay`` applies the AFML linear decay. Weights are
    normalized to average 1.0. Returns ``{"avg_uniqueness", "weight"}`` aligned to
    the input order.
    """
    if starts.size == 0:
        empty = np.asarray([], dtype=np.float64)
        return {"avg_uniqueness": empty, "weight": empty}
    lo = int(starts.min())
    s = starts - lo
    e = ends - lo
    n = int(e.max()) + 1
    conc = concurrency(s, e, n)
    avg_uniq = average_uniqueness(s, e, conc)

    base = avg_uniq.copy()
    if scheme == "uniqueness_return":
        if returns is None:
            raise ValueError("scheme='uniqueness_return' requires returns")
        base = base * np.abs(returns)

    weight = base * time_decay_factors(avg_uniq, ends, time_decay)
    total = weight.sum()
    if total > 0:
        weight = weight * (weight.size / total)  # normalize to mean 1.0
    return {"avg_uniqueness": avg_uniq, "weight": weight}
