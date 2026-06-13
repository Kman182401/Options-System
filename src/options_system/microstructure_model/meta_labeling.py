"""Pure meta-labeling primitives (Phase 20) — primary side, meta-label, gating.

Meta-labeling (López de Prado, *Advances in Financial ML*, Ch. 3) keeps a fixed,
deterministic **primary** that picks the side of every bet and trains a binary
**meta-model** that decides whether to *act* on it. These four functions are the
deterministic, causal pieces — no model, no cross-validation — and are unit-tested
directly so the leak-safety argument is checkable by eye:

* :func:`primary_side` — ``sign(ofi_top)`` evaluated as-of ``t0`` (a causal function of
  past data, not a fitted model). ``+1`` long / ``-1`` short / ``0`` no-side.
* :func:`meta_set_mask` — the rows that HAVE a primary side (``side != 0``); rows with
  ``ofi_top == 0`` (or a non-finite ``ofi_top``) have no side and are excluded.
* :func:`meta_label` — the binary target: ``1`` iff the realized micro-label equals the
  primary side (the primary called the correct barrier side), else ``0`` (wrong
  direction OR a ``0`` timeout). Reads ONLY the already-resolved label.
* :func:`gated_position` — the arm-M position: take the primary side when
  ``P(meta_label = 1) > tau``, otherwise flat (``0``).

The primary side and the meta-label never read ``t1``, ``ret_t1`` or any future row;
the only outcome consulted is the already-resolved ``label`` (whose own ``t1`` purge/
embargo are applied downstream exactly as in Phase 14).
"""

from __future__ import annotations

import numpy as np


def primary_side(ofi_top: np.ndarray) -> np.ndarray:
    """``sign(ofi_top)`` as-of ``t0``: ``+1`` long / ``-1`` short / ``0`` no-side.

    Positive top-of-book order-flow imbalance ⇒ buying pressure ⇒ long; negative ⇒
    short. ``ofi_top == 0`` (no imbalance) and any non-finite ``ofi_top`` (defensive;
    should not occur for a real bar) map to ``0`` — "no primary side" — and are excluded
    from the meta-set by :func:`meta_set_mask`.
    """
    o = np.asarray(ofi_top, dtype=float)
    s = np.sign(o)  # +1 / -1 / 0 (note: sign(+inf)=1, sign(-inf)=-1, sign(NaN)=NaN)
    s = np.where(np.isfinite(o), s, 0.0)  # non-finite ofi_top (NaN/±inf) -> no side
    return s.astype(int)


def meta_set_mask(side: np.ndarray) -> np.ndarray:
    """Boolean mask of rows that have a defined primary side (``side != 0``) — the meta-set."""
    return np.asarray(side).astype(int) != 0


def meta_label(label: np.ndarray, side: np.ndarray) -> np.ndarray:
    """Binary meta-target: ``1`` iff ``label == side`` (primary called the correct side).

    With ``label ∈ {-1, 0, +1}`` (lower / timeout-or-close / upper) and ``side ∈
    {-1, +1}``: a correct directional call (``label == side``) is ``1`` (true positive);
    a wrong direction OR a ``0`` timeout (the directional bet did not pay) is ``0``.
    Consults only the already-resolved ``label`` — never ``t1`` or the return.
    """
    label = np.asarray(label).astype(int)
    side = np.asarray(side).astype(int)
    return (label == side).astype(int)


def gated_position(side: np.ndarray, p_meta: np.ndarray, tau: float) -> np.ndarray:
    """Arm-M position: take the primary ``side`` when ``P(meta_label = 1) > tau``, else 0.

    ``tau`` is the fixed decision threshold (0.5, never tuned). Returns a float position
    in ``{-1.0, 0.0, +1.0}`` so the gross proxy ``position · ret_t1`` is ``side · ret`` on
    acted-on events and ``0`` when flat.
    """
    act = np.asarray(p_meta, dtype=float) > tau
    return np.where(act, np.asarray(side, dtype=float), 0.0)
