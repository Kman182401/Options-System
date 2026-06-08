"""Pure order-flow math — no I/O, no state, fully unit-testable.

Everything here is a plain function of book/trade scalars, so the same code paths
that run on a multi-billion-record Databento stream are exercised exactly by the
hand-built fixtures in ``tests/test_microstructure_ofi.py``.

The headline quantity is **Order-Flow Imbalance (OFI)** per Cont, Kukanov &
Stoikov (2014). Between two consecutive top-of-book states (prev -> cur), the
signed change in resting size is

    dW_bid = q_bid_cur * [P_bid_cur >= P_bid_prev]  -  q_bid_prev * [P_bid_cur <= P_bid_prev]
    dW_ask = q_ask_cur * [P_ask_cur <= P_ask_prev]  -  q_ask_prev * [P_ask_cur >= P_ask_prev]
    OFI    = dW_bid - dW_ask

Intuition: a rising bid or a withdrawn ask is buy pressure (OFI > 0); a falling
bid or a growing ask is sell pressure (OFI < 0). A bar's OFI is the sum over all
consecutive best-bid/offer transitions inside the bar.

This stage is **single-level only** (we ingest the cheap MBP-1 top-of-book
schema). Multi-level OFI across the deeper book (MLOFI) requires MBP-10 and is a
deliberate later escalation — intentionally not implemented here.
"""

from __future__ import annotations

import math


def _valid(price: float | None) -> bool:
    """A book level is present when its price is a positive, finite number.

    Databento encodes an absent level with an undefined/sentinel price; we map
    those to NaN upstream. A transition touching an absent level contributes 0
    OFI (we never invent flow from a level that does not exist on both sides).
    """
    return price is not None and math.isfinite(price) and price > 0.0


def level_ofi(
    pb_prev: float,
    qb_prev: float,
    pa_prev: float,
    qa_prev: float,
    pb_cur: float,
    qb_cur: float,
    pa_cur: float,
    qa_cur: float,
) -> float:
    """Single-level OFI contribution for one book transition (prev -> cur).

    Returns 0.0 for a level absent on either side of the transition.
    """
    if _valid(pb_prev) and _valid(pb_cur):
        d_bid = qb_cur * (pb_cur >= pb_prev) - qb_prev * (pb_cur <= pb_prev)
    else:
        d_bid = 0.0
    if _valid(pa_prev) and _valid(pa_cur):
        d_ask = qa_cur * (pa_cur <= pa_prev) - qa_prev * (pa_cur >= pa_prev)
    else:
        d_ask = 0.0
    return float(d_bid - d_ask)


def mid_price(bid_px0: float, ask_px0: float) -> float:
    """Top-of-book mid. NaN if either side is absent."""
    if _valid(bid_px0) and _valid(ask_px0):
        return (bid_px0 + ask_px0) / 2.0
    return float("nan")


def micro_price(bid_px0: float, ask_px0: float, bid_sz0: float, ask_sz0: float) -> float:
    """Size-weighted mid (a.k.a. micro-price): each side's price weighted by the
    OPPOSITE queue size, so it leans toward the side likelier to be hit.

        micro = (P_bid * Q_ask + P_ask * Q_bid) / (Q_bid + Q_ask)

    Falls back to the plain mid when both top sizes are zero.
    """
    if not (_valid(bid_px0) and _valid(ask_px0)):
        return float("nan")
    denom = bid_sz0 + ask_sz0
    if denom <= 0:
        return (bid_px0 + ask_px0) / 2.0
    return (bid_px0 * ask_sz0 + ask_px0 * bid_sz0) / denom


def book_imbalance(bid_szs: tuple[float, ...], ask_szs: tuple[float, ...]) -> float:
    """Queue imbalance over the supplied levels in [-1, 1].

        (sum_bid - sum_ask) / (sum_bid + sum_ask)

    +1 = all depth on the bid (buy pressure), -1 = all on the ask, 0 = balanced.
    Returns 0.0 when there is no depth at all.
    """
    sb = float(sum(s for s in bid_szs if math.isfinite(s)))
    sa = float(sum(s for s in ask_szs if math.isfinite(s)))
    total = sb + sa
    if total <= 0:
        return 0.0
    return (sb - sa) / total
