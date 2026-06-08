"""Shared synthetic MBP-1 fixtures for the microstructure tests.

Underscore-prefixed so pytest does not collect it; imported directly by the test
modules (pytest puts the tests dir on sys.path in default import mode). Everything
here is deterministic — no Databento, no network, no credits.
"""

from __future__ import annotations

from datetime import UTC, datetime

from options_system.microstructure.bars import BookEvent
from options_system.microstructure.config import Instrument

# A Monday inside RTH (10:00 ET = 14:00 UTC, EDT). minutes-from-midnight ET = 600.
_BASE = datetime(2025, 6, 2, 14, 0, 0, tzinfo=UTC)
BASE_NS = int(_BASE.timestamp()) * 1_000_000_000
DAY_NS = 86_400 * 1_000_000_000

# A small test instrument: multiplier 1 and a tiny threshold so bars close on
# predictable traded notional (price * size * 1).
TEST_INST = Instrument(
    symbol="T",
    continuous_symbol="T.v.0",
    exec_symbol="t",
    multiplier=1.0,
    tick_size=0.25,
    dollar_threshold=5000.0,
)


def ts(sec: float = 0.0, *, day: int = 0) -> int:
    """Nanoseconds at the base instant + ``sec`` seconds (+ ``day`` whole days)."""
    return BASE_NS + day * DAY_NS + int(round(sec * 1e9))


def ev(
    ts_ns: int,
    *,
    iid: int = 1,
    bid0: float = 100.0,
    ask0: float = 100.25,
    bsz: float = 10,
    asz: float = 10,
    trade: tuple[float, float, int] | None = None,
) -> BookEvent:
    """A top-of-book BookEvent. ``trade=(price, size, sign)`` makes it a trade."""
    is_trade = trade is not None
    price, size, sign = trade if trade is not None else (float("nan"), 0.0, 0)
    return BookEvent(
        ts_ns=ts_ns,
        instrument_id=iid,
        is_trade=is_trade,
        trade_price=price,
        trade_size=size,
        trade_sign=sign,
        bid_px=bid0,
        bid_sz=float(bsz),
        ask_px=ask0,
        ask_sz=float(asz),
    )


def comoving_stream(n_bars: int) -> list[BookEvent]:
    """A stream where buy pressure (positive OFI) coincides with a rising mid and
    sell pressure with a falling mid — the OFI↔Δmid stylized fact, by construction.

    Each bar: an open snapshot, a directional book move (up = both sides +1 tick =>
    OFI>0 & Δmid>0; down = both sides -1 tick => OFI<0 & Δmid<0), then a trade that
    crosses the 5000 dollar threshold (price 100 * size 50). Direction alternates.
    """
    events: list[BookEvent] = []
    t = 0.0
    mid = 100.0
    for i in range(n_bars):
        d = 1 if i % 2 == 0 else -1
        bid0, ask0 = mid - 0.125, mid + 0.125
        events.append(ev(ts(t), bid0=bid0, ask0=ask0))
        t += 0.001
        bid0b, ask0b = bid0 + 0.25 * d, ask0 + 0.25 * d
        events.append(ev(ts(t), bid0=bid0b, ask0=ask0b))
        t += 0.001
        events.append(ev(ts(t), bid0=bid0b, ask0=ask0b, trade=(100.0, 50.0, d)))
        t += 0.001
        mid = (bid0b + ask0b) / 2
    return events
