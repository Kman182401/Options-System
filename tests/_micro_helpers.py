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


# A second small instrument (different symbol) for multi-symbol parallel tests.
TEST_INST_U = Instrument(
    symbol="U",
    continuous_symbol="U.v.0",
    exec_symbol="u",
    multiplier=1.0,
    tick_size=0.25,
    dollar_threshold=5000.0,
)


def partial_tail_stream(n_full: int, *, day: int = 0) -> list[BookEvent]:
    """``n_full`` threshold-closing trades (each 100*60 = 6000 > 5000 ⇒ one bar
    apiece) followed by ONE sub-threshold trade (100*10 = 1000) — so the stream ends
    on a trailing **partial** bar (``bar_complete=False``)."""
    events = [
        ev(ts(i * 0.001, day=day), trade=(100.0, 60.0, 1 if i % 2 == 0 else -1))
        for i in range(n_full)
    ]
    events.append(ev(ts(n_full * 0.001, day=day), trade=(100.0, 10.0, 1)))  # leftover partial
    return events


def quote_only_then_trades_stream(*, day: int = 0) -> list[BookEvent]:
    """A mix of **quote-only** events (book moves, no trade) and trades. The opening
    quote-only events exercise the TWA / OFI accumulators with no trade; trades then
    close two bars; a trailing sub-threshold trade leaves a partial bar."""
    events: list[BookEvent] = []
    t = 0.0
    for k in range(4):  # quote-only book moves (no trades) -> OFI/TWA accumulate
        events.append(ev(ts(t, day=day), bid0=100.0 + 0.25 * k, ask0=100.25 + 0.25 * k))
        t += 0.001
    for _ in range(2):  # two threshold-closing trades (6000 each)
        events.append(ev(ts(t, day=day), trade=(100.0, 60.0, 1)))
        t += 0.001
    events.append(ev(ts(t, day=day), trade=(100.0, 15.0, -1)))  # trailing partial (1500)
    return events


def roll_within_day_stream(*, day: int = 0) -> list[BookEvent]:
    """A contract **roll** inside one session: two sub-threshold trades on contract
    ``iid=1`` (4000 < 5000, so no threshold close), then ``iid=2`` appears — the
    boundary closes the contract-1 bar and severs order-flow continuity, then
    contract-2 trades accumulate. Proves boundary behaviour is preserved per-unit."""
    return [
        ev(ts(0.000, day=day), iid=1, bid0=100.0, ask0=100.25, trade=(100.0, 20.0, 1)),
        ev(ts(0.001, day=day), iid=1, bid0=100.25, ask0=100.50, trade=(100.0, 20.0, 1)),
        ev(ts(0.002, day=day), iid=2, bid0=200.0, ask0=200.25, trade=(200.0, 20.0, 1)),
        ev(ts(0.003, day=day), iid=2, bid0=200.25, ask0=200.50, trade=(200.0, 15.0, -1)),
    ]


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
