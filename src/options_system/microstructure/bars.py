"""Streaming dollar-bar reducer + order-flow feature assembly (single-level / MBP-1).

Two stages, both deterministic and leak-free by construction:

1. :func:`build_dollar_bars` consumes an *iterator* of :class:`BookEvent` (MBP-1
   top-of-book + trade events) and emits one raw aggregate dict per **dollar
   bar** — a bar closes the moment its accumulated traded notional (``price *
   size * multiplier``) reaches the instrument's ``dollar_threshold``. Memory is
   O(1): only the current best-bid/offer and the current bar's accumulators are
   held, so a whole day streams through without ever materialising.

2. :func:`assemble_features` turns the raw aggregates into the final, named,
   version-stamped feature frame (``microstructure_feature_version``), adding the
   causal rolling-history features.

**Causality.** Every quantity in bar *t* is a function only of events with
``ts_event <= bar t's close``. Order-flow contributions are summed over the
best-bid/offer transitions *inside* the bar (the first transition compares to the
previous event, which is strictly earlier). Bars **never span a contract roll or
a session boundary** — a change in ``instrument_id`` or RTH session date closes
the current bar and severs order-flow continuity, so no flow is attributed across
a seam. ``tests/test_microstructure_leakage.py`` proves this with a truncation-
invariance test (and a planted forward-looking leak the test rejects).

This stage is single-level (top of book) only; multi-level OFI needs MBP-10 and
is a deliberate later escalation.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

import polars as pl

from .config import Instrument, MicrostructureConfig, SessionCfg
from .ofi import book_imbalance, level_ofi, micro_price, mid_price

_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


@dataclass(frozen=True)
class BookEvent:
    """One normalised MBP-1 event: the best-bid/offer snapshot plus, if a trade,
    the trade's price/size/aggressor sign.

    Prices/sizes are floats in natural units (dollars / contracts); an absent side
    carries ``price=NaN`` and ``size=0.0``. ``trade_sign`` is +1 for a
    buy-initiated trade, -1 for sell-initiated, 0 if unknown.
    """

    ts_ns: int
    instrument_id: int
    is_trade: bool
    trade_price: float
    trade_size: float
    trade_sign: int
    bid_px: float
    bid_sz: float
    ask_px: float
    ask_sz: float


def _ns_to_dt(ts_ns: int) -> datetime:
    """Nanoseconds-since-epoch -> microsecond-resolution UTC datetime (lake dtype)."""
    return _EPOCH + timedelta(microseconds=ts_ns // 1000)


def _instant(ev: BookEvent, tick_size: float) -> dict[str, float]:
    """Instantaneous metrics computed from a single event's top-of-book snapshot."""
    m = mid_price(ev.bid_px, ev.ask_px)
    mp = micro_price(ev.bid_px, ev.ask_px, ev.bid_sz, ev.ask_sz)
    spread_ticks = (
        (ev.ask_px - ev.bid_px) / tick_size
        if (math.isfinite(ev.bid_px) and math.isfinite(ev.ask_px))
        else math.nan
    )
    return {
        "mid": m,
        "spread_ticks": spread_ticks,
        "qimb_top": book_imbalance((ev.bid_sz,), (ev.ask_sz,)),
        "depth_top": float(ev.bid_sz + ev.ask_sz),
        "micro_md": (mp - m) if (math.isfinite(mp) and math.isfinite(m)) else math.nan,
    }


def _new_bar(ev: BookEvent, sdate: date, inst: dict[str, float]) -> dict:
    return {
        "ts_open_ns": ev.ts_ns,
        "ts_close_ns": ev.ts_ns,
        "instrument_id": ev.instrument_id,
        "session_date": sdate,
        "o": None,
        "h": -math.inf,
        "l": math.inf,
        "c": None,
        "volume": 0.0,
        "dollar_volume": 0.0,
        "vwap_num": 0.0,
        "n_trades": 0,
        "n_events": 0,
        "signed_vol": 0.0,
        "mid_open": inst["mid"],
        "mid_close": inst["mid"],
        "ofi": 0.0,
        # time-weighted-average accumulators (value * dt) + total dt
        "twa_dt": 0,
        "spread_acc": 0.0,
        "qimb_top_acc": 0.0,
        "micro_md_acc": 0.0,
        "depth_top_acc": 0.0,
        # closing-snapshot values (overwritten each event -> last == close)
        "spread_close": inst["spread_ticks"],
        "qimb_top_close": inst["qimb_top"],
        "micro_md_close": inst["micro_md"],
        "rv_acc": 0.0,
        "bar_complete": True,
    }


def _accum_twa(bar: dict, bp: dict, dt: int) -> None:
    """Weight the previous (within-bar) book state ``bp`` by the time ``dt`` (ns)
    it persisted, accumulating toward the bar's time-weighted averages."""
    bar["twa_dt"] += dt

    def _add(key: str, val: float) -> None:
        if math.isfinite(val):
            bar[key] += val * dt

    _add("spread_acc", bp["spread_ticks"])
    _add("qimb_top_acc", bp["qimb_top"])
    _add("micro_md_acc", bp["micro_md"])
    _add("depth_top_acc", bp["depth_top"])


def _apply_trade(bar: dict, ev: BookEvent, multiplier: float) -> None:
    px, sz = ev.trade_price, ev.trade_size
    if bar["o"] is None:
        bar["o"] = px
    bar["h"] = max(bar["h"], px)
    bar["l"] = min(bar["l"], px)
    bar["c"] = px
    bar["volume"] += sz
    bar["vwap_num"] += px * sz
    bar["dollar_volume"] += px * sz * multiplier
    bar["signed_vol"] += ev.trade_sign * sz
    bar["n_trades"] += 1


def build_dollar_bars(
    events: Iterable[BookEvent],
    *,
    instrument: Instrument,
    session: SessionCfg,
) -> list[dict]:
    """Reduce a stream of :class:`BookEvent` to raw dollar-bar aggregate dicts.

    Bars with zero trades are dropped (a dollar bar is defined by traded value).
    The trailing partial bar (threshold not reached at end of stream) is kept and
    flagged ``bar_complete=False``.
    """
    tz = ZoneInfo(session.tz)
    rth_only = session.rth_only
    open_m = session.rth_open_min
    close_m = session.rth_close_min
    threshold = instrument.dollar_threshold
    mult = instrument.multiplier
    tick = instrument.tick_size

    def _et_parts(ts_ns: int) -> tuple[date, int, int]:
        et = _ns_to_dt(ts_ns).astimezone(tz)
        return et.date(), et.weekday(), et.hour * 60 + et.minute

    bars: list[dict] = []
    cur: dict | None = None
    ofi_prev: dict | None = None  # last event's BBO + its (instrument, session) context
    bp: dict | None = None  # within-bar previous state, for TWA + realized vol

    def _finish(bar: dict, *, complete: bool) -> None:
        bar["bar_complete"] = complete
        if bar["n_trades"] > 0:  # drop degenerate trade-less bars
            bars.append(bar)

    for ev in events:
        sdate, weekday, minutes = _et_parts(ev.ts_ns)
        if rth_only and (weekday >= 5 or not (open_m <= minutes < close_m)):
            continue

        # Boundary: a contract roll or a new session closes the current bar.
        if cur is not None and (
            ev.instrument_id != cur["instrument_id"] or sdate != cur["session_date"]
        ):
            cur["ts_close_ns"] = bp["ts"] if bp else cur["ts_close_ns"]  # last event in the bar
            _finish(cur, complete=True)
            cur, bp = None, None

        # Sever order-flow continuity across any roll/session seam.
        if ofi_prev is not None and (
            ofi_prev["instrument_id"] != ev.instrument_id or ofi_prev["session_date"] != sdate
        ):
            ofi_prev = None

        inst = _instant(ev, tick)
        if cur is None:
            cur = _new_bar(ev, sdate, inst)
            bp = {"ts": ev.ts_ns, **inst}
        else:
            assert bp is not None  # bp and cur are set/cleared together
            dt = int(ev.ts_ns - bp["ts"])
            if dt > 0:
                _accum_twa(cur, bp, dt)
                pm, cm = bp["mid"], inst["mid"]
                if math.isfinite(pm) and math.isfinite(cm) and pm > 0 and cm > 0:
                    cur["rv_acc"] += math.log(cm / pm) ** 2
            bp = {"ts": ev.ts_ns, **inst}

        if ofi_prev is not None:
            cur["ofi"] += level_ofi(
                ofi_prev["bid_px"],
                ofi_prev["bid_sz"],
                ofi_prev["ask_px"],
                ofi_prev["ask_sz"],
                ev.bid_px,
                ev.bid_sz,
                ev.ask_px,
                ev.ask_sz,
            )

        cur["ts_close_ns"] = ev.ts_ns
        cur["mid_close"] = inst["mid"]
        cur["spread_close"] = inst["spread_ticks"]
        cur["qimb_top_close"] = inst["qimb_top"]
        cur["micro_md_close"] = inst["micro_md"]
        cur["n_events"] += 1

        if ev.is_trade and math.isfinite(ev.trade_price) and ev.trade_size > 0:
            _apply_trade(cur, ev, mult)
            if cur["dollar_volume"] >= threshold:
                _finish(cur, complete=True)
                cur, bp = None, None

        ofi_prev = {
            "instrument_id": ev.instrument_id,
            "session_date": sdate,
            "bid_px": ev.bid_px,
            "bid_sz": ev.bid_sz,
            "ask_px": ev.ask_px,
            "ask_sz": ev.ask_sz,
        }

    if cur is not None:
        _finish(cur, complete=False)
    return bars


# --- feature assembly ------------------------------------------------------- #

# Canonical order-flow feature columns (each documented one line in docs/MICROSTRUCTURE.md).
_BASE_FEATURES: tuple[str, ...] = (
    "ofi_top",  # single-level (top-of-book) OFI summed over the bar — the centerpiece signal
    "qimb_top_close",  # top-of-book queue imbalance at bar close
    "qimb_top_twa",  # time-weighted top-of-book imbalance over the bar
    "signed_vol",  # signed aggressor volume (buy +, sell -) over the bar
    "trade_imbalance",  # signed_vol / traded volume in [-1, 1]
    "micro_minus_mid_close",  # micro-price minus mid at bar close (drift toward heavier side)
    "micro_minus_mid_twa",  # time-weighted micro-minus-mid over the bar
    "spread_ticks_close",  # closing bid-ask spread in ticks
    "spread_ticks_twa",  # time-weighted spread in ticks over the bar
    "depth_top_twa",  # time-weighted top-of-book total size over the bar
    "rv_intrabar",  # realized intrabar vol = sqrt(sum of squared mid log-returns)
    "duration_s",  # bar wall-clock duration in seconds (event-clock regime signal)
    "dmid",  # mid_close - mid_open (contemporaneous price change; OFI sanity target)
    "ret_bar",  # log(mid_close / mid_open) over the bar
)
# Causal rolling-history features (over the last k bars within a contract segment).
_ROLLING_FEATURES: tuple[str, ...] = (
    "ofi_top_lag1",  # ofi_top of the previous bar
    "ofi_top_roll3",  # rolling sum of ofi_top over the last k bars (incl. current)
    "signed_vol_roll3",  # rolling sum of signed_vol over the last k bars
)


def feature_names(cfg: MicrostructureConfig | None = None) -> list[str]:
    """The canonical order-flow feature column names, in stored order."""
    return [*_BASE_FEATURES, *_ROLLING_FEATURES]


def _twa(acc: float, total_dt: int, fallback: float) -> float:
    return acc / total_dt if total_dt > 0 else fallback


def assemble_features(
    raw_bars: list[dict],
    *,
    symbol: str,
    cfg: MicrostructureConfig,
    contract_map: dict[int, str] | None = None,
    source: str = "databento",
) -> pl.DataFrame:
    """Turn raw dollar-bar aggregates into the final, version-stamped feature frame.

    Adds the causal rolling features and stamps ``microstructure_feature_version``.
    Returns an empty (schema-bearing) frame when ``raw_bars`` is empty.
    """
    cmap = contract_map or {}
    rows: list[dict] = []
    for b in raw_bars:
        volume = b["volume"]
        mid_o, mid_c = b["mid_open"], b["mid_close"]
        ret_bar = (
            math.log(mid_c / mid_o)
            if (math.isfinite(mid_o) and math.isfinite(mid_c) and mid_o > 0 and mid_c > 0)
            else math.nan
        )
        iid = b["instrument_id"]
        rows.append(
            {
                "ts_event": _ns_to_dt(b["ts_close_ns"]),
                "ts_open": _ns_to_dt(b["ts_open_ns"]),
                "symbol": symbol,
                "contract_id": cmap.get(iid, f"id{iid}"),
                "con_id": int(iid),
                "session": "RTH" if cfg.session.rth_only else "ALL",
                "open": b["o"],
                "high": b["h"],
                "low": b["l"],
                "close": b["c"],
                "vwap": (b["vwap_num"] / volume) if volume > 0 else math.nan,
                "volume": volume,
                "dollar_volume": b["dollar_volume"],
                "mid_open": mid_o,
                "mid_close": mid_c,
                "n_trades": int(b["n_trades"]),
                "n_events": int(b["n_events"]),
                "bar_complete": bool(b["bar_complete"]),
                # --- features ---
                "ofi_top": b["ofi"],
                "qimb_top_close": b["qimb_top_close"],
                "qimb_top_twa": _twa(b["qimb_top_acc"], b["twa_dt"], b["qimb_top_close"]),
                "signed_vol": b["signed_vol"],
                "trade_imbalance": (b["signed_vol"] / volume) if volume > 0 else math.nan,
                "micro_minus_mid_close": b["micro_md_close"],
                "micro_minus_mid_twa": _twa(b["micro_md_acc"], b["twa_dt"], b["micro_md_close"]),
                "spread_ticks_close": b["spread_close"],
                "spread_ticks_twa": _twa(b["spread_acc"], b["twa_dt"], b["spread_close"]),
                "depth_top_twa": _twa(b["depth_top_acc"], b["twa_dt"], math.nan),
                "rv_intrabar": math.sqrt(b["rv_acc"]),
                "duration_s": (b["ts_close_ns"] - b["ts_open_ns"]) / 1e9,
                "dmid": (mid_c - mid_o)
                if (math.isfinite(mid_o) and math.isfinite(mid_c))
                else math.nan,
                "ret_bar": ret_bar,
                "source": source,
                "microstructure_feature_version": cfg.microstructure_feature_version,
            }
        )

    if not rows:
        return _empty_frame(cfg)

    df = pl.DataFrame(rows).sort("ts_event")
    k = cfg.ofi.rolling_bars
    # Causal rolling history, computed WITHIN each contract segment so flow never
    # bleeds across a roll. lag1/rolling include only past-or-current bars.
    df = df.with_columns(
        pl.col("ofi_top").shift(1).over("contract_id").alias("ofi_top_lag1"),
        pl.col("ofi_top").rolling_sum(k, min_samples=1).over("contract_id").alias("ofi_top_roll3"),
        pl.col("signed_vol")
        .rolling_sum(k, min_samples=1)
        .over("contract_id")
        .alias("signed_vol_roll3"),
    )
    return df.select(_column_order(cfg))


def _column_order(cfg: MicrostructureConfig) -> list[str]:
    structural = [
        "ts_event",
        "ts_open",
        "symbol",
        "contract_id",
        "con_id",
        "session",
        "open",
        "high",
        "low",
        "close",
        "vwap",
        "volume",
        "dollar_volume",
        "mid_open",
        "mid_close",
        "n_trades",
        "n_events",
        "bar_complete",
    ]
    return [*structural, *feature_names(cfg), "source", "microstructure_feature_version"]


def _empty_frame(cfg: MicrostructureConfig) -> pl.DataFrame:
    schema: dict[str, pl.DataType | type[pl.DataType]] = {
        "ts_event": pl.Datetime("us", "UTC"),
        "ts_open": pl.Datetime("us", "UTC"),
        "symbol": pl.Utf8,
        "contract_id": pl.Utf8,
        "con_id": pl.Int64,
        "session": pl.Utf8,
        "open": pl.Float64,
        "high": pl.Float64,
        "low": pl.Float64,
        "close": pl.Float64,
        "vwap": pl.Float64,
        "volume": pl.Float64,
        "dollar_volume": pl.Float64,
        "mid_open": pl.Float64,
        "mid_close": pl.Float64,
        "n_trades": pl.Int64,
        "n_events": pl.Int64,
        "bar_complete": pl.Boolean,
    }
    for f in feature_names(cfg):
        schema[f] = pl.Float64
    schema["source"] = pl.Utf8
    schema["microstructure_feature_version"] = pl.Utf8
    return pl.DataFrame(schema=schema)


def from_records(records: Iterable, instrument: Instrument) -> Iterator[BookEvent]:
    """Adapt Databento ``MBP1Msg`` records into :class:`BookEvent` (a generator,
    so the source stream is never materialised).

    Databento prices are 1e-9 fixed-point ints; an absent side uses a large
    sentinel which we map to NaN price / 0 size. Trade side 'B' (bid) = buy
    aggressor (+1), 'A' (ask) = sell aggressor (-1), else 0.
    """
    scale = 1e-9
    undef = 9223372036854775807  # INT64_MAX sentinel for an undefined price

    def _px(v: int) -> float:
        return math.nan if v == undef else v * scale

    for r in records:
        lv = r.levels[0]  # MBP-1: a single (top-of-book) level
        action = str(getattr(r.action, "value", r.action))
        is_trade = action == "T"
        side = str(getattr(r.side, "value", r.side))
        sign = 1 if side == "B" else (-1 if side == "A" else 0)
        yield BookEvent(
            ts_ns=int(r.ts_event),
            instrument_id=int(r.instrument_id),
            is_trade=is_trade,
            trade_price=_px(r.price) if is_trade else math.nan,
            trade_size=float(r.size) if is_trade else 0.0,
            trade_sign=sign if is_trade else 0,
            bid_px=_px(lv.bid_px),
            bid_sz=float(lv.bid_sz),
            ask_px=_px(lv.ask_px),
            ask_sz=float(lv.ask_sz),
        )
