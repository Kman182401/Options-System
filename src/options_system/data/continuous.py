"""Continuous-contract / roll handling for CME quarterly futures (MES, MNQ).

Three jobs, all kept deliberately separate from the raw data:

1. **Front-month selection** (:func:`pick_front_month`) — given the listed
   quarterly contracts (H/M/U/Z = Mar/Jun/Sep/Dec) and, optionally, current
   volume, decide which contract is "the front month" right now. Default rule:
   the nearest non-expired contract, rolling to the next when its volume
   overtakes the current one (volume crossover) or when we're within
   ``roll_calendar_days`` of expiry (calendar fallback) — whichever comes first.

2. **Roll detection** (:func:`detect_rolls`) — replay a daily volume table and
   emit a roll event each time the front month changes.

3. **Continuous series** (:func:`build_continuous`) — stitch the active segments
   of each contract into one price series and **back-adjust** older segments so
   the series is continuous across each roll seam. Default convention is
   **ratio** adjustment (multiply); **panama** (difference) is also supported.
   See ``docs/DECISIONS.md`` for the rationale.

The raw per-``contract_id`` bars are NEVER mutated — back-adjustment is applied
to a copy. We keep raw and continuous separate so the adjustment convention can
change later without re-recording anything.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, time

import polars as pl

# CME quarterly month codes.
MONTH_CODE = {3: "H", 6: "M", 9: "U", 12: "Z"}
CODE_MONTH = {v: k for k, v in MONTH_CODE.items()}

_PRICE_COLS = ("open", "high", "low", "close", "wap")


@dataclass(frozen=True)
class Contract:
    """A specific futures expiry."""

    contract_id: str
    expiry: date
    con_id: int | None = None


def pick_front_month(
    contracts: list[Contract],
    as_of: date,
    calendar_days: int,
    volume: dict[str, float] | None = None,
) -> Contract:
    """Choose the front-month contract as of ``as_of``.

    Rolls to the next contract when its ``volume`` exceeds the current one, or
    when the current contract is within ``calendar_days`` of expiry.
    """
    if not contracts:
        raise ValueError("no contracts provided")
    active = sorted((c for c in contracts if c.expiry >= as_of), key=lambda c: c.expiry)
    if not active:  # everything expired -> latest available
        return sorted(contracts, key=lambda c: c.expiry)[-1]

    current = active[0]
    nxt = active[1] if len(active) > 1 else None
    if (
        nxt is not None
        and volume is not None
        and volume.get(nxt.contract_id, 0.0) > volume.get(current.contract_id, 0.0)
    ):
        return nxt
    if nxt is not None and (current.expiry - as_of).days <= calendar_days:
        return nxt
    return current


def _midnight_utc(d: date) -> datetime:
    return datetime.combine(d, time(0), tzinfo=UTC)


def detect_rolls(daily: pl.DataFrame, symbol: str, calendar_days: int) -> pl.DataFrame:
    """Detect roll events from a daily table.

    ``daily`` columns: ``date`` (date), ``contract_id``, ``expiry`` (date),
    ``volume`` (float); ``con_id`` (int) optional. Returns a DataFrame of roll
    events (one row per roll) with ``ts_event`` at the roll date's UTC midnight.
    Empty if no roll occurs.
    """
    rows = daily.sort("date")
    meta: dict[str, tuple[date, int | None]] = {}
    for r in rows.iter_rows(named=True):
        meta[r["contract_id"]] = (r["expiry"], r.get("con_id"))
    contracts = [Contract(cid, exp, con) for cid, (exp, con) in meta.items()]

    current: Contract | None = None
    events: list[dict] = []
    for d in rows["date"].unique(maintain_order=True).to_list():
        day = rows.filter(pl.col("date") == d)
        vol = {r["contract_id"]: float(r["volume"]) for r in day.iter_rows(named=True)}
        front = pick_front_month(contracts, d, calendar_days, vol)
        if current is None:
            current = front
        elif front.contract_id != current.contract_id:
            rule = (
                "volume_oi"
                if vol.get(front.contract_id, 0.0) > vol.get(current.contract_id, 0.0)
                else "calendar"
            )
            events.append(
                {
                    "ts_event": _midnight_utc(d),
                    "symbol": symbol,
                    "from_contract_id": current.contract_id,
                    "to_contract_id": front.contract_id,
                    "from_con_id": meta[current.contract_id][1],
                    "to_con_id": meta[front.contract_id][1],
                    "rule": rule,
                }
            )
            current = front

    if not events:
        return pl.DataFrame(
            schema={
                "ts_event": pl.Datetime("us", "UTC"),
                "symbol": pl.Utf8,
                "from_contract_id": pl.Utf8,
                "to_contract_id": pl.Utf8,
                "from_con_id": pl.Int64,
                "to_con_id": pl.Int64,
                "rule": pl.Utf8,
            }
        )
    return pl.DataFrame(events).with_columns(pl.col("ts_event").dt.cast_time_unit("us"))


def _seam_price(bars: pl.DataFrame, contract_id: str, t: datetime, *, before: bool) -> float | None:
    sub = bars.filter(pl.col("contract_id") == contract_id)
    sub = sub.filter(pl.col("ts_event") < t) if before else sub.filter(pl.col("ts_event") >= t)
    if sub.is_empty():
        return None
    sub = sub.sort("ts_event")
    return float(sub["close"][-1] if before else sub["close"][0])


def build_continuous(
    bars: pl.DataFrame,
    rolls: pl.DataFrame,
    adjustment: str = "ratio",
) -> pl.DataFrame:
    """Build a back-adjusted continuous series from raw bars + roll events.

    ``bars`` is raw per-contract bars (canonical bar schema). ``rolls`` is the
    output of :func:`detect_rolls`. The returned frame is the continuous series
    with an ``adj_factor`` column (the multiplicative factor, or additive offset
    for panama, applied to that segment). Raw ``bars`` is not modified.
    """
    if adjustment not in {"ratio", "panama"}:
        raise ValueError(f"adjustment={adjustment!r} must be 'ratio' or 'panama'")
    bars = bars.sort("ts_event")
    if rolls.is_empty():
        return bars.with_columns(pl.lit(1.0 if adjustment == "ratio" else 0.0).alias("adj_factor"))

    rolls = rolls.sort("ts_event")
    seg_contracts = [rolls["from_contract_id"][0], *rolls["to_contract_id"].to_list()]
    seg_starts: list[datetime | None] = [None, *rolls["ts_event"].to_list()]
    seg_ends: list[datetime | None] = [*rolls["ts_event"].to_list(), None]
    n = len(seg_contracts)

    factors = [1.0] * n  # multiplicative (ratio)
    offsets = [0.0] * n  # additive (panama)
    for i in range(n - 2, -1, -1):
        t = rolls["ts_event"][i]
        p_from = _seam_price(bars, seg_contracts[i], t, before=True)
        p_to = _seam_price(bars, seg_contracts[i + 1], t, before=False)
        if p_from is None or p_to is None:
            continue
        if adjustment == "ratio":
            factors[i] = factors[i + 1] * (p_to / p_from if p_from else 1.0)
        else:
            offsets[i] = offsets[i + 1] + (p_to - p_from)

    segments: list[pl.DataFrame] = []
    for i in range(n):
        seg = bars.filter(pl.col("contract_id") == seg_contracts[i])
        if seg_starts[i] is not None:
            seg = seg.filter(pl.col("ts_event") >= seg_starts[i])
        if seg_ends[i] is not None:
            seg = seg.filter(pl.col("ts_event") < seg_ends[i])
        if seg.is_empty():
            continue
        present = [c for c in _PRICE_COLS if c in seg.columns]
        if adjustment == "ratio":
            seg = seg.with_columns([(pl.col(c) * factors[i]).alias(c) for c in present])
            seg = seg.with_columns(pl.lit(factors[i]).alias("adj_factor"))
        else:
            seg = seg.with_columns([(pl.col(c) + offsets[i]).alias(c) for c in present])
            seg = seg.with_columns(pl.lit(offsets[i]).alias("adj_factor"))
        segments.append(seg)
    return pl.concat(segments).sort("ts_event")


def persist_rolls(lake, rolls: pl.DataFrame, ingest_ts: datetime, source: str = "derived") -> int:
    """Write roll events to the lake's ``roll_events`` dataset. Returns rows written."""
    if rolls.is_empty():
        return 0
    enriched = rolls.with_columns(
        pl.lit(ingest_ts).cast(pl.Datetime("us", "UTC")).alias("ts_ingest"),
        pl.lit(None, dtype=pl.Float64).alias("adj_factor"),
        pl.lit(None, dtype=pl.Utf8).alias("note"),
        pl.lit(source).alias("source"),
    )
    return lake.write("roll_events", enriched)
