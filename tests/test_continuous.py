"""Tests for continuous-contract roll handling (data/continuous.py)."""

from __future__ import annotations

from datetime import UTC, date, datetime

import polars as pl

from options_system.data.continuous import (
    Contract,
    build_continuous,
    detect_rolls,
    persist_rolls,
    pick_front_month,
)
from options_system.data.lake import Lake

A = "MESH6"  # expires 2026-03-20
B = "MESM6"  # expires 2026-06-19
EXP_A = date(2026, 3, 20)
EXP_B = date(2026, 6, 19)


def _daily() -> pl.DataFrame:
    # A leads until 2026-03-12, when B's volume overtakes it (crossover).
    rows = [
        ("2026-03-10", A, EXP_A, 1000.0, 1),
        ("2026-03-10", B, EXP_B, 200.0, 2),
        ("2026-03-11", A, EXP_A, 900.0, 1),
        ("2026-03-11", B, EXP_B, 400.0, 2),
        ("2026-03-12", A, EXP_A, 500.0, 1),
        ("2026-03-12", B, EXP_B, 800.0, 2),  # B > A -> roll
        ("2026-03-13", A, EXP_A, 100.0, 1),
        ("2026-03-13", B, EXP_B, 1000.0, 2),
    ]
    return pl.DataFrame(
        {
            "date": [date.fromisoformat(r[0]) for r in rows],
            "contract_id": [r[1] for r in rows],
            "expiry": [r[2] for r in rows],
            "volume": [r[3] for r in rows],
            "con_id": [r[4] for r in rows],
        }
    )


def _bar(cid: str, dt: datetime, close: float) -> dict:
    return {
        "ts_event": dt,
        "ts_ingest": dt,
        "symbol": "MES",
        "contract_id": cid,
        "con_id": 1 if cid == A else 2,
        "open": close - 1,
        "high": close + 1,
        "low": close - 2,
        "close": close,
        "volume": 50.0,
        "wap": close,
        "n_trades": 5,
        "session": "RTH",
        "source": "test",
    }


def _bars() -> pl.DataFrame:
    rows = [
        _bar(A, datetime(2026, 3, 10, 15, tzinfo=UTC), 5000.0),
        _bar(A, datetime(2026, 3, 11, 15, tzinfo=UTC), 5010.0),  # last A before seam
        _bar(B, datetime(2026, 3, 12, 15, tzinfo=UTC), 5110.0),  # first B at/after seam
        _bar(B, datetime(2026, 3, 13, 15, tzinfo=UTC), 5120.0),
    ]
    return pl.DataFrame(rows).with_columns(
        pl.col("ts_event").dt.cast_time_unit("us"),
        pl.col("ts_ingest").dt.cast_time_unit("us"),
    )


def test_pick_front_month_volume_crossover():
    contracts = [Contract(A, EXP_A, 1), Contract(B, EXP_B, 2)]
    # before crossover (A has more volume, far from expiry) -> A
    front = pick_front_month(contracts, date(2026, 3, 10), 5, {A: 1000, B: 200})
    assert front.contract_id == A
    # B overtakes on volume -> B
    front = pick_front_month(contracts, date(2026, 3, 12), 5, {A: 500, B: 800})
    assert front.contract_id == B


def test_pick_front_month_calendar_fallback():
    contracts = [Contract(A, EXP_A, 1), Contract(B, EXP_B, 2)]
    # within 5 days of A's expiry, no volume info -> roll to B
    front = pick_front_month(contracts, date(2026, 3, 17), 5, None)
    assert front.contract_id == B


def test_detect_rolls_fires_on_crossover():
    rolls = detect_rolls(_daily(), symbol="MES", calendar_days=5)
    assert rolls.height == 1
    row = rolls.row(0, named=True)
    assert row["from_contract_id"] == A
    assert row["to_contract_id"] == B
    assert row["rule"] == "volume_oi"
    assert row["ts_event"] == datetime(2026, 3, 12, tzinfo=UTC)


def test_build_continuous_is_continuous_across_seam_and_preserves_raw():
    bars = _bars()
    snapshot = bars.clone()
    rolls = detect_rolls(_daily(), symbol="MES", calendar_days=5)

    cont = build_continuous(bars, rolls, adjustment="ratio")

    # raw bars untouched
    assert bars.equals(snapshot)

    cont = cont.sort("ts_event")
    last_a = cont.filter(pl.col("contract_id") == A).sort("ts_event")["close"][-1]
    first_b = cont.filter(pl.col("contract_id") == B).sort("ts_event")["close"][0]
    # adjusted A close at the seam meets B's close -> no artificial jump
    assert abs(last_a - first_b) < 1e-6
    # the unadjusted gap was large (5010 vs 5110); adjustment removed it
    assert abs(snapshot.filter(pl.col("contract_id") == A)["close"][-1] - first_b) > 50


def test_persist_rolls_writes_event(tmp_path):
    lake = Lake(root=tmp_path)
    rolls = detect_rolls(_daily(), symbol="MES", calendar_days=5)
    n = persist_rolls(lake, rolls, ingest_ts=datetime(2026, 3, 12, 1, tzinfo=UTC))
    assert n == 1
    back = pl.read_parquet(lake.partition_glob("roll_events"))
    assert back.height == 1
    assert back["to_contract_id"][0] == B
    assert back["ts_ingest"].dtype.time_zone == "UTC"  # type: ignore[union-attr]
