"""Tests for the DuckDB store (data/store.py), incl. the leak-free asof_join."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl

from options_system.data.lake import Lake
from options_system.data.store import DuckStore


def _bars(symbol="MES", contract="MESM6", n=6, start_min=30):
    base = datetime(2026, 6, 5, 14, start_min, tzinfo=UTC)
    rows = []
    for i in range(n):
        ts = base + timedelta(minutes=i)
        rows.append(
            {
                "ts_event": ts,
                "ts_ingest": ts + timedelta(milliseconds=100),
                "symbol": symbol,
                "contract_id": contract,
                "con_id": 1,
                "open": 5000.0 + i,
                "high": 5001.0 + i,
                "low": 4999.0 + i,
                "close": 5000.5 + i,
                "volume": 10.0,
                "wap": 5000.5 + i,
                "n_trades": 3,
                "session": "RTH",
                "source": "test",
            }
        )
    return pl.DataFrame(rows).with_columns(
        pl.col("ts_event").dt.cast_time_unit("us"),
        pl.col("ts_ingest").dt.cast_time_unit("us"),
    )


def test_get_bars_returns_window_ordered(tmp_path):
    lake = Lake(root=tmp_path)
    lake.write("bars_1m", _bars(n=6))
    store = DuckStore(lake)

    out = store.get_bars(
        "MES",
        datetime(2026, 6, 5, 14, 31, tzinfo=UTC),
        datetime(2026, 6, 5, 14, 33, tzinfo=UTC),
        freq="1m",
    )
    assert out.height == 3  # 14:31, 14:32, 14:33
    assert out["ts_event"].is_sorted()
    assert out["ts_event"].min() >= datetime(2026, 6, 5, 14, 31, tzinfo=UTC)
    assert out["ts_event"].max() <= datetime(2026, 6, 5, 14, 33, tzinfo=UTC)


def test_get_bars_empty_when_no_data(tmp_path):
    store = DuckStore(Lake(root=tmp_path))
    out = store.get_bars(
        "MES",
        datetime(2026, 6, 5, tzinfo=UTC),
        datetime(2026, 6, 6, tzinfo=UTC),
    )
    assert out.is_empty()


def test_asof_join_is_leak_free(tmp_path):
    store = DuckStore(Lake(root=tmp_path))
    # bars at :29 (before any aux), :30, :31, :32, :33
    bar_ts = [datetime(2026, 6, 5, 14, m, tzinfo=UTC) for m in (29, 30, 31, 32, 33)]
    left = pl.DataFrame({"ts_event": bar_ts, "bar_id": list(range(5))}).with_columns(
        pl.col("ts_event").dt.cast_time_unit("us")
    )
    aux_ts = [
        datetime(2026, 6, 5, 14, 29, 30, tzinfo=UTC),
        datetime(2026, 6, 5, 14, 31, 30, tzinfo=UTC),
        datetime(2026, 6, 5, 14, 33, 30, tzinfo=UTC),
    ]
    right = pl.DataFrame(
        {"ts_event": aux_ts, "aux_ts": aux_ts, "aux_value": [1.0, 2.0, 3.0]}
    ).with_columns(
        pl.col("ts_event").dt.cast_time_unit("us"),
        pl.col("aux_ts").dt.cast_time_unit("us"),
    )

    joined = store.asof_join(left, right, on="ts_event", right_select=["aux_ts", "aux_value"])
    joined = joined.sort("ts_event")

    # the :29 bar precedes all aux -> no match (left row kept, aux null)
    assert joined.filter(pl.col("bar_id") == 0)["aux_value"][0] is None
    # every matched aux timestamp is at or before its bar -> no future leakage
    matched = joined.filter(pl.col("aux_ts").is_not_null())
    assert (matched["aux_ts"] <= matched["ts_event"]).all()
    # spot-check the expected latest-known values
    by_id = {r["bar_id"]: r["aux_value"] for r in joined.iter_rows(named=True)}
    assert by_id[1] == 1.0  # 14:30 -> 14:29:30
    assert by_id[2] == 1.0  # 14:31 -> 14:29:30 (14:31:30 is in the future)
    assert by_id[3] == 2.0  # 14:32 -> 14:31:30
    assert by_id[4] == 2.0  # 14:33 -> 14:31:30


def test_get_bars_continuous(tmp_path):
    from datetime import date

    from options_system.data.continuous import detect_rolls, persist_rolls

    lake = Lake(root=tmp_path)
    # two contracts of the same symbol
    a = _bars(contract="MESH6", n=3, start_min=30)
    b = _bars(contract="MESM6", n=3, start_min=33).with_columns(
        (pl.col("close") + 100).alias("close"), (pl.col("open") + 100).alias("open")
    )
    lake.write("bars_1m", a)
    lake.write("bars_1m", b)
    daily = pl.DataFrame(
        {
            "date": [date(2026, 6, 5), date(2026, 6, 5)],
            "contract_id": ["MESH6", "MESM6"],
            "expiry": [date(2026, 6, 5), date(2026, 9, 18)],
            "volume": [10.0, 9999.0],  # MESM6 dominates -> roll
            "con_id": [1, 2],
        }
    )
    persist_rolls(lake, detect_rolls(daily, "MES", 5), ingest_ts=datetime(2026, 6, 5, tzinfo=UTC))

    store = DuckStore(lake)
    cont = store.get_bars(
        "MES",
        datetime(2026, 6, 5, 14, 0, tzinfo=UTC),
        datetime(2026, 6, 5, 15, 0, tzinfo=UTC),
        continuous=True,
    )
    assert "adj_factor" in cont.columns
    assert not cont.is_empty()
