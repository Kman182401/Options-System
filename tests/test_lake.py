"""Tests for the Parquet lake writer (data/lake.py)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import duckdb
import polars as pl

from options_system.data.lake import SCHEMA_VERSION, Lake, schema


def _synthetic_bars(n: int = 5, symbol: str = "MES") -> pl.DataFrame:
    base = datetime(2026, 6, 5, 14, 30, tzinfo=UTC)
    rows = []
    for i in range(n):
        ts = base + timedelta(minutes=i)
        rows.append(
            {
                "ts_event": ts,
                "ts_ingest": ts + timedelta(milliseconds=120),  # received slightly later
                "symbol": symbol,
                "contract_id": f"{symbol}M6",
                "con_id": 700000000 + i,
                "open": 5000.0 + i,
                "high": 5002.0 + i,
                "low": 4999.0 + i,
                "close": 5001.0 + i,
                "volume": 100.0 + i,
                "wap": 5000.5 + i,
                "n_trades": 10 + i,
                "session": "RTH",
                "source": "test",
            }
        )
    return pl.DataFrame(rows).with_columns(
        pl.col("ts_event").dt.cast_time_unit("us"),
        pl.col("ts_ingest").dt.cast_time_unit("us"),
    )


def test_write_then_read_back_via_duckdb(tmp_path):
    lake = Lake(root=tmp_path)
    df = _synthetic_bars()
    written = lake.write("bars_1m", df)
    assert written == df.height

    glob = lake.partition_glob("bars_1m")
    rel = duckdb.sql(f"SELECT * FROM read_parquet('{glob}') ORDER BY ts_event")
    out = rel.pl()

    # schema: every canonical column present, schema_version stamped
    assert set(schema("bars_1m")).issubset(set(out.columns))
    assert out["schema_version"].unique().to_list() == [SCHEMA_VERSION]
    assert out.height == df.height


def test_both_timestamps_present_and_utc(tmp_path):
    lake = Lake(root=tmp_path)
    lake.write("bars_1m", _synthetic_bars())
    out = pl.read_parquet(lake.partition_glob("bars_1m"))

    for col in ("ts_event", "ts_ingest"):
        assert col in out.columns
        assert out[col].null_count() == 0
        assert out[col].dtype.time_zone == "UTC"  # type: ignore[union-attr]
    # ts_ingest must be at/after ts_event everywhere (no future-leak by construction)
    assert (out["ts_ingest"] >= out["ts_event"]).all()


def test_writer_is_idempotent(tmp_path):
    lake = Lake(root=tmp_path)
    df = _synthetic_bars()
    assert lake.write("bars_1m", df) == df.height
    # re-writing the identical rows adds nothing (same-session cache)
    assert lake.write("bars_1m", df) == 0
    # ...and also across a fresh Lake (keys reloaded from disk)
    assert Lake(root=tmp_path).write("bars_1m", df) == 0

    n_rows = duckdb.sql(
        f"SELECT count(*) FROM read_parquet('{lake.partition_glob('bars_1m')}')"
    ).fetchone()[0]
    assert n_rows == df.height  # no duplicates on disk-read


def test_partial_overlap_only_writes_new_rows(tmp_path):
    lake = Lake(root=tmp_path)
    lake.write("bars_1m", _synthetic_bars(n=3))
    # 5 rows where the first 3 overlap -> only 2 new
    assert lake.write("bars_1m", _synthetic_bars(n=5)) == 2
