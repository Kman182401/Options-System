"""Tests for the recorder's pure logic (data/recorder.py).

The IBKR-connected paths need a live paper login and are not tested here; the
aggregation, session tagging, and buffer->lake flush are.
"""

from __future__ import annotations

from datetime import UTC, datetime

import polars as pl

from options_system.data.lake import Lake
from options_system.data.recorder import MinuteAggregator, Recorder, session_for


def _dt(minute: int, second: int = 0) -> datetime:
    # A fixed PAST weekday during RTH (Thu 2026-06-04, 10:30 ET) so that the
    # recorder's ts_ingest = now() is always >= these event times.
    return datetime(2026, 6, 4, 14, minute, second, tzinfo=UTC)


def test_session_for_rth_eth_boundaries():
    assert session_for(datetime(2026, 6, 5, 14, 30, tzinfo=UTC)) == "RTH"  # 10:30 ET
    assert session_for(datetime(2026, 6, 5, 13, 30, tzinfo=UTC)) == "RTH"  # 09:30 ET open
    assert session_for(datetime(2026, 6, 5, 20, 0, tzinfo=UTC)) == "ETH"  # 16:00 ET close (excl)
    assert session_for(datetime(2026, 6, 5, 2, 0, tzinfo=UTC)) == "ETH"  # overnight
    assert session_for(datetime(2026, 6, 6, 14, 30, tzinfo=UTC)) == "ETH"  # Saturday


def test_minute_aggregator_rolls_up_5s_into_1m():
    agg = MinuteAggregator()
    assert agg.add("MESM6", _dt(30, 0), 5000, 5001, 4999, 5000.5, 10, 5000.2, 4) is None
    assert agg.add("MESM6", _dt(30, 5), 5000.5, 5003, 5000, 5002, 20, 5001.5, 6) is None
    # first bar of the next minute emits the completed 14:30 bar
    emitted = agg.add("MESM6", _dt(31, 0), 5002, 5004, 5001, 5003, 5, 5002.5, 3)
    assert emitted is not None
    assert emitted["ts_event"] == _dt(30, 0)
    assert emitted["open"] == 5000
    assert emitted["high"] == 5003  # max(5001, 5003)
    assert emitted["low"] == 4999  # min(4999, 5000)
    assert emitted["close"] == 5002  # last close of the minute
    assert emitted["volume"] == 30
    assert emitted["n_trades"] == 10
    assert abs(emitted["wap"] - (5000.2 * 10 + 5001.5 * 20) / 30) < 1e-9

    # the in-progress 14:31 minute flushes on shutdown
    left = agg.flush()
    assert len(left) == 1 and left[0]["ts_event"] == _dt(31, 0)


def test_recorder_buffer_flush_to_lake(tmp_path):
    rec = Recorder(lake=Lake(root=tmp_path))
    rec._buffer_bar5s("MES", "MESM6", 1, _dt(30, 0), 5000, 5001, 4999, 5000.5, 10, 5000.2, 4)
    rec._buffer_bar5s("MES", "MESM6", 1, _dt(30, 5), 5000.5, 5003, 5000, 5002, 20, 5001.5, 6)
    # crossing into 14:31 emits the 14:30 one-minute bar into the buffer
    rec._buffer_bar5s("MES", "MESM6", 1, _dt(31, 0), 5002, 5004, 5001, 5003, 5, 5002.5, 3)
    rec._buffer_quote("MES", "MESM6", 1, _dt(30, 1), 5000.0, 5000.5, 3.0, 4.0, 5000.25, 1.0)

    written = rec.flush()
    assert written > 0

    bars5 = pl.read_parquet(rec.lake.partition_glob("bars_5s"))
    bars1 = pl.read_parquet(rec.lake.partition_glob("bars_1m"))
    quotes = pl.read_parquet(rec.lake.partition_glob("quotes_l1"))

    assert bars5.height == 3
    assert bars1.height == 1  # only the completed 14:30 minute
    assert quotes.height == 1
    # dual timestamps present, UTC, and ingest >= event
    for frame in (bars5, bars1, quotes):
        assert frame["ts_event"].dtype.time_zone == "UTC"  # type: ignore[union-attr]
        assert (frame["ts_ingest"] >= frame["ts_event"]).all()
        assert frame["session"].unique().to_list() == ["RTH"]


def test_recorder_module_imports():
    import options_system.data.recorder as mod

    assert hasattr(mod, "Recorder")
    assert hasattr(mod, "main")
