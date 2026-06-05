"""Tests for the data-health gathering logic (observability/data_health.py)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl

from options_system.data.lake import Lake
from options_system.data.store import DuckStore
from options_system.observability.data_health import gather_health


def _bars(symbol="MES", contract="MESM6", n=5):
    base = datetime(2026, 6, 4, 14, 30, tzinfo=UTC)
    rows = []
    for i in range(n):
        ts = base + timedelta(minutes=i)
        rows.append(
            {
                "ts_event": ts,
                "ts_ingest": ts + timedelta(milliseconds=80),
                "symbol": symbol,
                "contract_id": contract,
                "con_id": 1,
                "open": 5000.0,
                "high": 5001.0,
                "low": 4999.0,
                "close": 5000.5,
                "volume": 10.0,
                "wap": 5000.5,
                "n_trades": 4,
                "session": "RTH",
                "source": "test",
            }
        )
    return pl.DataFrame(rows).with_columns(
        pl.col("ts_event").dt.cast_time_unit("us"),
        pl.col("ts_ingest").dt.cast_time_unit("us"),
    )


def test_gather_health_reports_metrics(tmp_path):
    lake = Lake(root=tmp_path)
    lake.write("bars_1m", _bars())
    store = DuckStore(lake)

    as_of = datetime(2026, 6, 4, 16, 0, tzinfo=UTC)
    health = gather_health(store, ["MES"], as_of=as_of, lookback_days=2)
    assert len(health) == 1
    h = health[0]
    assert h["rows"] == 5
    assert h["front_contract"] == "MESM6"
    assert h["last_bar_age_s"] is not None and h["last_bar_age_s"] > 0
    assert h["validation"]["ok"] is True


def test_gather_health_handles_empty(tmp_path):
    health = gather_health(DuckStore(Lake(root=tmp_path)), ["MES"], lookback_days=1)
    assert health[0]["rows"] == 0
    assert health[0]["front_contract"] is None
