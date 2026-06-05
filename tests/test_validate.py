"""Tests for data validation (data/validate.py) — every defect must be flagged."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl

from options_system.data.validate import validate_bars, validate_quotes


def _row(
    minute: int,
    *,
    contract="MESM6",
    ingest_ms=100,
    o=5000.0,
    h=5002.0,
    low=4999.0,
    c=5001.0,
    vol=10.0,
):
    ts = datetime(2026, 6, 4, 14, minute, tzinfo=UTC)
    return {
        "ts_event": ts,
        "ts_ingest": ts + timedelta(milliseconds=ingest_ms),
        "symbol": "MES",
        "contract_id": contract,
        "con_id": 1,
        "open": o,
        "high": h,
        "low": low,
        "close": c,
        "volume": vol,
        "wap": c,
        "n_trades": 5,
        "session": "RTH",
        "source": "test",
    }


def _frame(rows) -> pl.DataFrame:
    return pl.DataFrame(rows).with_columns(
        pl.col("ts_event").dt.cast_time_unit("us"),
        pl.col("ts_ingest").dt.cast_time_unit("us"),
    )


def test_clean_data_passes():
    report = validate_bars(_frame([_row(30), _row(31), _row(32)]))
    assert report.ok
    assert report.findings == []


def test_duplicates_flagged():
    report = validate_bars(_frame([_row(30), _row(30)]))
    assert not report.ok
    assert "duplicates" in report.checks_failed()
    assert "monotonic" in report.checks_failed()  # dup timestamp also breaks monotonicity


def test_inverted_ohlc_flagged():
    # high < low
    report = validate_bars(_frame([_row(30, h=4990.0, low=4999.0)]))
    assert "ohlc" in report.checks_failed()
    assert not report.ok


def test_negative_price_flagged():
    report = validate_bars(_frame([_row(30, o=-1.0)]))
    assert "ohlc" in report.checks_failed()


def test_future_ingest_flagged():
    # ts_ingest 1s BEFORE ts_event
    report = validate_bars(_frame([_row(30, ingest_ms=-1000)]))
    assert "ingest" in report.checks_failed()
    assert not report.ok


def test_rth_gap_flagged():
    # 14:30 then 14:40 -> 600s hole during RTH
    report = validate_bars(_frame([_row(30), _row(40)]))
    assert report.checks_failed() == {"gaps"}
    assert report.n_warnings == 1


def test_crossed_book_quote_flagged():
    ts = datetime(2026, 6, 4, 14, 30, tzinfo=UTC)
    q = pl.DataFrame(
        [
            {
                "ts_event": ts,
                "ts_ingest": ts + timedelta(milliseconds=50),
                "symbol": "MES",
                "contract_id": "MESM6",
                "con_id": 1,
                "bid": 5001.0,  # bid > ask -> crossed
                "ask": 5000.0,
                "bid_size": 1.0,
                "ask_size": 1.0,
                "last": 5000.5,
                "last_size": 1.0,
                "session": "RTH",
                "source": "test",
            }
        ]
    ).with_columns(
        pl.col("ts_event").dt.cast_time_unit("us"),
        pl.col("ts_ingest").dt.cast_time_unit("us"),
    )
    assert "crossed_book" in validate_quotes(q).checks_failed()
