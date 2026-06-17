"""Macro-event ingestion: timestamp construction, storage idempotency, key-gating.

These are offline (no FRED network calls): the pure helpers and the lake
writer/reader are exercised with synthetic data, and key-gating is verified by
running ``ingest`` with no API key.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, time
from zoneinfo import ZoneInfo

import polars as pl

from config.settings import Settings
from options_system.macro import ingest as mi

_ET = ZoneInfo("America/New_York")


def test_event_time_handles_dst():
    # 08:30 ET data release: summer (EDT, UTC-4) → 12:30 UTC; winter (EST, UTC-5) → 13:30 UTC.
    summer = mi._event_time_utc(date(2024, 7, 11), time(8, 30), _ET)
    winter = mi._event_time_utc(date(2024, 1, 11), time(8, 30), _ET)
    assert (summer.hour, summer.minute, summer.tzinfo) == (12, 30, UTC)
    assert (winter.hour, winter.minute) == (13, 30)
    # 14:00 ET FOMC statement: summer → 18:00 UTC; winter → 19:00 UTC.
    fomc_summer = mi._event_time_utc(date(2024, 7, 31), time(14, 0), _ET)
    fomc_winter = mi._event_time_utc(date(2024, 12, 18), time(14, 0), _ET)
    assert fomc_summer.hour == 18
    assert fomc_winter.hour == 19


def test_asof_value_is_backward():
    frame = pl.DataFrame(
        {
            "ref_period": [date(2020, 1, 1), date(2020, 2, 1), date(2020, 3, 1)],
            "value": [1.0, 2.0, 3.0],
        }
    )
    assert mi._asof_value(frame, date(2020, 2, 15)) == 2.0  # latest <= when
    assert mi._asof_value(frame, date(2020, 3, 1)) == 3.0  # inclusive
    assert mi._asof_value(frame, date(2019, 12, 1)) is None  # before all → none


def _synthetic_events(n: int = 3) -> pl.DataFrame:
    base = datetime(2024, 7, 11, 12, 30, tzinfo=UTC)
    ingest_ts = datetime(2026, 1, 1, tzinfo=UTC)
    rows = [
        {
            "event_time": base.replace(month=7 + i),
            "event_type": "cpi",
            "ref_period": date(2024, 6 + i, 1),
            "actual_pit": 313.0 + i,
            "prior": 312.0 + i,
            "surprise": None,
            "source": "FRED:CPIAUCSL",
            "fred_series_id": "CPIAUCSL",
            "ingest_ts": ingest_ts,
            "macro_version": "v1",
        }
        for i in range(n)
    ]
    return pl.DataFrame(rows, schema=mi._SCHEMA)


def test_write_read_roundtrip_and_idempotent(tmp_path, monkeypatch):
    monkeypatch.setenv("OPTIONS_DATA_DIR", str(tmp_path))
    frame = _synthetic_events(3)
    assert mi.write_macro_events(frame) == 3
    back = mi.read_macro_events()
    assert back.height == 3
    assert back["actual_pit"].to_list() == [313.0, 314.0, 315.0]
    assert back["surprise"].null_count() == 3  # surprise never fabricated
    # re-write the same rows → idempotent, zero new rows.
    assert mi.write_macro_events(frame) == 0


def test_ingest_no_ops_without_key(monkeypatch):
    monkeypatch.delenv("OPTIONS_FRED_API_KEY", raising=False)
    settings = Settings(fred_api_key=None)
    assert settings.fred_api_key is None
    # No key → no-op, no network is attempted (so the network gate is not reached).
    assert mi.ingest(settings=settings) == {}


def test_ingest_blocked_with_key_but_no_allow_network(monkeypatch):
    # With a key SET, a real FRED fetch is fail-closed: it requires --allow-network.
    import pytest

    from options_system.common.external_data_policy import ExternalAccessNotAuthorized

    settings = Settings(fred_api_key="dummy-key")
    with pytest.raises(ExternalAccessNotAuthorized):
        mi.ingest(settings=settings, allow_network=False)
