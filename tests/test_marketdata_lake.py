"""Daily market-state lake: PIT stamp, idempotency, latest-ingest-wins — offline."""

from __future__ import annotations

from datetime import UTC, date, datetime

import polars as pl

from options_system.marketdata.lake import MarketDailyLake, build_rows, pit_observed_at


def _obs(pairs: dict[date, float]) -> pl.DataFrame:
    return pl.DataFrame({"obs_date": list(pairs), "value": list(pairs.values())}).sort("obs_date")


def test_build_rows_schema_and_pit_stamp():
    rows = build_rows(
        "VIXCLS", _obs({date(2024, 1, 2): 10.0}), ingested_at=datetime(2026, 1, 1, tzinfo=UTC)
    )
    assert rows.columns == ["series_id", "obs_date", "value", "observed_at", "ingested_at"]
    assert rows["observed_at"][0] == pit_observed_at(date(2024, 1, 2))
    assert rows["series_id"][0] == "VIXCLS"


def test_write_idempotent_on_series_date(tmp_path):
    lake = MarketDailyLake(root=tmp_path)
    rows = build_rows(
        "VIXCLS",
        _obs({date(2024, 1, 2): 10.0, date(2024, 1, 3): 11.0}),
        ingested_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    assert lake.write(rows) == 2
    assert lake.write(rows) == 0  # pure re-run writes nothing
    assert lake.read().height == 2


def test_partitions_by_series(tmp_path):
    lake = MarketDailyLake(root=tmp_path)
    lake.write(
        build_rows(
            "VIXCLS", _obs({date(2024, 1, 2): 10.0}), ingested_at=datetime(2026, 1, 1, tzinfo=UTC)
        )
    )
    lake.write(
        build_rows(
            "DGS10", _obs({date(2024, 1, 2): 4.5}), ingested_at=datetime(2026, 1, 1, tzinfo=UTC)
        )
    )
    assert (tmp_path / "market_daily" / "series=VIXCLS").exists()
    assert (tmp_path / "market_daily" / "series=DGS10").exists()
    assert lake.read(series_ids=["DGS10"]).height == 1


def test_incremental_append_only_writes_new_dates(tmp_path):
    lake = MarketDailyLake(root=tmp_path)
    lake.write(
        build_rows(
            "VIXCLS", _obs({date(2024, 1, 2): 10.0}), ingested_at=datetime(2026, 1, 1, tzinfo=UTC)
        )
    )
    # A later daily run re-sees Jan-2 (skipped) and adds Jan-3 (new). These series are
    # never revised, so an already-stored date is idempotently left as-is.
    written = lake.write(
        build_rows(
            "VIXCLS",
            _obs({date(2024, 1, 2): 12.5, date(2024, 1, 3): 11.0}),
            ingested_at=datetime(2026, 2, 1, tzinfo=UTC),
        )
    )
    assert written == 1  # only Jan-3 is new
    df = lake.read().sort("obs_date")
    assert df["obs_date"].to_list() == [date(2024, 1, 2), date(2024, 1, 3)]
    assert df["value"][0] == 10.0  # Jan-2 unchanged (idempotent skip, not overwritten)


def test_empty_read(tmp_path):
    lake = MarketDailyLake(root=tmp_path)
    assert lake.read().height == 0  # typed empty frame, not an error
