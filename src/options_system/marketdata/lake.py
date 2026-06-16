"""Long-format Parquet lake for daily market-state series (System B).

One physical table at ``data/market_daily/``, partitioned by ``series=<id>`` (so a new
series adds a partition, never a schema migration), **idempotent on
``(series_id, obs_date)``**. Long format keeps adding series friction-free.

Point-in-time stamp: each value's ``observed_at`` is set to the **end of its observation
day in UTC**. A daily EOD value is only conservatively "knowable" once the day is over, so
this guarantees a feature built for label time ``t`` can only consume series values from
strictly-earlier days (lag-by-one). That is deliberately the leak-safe choice — it cannot
look ahead — and matches the project's refusal to fool itself with same-bar information.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, date, datetime, time
from glob import glob as _glob
from pathlib import Path
from uuid import uuid4

import polars as pl

from config.settings import Settings

_TS = pl.Datetime("us", "UTC")
_SCHEMA: dict[str, pl.DataType] = {
    "series_id": pl.Utf8(),
    "obs_date": pl.Date(),
    "value": pl.Float64(),
    "observed_at": _TS,  # END of obs_date in UTC — the conservative point-in-time clock
    "ingested_at": _TS,
}


def pit_observed_at(obs_date: date) -> datetime:
    """The conservative 'knowable at' instant for a daily value: end of its day (UTC)."""
    return datetime.combine(obs_date, time(23, 59, 59), tzinfo=UTC)


def build_rows(
    series_id: str, observations: pl.DataFrame, *, ingested_at: datetime
) -> pl.DataFrame:
    """``[obs_date, value]`` for one series -> the canonical long-format rows."""
    if observations.is_empty():
        return pl.DataFrame(schema=_SCHEMA)
    return observations.select(
        pl.lit(series_id).alias("series_id"),
        pl.col("obs_date"),
        pl.col("value").cast(pl.Float64),
        pl.col("obs_date").map_elements(pit_observed_at, return_dtype=_TS).alias("observed_at"),
        pl.lit(ingested_at).cast(_TS).alias("ingested_at"),
    )


class MarketDailyLake:
    """Idempotent long-format store for daily market-state series."""

    def __init__(self, root: Path | None = None, *, dataset: str = "market_daily") -> None:
        self.root = Path(root) if root is not None else Settings().data_dir
        self.dataset = dataset

    def _series_dir(self, series_id: str) -> Path:
        return self.root / self.dataset / f"series={series_id}"

    def _existing_dates(self, series_id: str) -> set[date]:
        d = self._series_dir(series_id)
        if not d.exists() or not any(d.glob("*.parquet")):
            return set()
        return set(pl.read_parquet(d / "*.parquet", columns=["obs_date"])["obs_date"].to_list())

    def write(self, rows: pl.DataFrame) -> int:
        """Append ``rows`` (canonical schema); idempotent on ``(series_id, obs_date)``.

        Returns the number of rows actually written (0 on a pure re-run).
        """
        if rows.is_empty():
            return 0
        rows = rows.select(list(_SCHEMA))
        written = 0
        for (series_id,), group in rows.group_by(["series_id"], maintain_order=True):
            seen = self._existing_dates(str(series_id))
            fresh = group.filter(~pl.col("obs_date").is_in(list(seen))) if seen else group
            if fresh.is_empty():
                continue
            part_dir = self._series_dir(str(series_id))
            part_dir.mkdir(parents=True, exist_ok=True)
            fresh.write_parquet(part_dir / f"part-{uuid4().hex}.parquet", compression="zstd")
            written += fresh.height
        return written

    def read(self, series_ids: Sequence[str] | None = None) -> pl.DataFrame:
        """Read the lake (optionally a subset of series); latest-ingest wins per key."""
        base = self.root / self.dataset
        if series_ids is None:
            glob = str(base / "series=*" / "*.parquet")
            files = _glob(glob)
        else:
            files = [
                f for sid in series_ids for f in _glob(str(base / f"series={sid}" / "*.parquet"))
            ]
        if not files:
            return pl.DataFrame(schema=_SCHEMA)
        df = pl.read_parquet(files)
        return (
            df.sort("ingested_at")
            .group_by(["series_id", "obs_date"], maintain_order=True)
            .last()
            .sort(["series_id", "obs_date"])
        )
