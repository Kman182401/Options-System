"""Date-partitioned Parquet lake for GKG bulk events (System A).

The FinBERT path's :class:`~options_system.sentiment.lake.SentimentLake` re-reads the
whole source partition on every write to dedup by ``content_hash``. That is correct and
cheap at Phase-18 scale (a few hundred writes), but the GKG bulk backfill writes one part
per 15-minute file — ~260k writes / ~190M rows over seven years — where a full-partition
re-read per write is O(n²) and would never finish.

This lake is built for that scale instead:

* **Date-partitioned** — ``<dataset>/date=YYYY-MM-DD/`` so reads touch only the days they
  need and no directory holds more than a day's ~96 files.
* **Deterministic part names** — one part file per GKG 15-minute file, named by that file's
  timestamp. Re-processing the same file **overwrites** its part (idempotent) without ever
  reading the rest of the lake. Cross-file idempotency is the backfill manifest's job
  (each file is processed once); within a file, events are deduped by ``content_hash``.

It reuses the typed frame builders (:func:`events_to_frame`, :func:`scored_to_frame`) and
the PIT-validated event schema, so a GKG row is stored exactly like any other news event —
only the partitioning and write strategy differ.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, date, datetime
from glob import glob as _glob
from pathlib import Path

import polars as pl

from config.settings import Settings

from .lake import events_to_frame, scored_to_frame
from .schema import RawNewsEvent, ScoredNewsEvent, dedupe_by_hash


def _part_stamp(file_ts: datetime) -> str:
    return file_ts.astimezone(UTC).strftime("%Y%m%d%H%M%S")


def _dedup_scored(scored: Sequence[ScoredNewsEvent]) -> list[ScoredNewsEvent]:
    """One row per ``content_hash`` within a single file (last wins, deterministic)."""
    best: dict[str, ScoredNewsEvent] = {}
    for s in scored:
        best[s.content_hash] = s
    return list(best.values())


class GkgLake:
    """Date-partitioned, deterministically-named Parquet store for GKG events."""

    def __init__(
        self,
        root: Path | None = None,
        *,
        raw_dataset: str = "sentiment_gkg_raw",
        scored_dataset: str = "sentiment_gkg_scores",
    ) -> None:
        self.root = Path(root) if root is not None else Settings().data_dir
        self.raw_dataset = raw_dataset
        self.scored_dataset = scored_dataset

    def _raw_part(self, day: date, stamp: str) -> Path:
        return self.root / self.raw_dataset / f"date={day.isoformat()}" / f"part-{stamp}.parquet"

    def _scored_part(self, day: date, stamp: str) -> Path:
        return self.root / self.scored_dataset / f"date={day.isoformat()}" / f"part-{stamp}.parquet"

    def write_file(
        self,
        file_ts: datetime,
        raw: Sequence[RawNewsEvent],
        scored: Sequence[ScoredNewsEvent],
    ) -> int:
        """Write one GKG file's events to its date partition (idempotent by overwrite).

        ``file_ts`` is the 15-minute slot the events came from; it fixes both the date
        partition and the deterministic part-file name, so re-processing the same file
        replaces its part rather than appending duplicates. Returns the raw row count.
        """
        day = file_ts.astimezone(UTC).date()
        stamp = _part_stamp(file_ts)
        raw_dedup = dedupe_by_hash(list(raw))
        if raw_dedup:
            p = self._raw_part(day, stamp)
            p.parent.mkdir(parents=True, exist_ok=True)
            events_to_frame(raw_dedup).write_parquet(p, compression="zstd")
        scored_dedup = _dedup_scored(scored)
        if scored_dedup:
            p = self._scored_part(day, stamp)
            p.parent.mkdir(parents=True, exist_ok=True)
            scored_to_frame(scored_dedup).write_parquet(p, compression="zstd")
        return len(raw_dedup)

    def _glob_parts(self, dataset: str, start: date | None, end: date | None) -> list[str]:
        base = self.root / dataset
        parts = sorted(_glob(str(base / "date=*" / "*.parquet")))
        if start is None and end is None:
            return parts
        kept = []
        for p in parts:
            try:
                d = date.fromisoformat(Path(p).parent.name.split("=", 1)[1])
            except (IndexError, ValueError):
                continue
            if (start is None or d >= start) and (end is None or d <= end):
                kept.append(p)
        return kept

    def read_scored(self, *, start: date | None = None, end: date | None = None) -> pl.DataFrame:
        """Read scored GKG events (optionally a date range). Empty -> typed empty frame."""
        parts = self._glob_parts(self.scored_dataset, start, end)
        if not parts:
            return scored_to_frame([])
        return pl.read_parquet(parts)

    def read_raw(self, *, start: date | None = None, end: date | None = None) -> pl.DataFrame:
        """Read raw GKG events (optionally a date range). Empty -> typed empty frame."""
        parts = self._glob_parts(self.raw_dataset, start, end)
        if not parts:
            return events_to_frame([])
        return pl.read_parquet(parts)
