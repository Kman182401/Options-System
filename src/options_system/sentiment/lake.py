"""Local Parquet lake for raw news events and their sentiment scores.

Two datasets under ``data/`` (gitignored), kept separate so re-scoring never rewrites
the raw text, mirroring how the price/microstructure lakes split bars from labels:

* ``sentiment_raw``    — :class:`~options_system.sentiment.schema.RawNewsEvent` rows,
  partitioned by ``source=``. **Idempotent on ``content_hash``**: re-ingesting the
  same item writes nothing new.
* ``sentiment_scores`` — :class:`~options_system.sentiment.schema.ScoredNewsEvent`
  rows. **Idempotent on ``(content_hash, model_name)``**: re-scoring with the same
  model writes nothing new; a different model adds a row.

The lake root defaults to ``Settings().data_dir`` but is injectable so tests write to
a tmp dir and never touch the real lake.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import polars as pl

from config.settings import Settings

from .schema import RawNewsEvent, ScoredNewsEvent, dedupe_by_hash

_RAW_SCHEMA = {
    "content_hash": pl.Utf8,
    "source": pl.Utf8,
    "source_id": pl.Utf8,
    "source_url": pl.Utf8,
    "title": pl.Utf8,
    "snippet_or_text": pl.Utf8,
    "published_at": pl.Datetime("us", "UTC"),
    "observed_at": pl.Datetime("us", "UTC"),
    "ingested_at": pl.Datetime("us", "UTC"),
    "query_topic": pl.Utf8,
    "language": pl.Utf8,
    "entities": pl.List(pl.Utf8),
    "sentiment_feature_version": pl.Utf8,
    "degraded": pl.Boolean,
    "error": pl.Utf8,
}

_SCORED_SCHEMA = {
    "content_hash": pl.Utf8,
    "source": pl.Utf8,
    "query_topic": pl.Utf8,
    "published_at": pl.Datetime("us", "UTC"),
    "observed_at": pl.Datetime("us", "UTC"),
    "sentiment_feature_version": pl.Utf8,
    "positive_score": pl.Float64,
    "negative_score": pl.Float64,
    "neutral_score": pl.Float64,
    "sentiment_score": pl.Float64,
    "model_name": pl.Utf8,
    "model_version_or_hash": pl.Utf8,
    "scored_at": pl.Datetime("us", "UTC"),
}


def _utc(dt: datetime) -> datetime:
    return dt.astimezone(UTC) if dt.tzinfo else dt.replace(tzinfo=UTC)


def _raw_row(ev: RawNewsEvent) -> dict:
    return {
        "content_hash": ev.content_hash,
        "source": ev.source,
        "source_id": ev.source_id,
        "source_url": ev.source_url,
        "title": ev.title,
        "snippet_or_text": ev.snippet_or_text,
        "published_at": _utc(ev.published_at),
        "observed_at": _utc(ev.observed_at),
        "ingested_at": _utc(ev.ingested_at),
        "query_topic": ev.query_topic,
        "language": ev.language,
        "entities": list(ev.entities),
        "sentiment_feature_version": ev.sentiment_feature_version,
        "degraded": ev.degraded,
        "error": ev.error,
    }


def events_to_frame(events: Iterable[RawNewsEvent]) -> pl.DataFrame:
    """Raw events -> a typed polars frame (empty-safe via the pinned schema)."""
    rows = [_raw_row(e) for e in events]
    return pl.DataFrame(rows, schema=_RAW_SCHEMA)


def _scored_row(ev: ScoredNewsEvent) -> dict:
    s = ev.score
    return {
        "content_hash": ev.content_hash,
        "source": ev.source,
        "query_topic": ev.query_topic,
        "published_at": _utc(ev.published_at),
        "observed_at": _utc(ev.observed_at),
        "sentiment_feature_version": ev.sentiment_feature_version,
        "positive_score": s.positive_score,
        "negative_score": s.negative_score,
        "neutral_score": s.neutral_score,
        "sentiment_score": s.sentiment_score,
        "model_name": s.model_name,
        "model_version_or_hash": s.model_version_or_hash,
        "scored_at": _utc(s.scored_at),
    }


def scored_to_frame(scored: Iterable[ScoredNewsEvent]) -> pl.DataFrame:
    rows = [_scored_row(e) for e in scored]
    return pl.DataFrame(rows, schema=_SCORED_SCHEMA)


class SentimentLake:
    """Idempotent Parquet store for raw news events and sentiment scores."""

    def __init__(
        self,
        root: Path | None = None,
        *,
        raw_dataset: str = "sentiment_raw",
        scored_dataset: str = "sentiment_scores",
    ) -> None:
        self.root = Path(root) if root is not None else Settings().data_dir
        self.raw_dataset = raw_dataset
        self.scored_dataset = scored_dataset

    # --- raw ---------------------------------------------------------------- #

    def _raw_dir(self, source: str) -> Path:
        return self.root / self.raw_dataset / f"source={source}"

    def _existing_hashes(self, source: str) -> set[str]:
        d = self._raw_dir(source)
        if not d.exists() or not any(d.glob("*.parquet")):
            return set()
        df = pl.read_parquet(str(d / "*.parquet"), columns=["content_hash"])
        return set(df["content_hash"].to_list())

    def write_raw(self, events: Sequence[RawNewsEvent]) -> int:
        """Append new raw events; idempotent on ``content_hash`` per source. Returns
        the number of rows actually written (0 on a pure re-run)."""
        if not events:
            return 0
        deduped = dedupe_by_hash(events)
        written = 0
        by_source: dict[str, list[RawNewsEvent]] = {}
        for ev in deduped:
            by_source.setdefault(ev.source, []).append(ev)
        for source, evs in by_source.items():
            seen = self._existing_hashes(source)
            fresh = [e for e in evs if e.content_hash not in seen]
            if not fresh:
                continue
            part_dir = self._raw_dir(source)
            part_dir.mkdir(parents=True, exist_ok=True)
            events_to_frame(fresh).write_parquet(
                part_dir / f"part-{uuid4().hex}.parquet", compression="zstd"
            )
            written += len(fresh)
        return written

    def read_raw(self, source: str | None = None) -> pl.DataFrame:
        """Read raw events (optionally one source); latest-ingest wins per hash."""
        base = self.root / self.raw_dataset
        glob = (
            base / f"source={source}" / "*.parquet"
            if source is not None
            else base / "source=*" / "*.parquet"
        )
        from glob import glob as _glob

        if not _glob(str(glob)):
            return pl.DataFrame(schema=_RAW_SCHEMA)
        df = pl.read_parquet(str(glob))
        return (
            df.sort("ingested_at")
            .group_by("content_hash", maintain_order=True)
            .last()
            .sort("published_at")
        )

    # --- scored ------------------------------------------------------------- #

    def _scored_glob(self) -> Path:
        return self.root / self.scored_dataset / "*.parquet"

    def _existing_scored_keys(self) -> set[tuple[str, str]]:
        from glob import glob as _glob

        g = self._scored_glob()
        if not _glob(str(g)):
            return set()
        df = pl.read_parquet(str(g), columns=["content_hash", "model_name"])
        return set(zip(df["content_hash"].to_list(), df["model_name"].to_list(), strict=True))

    def write_scored(self, scored: Sequence[ScoredNewsEvent]) -> int:
        """Append new scores; idempotent on ``(content_hash, model_name)``."""
        if not scored:
            return 0
        seen = self._existing_scored_keys()
        fresh = [s for s in scored if (s.content_hash, s.score.model_name) not in seen]
        # de-dup within this batch too
        uniq: dict[tuple[str, str], ScoredNewsEvent] = {}
        for s in fresh:
            uniq[(s.content_hash, s.score.model_name)] = s
        if not uniq:
            return 0
        out_dir = self.root / self.scored_dataset
        out_dir.mkdir(parents=True, exist_ok=True)
        scored_to_frame(list(uniq.values())).write_parquet(
            out_dir / f"part-{uuid4().hex}.parquet", compression="zstd"
        )
        return len(uniq)

    def read_scored(self) -> pl.DataFrame:
        from glob import glob as _glob

        g = self._scored_glob()
        if not _glob(str(g)):
            return pl.DataFrame(schema=_SCORED_SCHEMA)
        return pl.read_parquet(str(g))
