"""Date-partitioned GKG lake: idempotent overwrite, partitioning, dedup — offline."""

from __future__ import annotations

from datetime import UTC, date, datetime

from options_system.sentiment.gkg_lake import GkgLake
from options_system.sentiment.schema import (
    RawNewsEvent,
    ScoredNewsEvent,
    SentimentScore,
)

FILE_TS = datetime(2024, 3, 5, 13, 15, tzinfo=UTC)


def _pair(i: int, *, ts: datetime = FILE_TS):
    ev = RawNewsEvent(
        source="gdelt_gkg",
        source_id=f"https://x.com/{i}",
        title=f"t{i}",
        snippet_or_text=f"t{i}",
        published_at=ts,
        observed_at=ts,
        ingested_at=datetime(2026, 1, 1, tzinfo=UTC),
        query_topic="gkg_markets",
        sentiment_feature_version="g1",
    )
    sc = ScoredNewsEvent(
        content_hash=ev.content_hash,
        source="gdelt_gkg",
        query_topic="gkg_markets",
        published_at=ts,
        observed_at=ts,
        sentiment_feature_version="g1",
        score=SentimentScore(
            positive_score=0.04,
            negative_score=0.02,
            neutral_score=0.94,
            sentiment_score=0.02,
            model_name="gdelt_v2tone",
            scored_at=datetime(2026, 1, 1, tzinfo=UTC),
        ),
    )
    return ev, sc


def test_write_creates_date_partition_and_reads_back(tmp_path):
    lake = GkgLake(root=tmp_path)
    raw, scored = zip(*[_pair(i) for i in range(3)], strict=True)
    n = lake.write_file(FILE_TS, list(raw), list(scored))
    assert n == 3
    part = tmp_path / "sentiment_gkg_raw" / "date=2024-03-05" / "part-20240305131500.parquet"
    assert part.exists()
    assert lake.read_raw().height == 3
    assert lake.read_scored().height == 3


def test_reprocessing_same_file_is_idempotent(tmp_path):
    lake = GkgLake(root=tmp_path)
    raw, scored = zip(*[_pair(i) for i in range(2)], strict=True)
    lake.write_file(FILE_TS, list(raw), list(scored))
    lake.write_file(FILE_TS, list(raw), list(scored))  # overwrite, not append
    assert lake.read_raw().height == 2
    assert lake.read_scored().height == 2


def test_within_file_content_hash_dedup(tmp_path):
    lake = GkgLake(root=tmp_path)
    ev, sc = _pair(0)
    lake.write_file(FILE_TS, [ev, ev], [sc, sc])  # identical pair twice in one file
    assert lake.read_raw().height == 1
    assert lake.read_scored().height == 1


def test_read_scored_date_range_filter(tmp_path):
    lake = GkgLake(root=tmp_path)
    r1, s1 = _pair(0, ts=datetime(2024, 3, 5, 0, 0, tzinfo=UTC))
    r2, s2 = _pair(1, ts=datetime(2024, 3, 7, 0, 0, tzinfo=UTC))
    lake.write_file(datetime(2024, 3, 5, 0, 0, tzinfo=UTC), [r1], [s1])
    lake.write_file(datetime(2024, 3, 7, 0, 0, tzinfo=UTC), [r2], [s2])
    assert lake.read_scored().height == 2
    only5 = lake.read_scored(start=date(2024, 3, 5), end=date(2024, 3, 6))
    assert only5.height == 1


def test_empty_write_and_read(tmp_path):
    lake = GkgLake(root=tmp_path)
    assert lake.write_file(FILE_TS, [], []) == 0
    assert lake.read_scored().height == 0  # typed empty frame, not an error
