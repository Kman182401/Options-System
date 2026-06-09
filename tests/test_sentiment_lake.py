"""Sentiment lake: idempotent raw + scored writes into a tmp root (no real lake)."""

from __future__ import annotations

from datetime import UTC, datetime

from options_system.sentiment.lake import SentimentLake
from options_system.sentiment.schema import (
    RawNewsEvent,
    ScoredNewsEvent,
    SentimentScore,
)


def _ev(source_id: str) -> RawNewsEvent:
    return RawNewsEvent(
        source="gdelt",
        source_id=source_id,
        title=f"title {source_id}",
        snippet_or_text=f"body {source_id}",
        published_at=datetime(2026, 2, 3, 14, 30, tzinfo=UTC),
        observed_at=datetime(2026, 2, 3, 14, 30, tzinfo=UTC),
        ingested_at=datetime(2026, 2, 4, tzinfo=UTC),
        query_topic="fed",
        sentiment_feature_version="s1",
    )


def test_write_raw_is_idempotent(tmp_path):
    lake = SentimentLake(root=tmp_path)
    events = [_ev("a"), _ev("b")]
    assert lake.write_raw(events) == 2
    assert lake.write_raw(events) == 0  # rerun writes nothing new
    df = lake.read_raw("gdelt")
    assert df.height == 2
    assert set(df["source_id"].to_list()) == {"a", "b"}


def test_write_raw_dedupes_within_batch(tmp_path):
    lake = SentimentLake(root=tmp_path)
    assert lake.write_raw([_ev("a"), _ev("a")]) == 1  # same hash collapses


def _scored(ev: RawNewsEvent) -> ScoredNewsEvent:
    return ScoredNewsEvent(
        content_hash=ev.content_hash,
        source=ev.source,
        query_topic=ev.query_topic,
        published_at=ev.published_at,
        observed_at=ev.observed_at,
        sentiment_feature_version=ev.sentiment_feature_version,
        score=SentimentScore(
            positive_score=0.6,
            negative_score=0.1,
            neutral_score=0.3,
            sentiment_score=0.5,
            model_name="fake-lexicon-v1",
            model_version_or_hash="v1",
            scored_at=datetime(2026, 2, 4, 1, tzinfo=UTC),
        ),
    )


def test_write_scored_is_idempotent(tmp_path):
    lake = SentimentLake(root=tmp_path)
    scored = [_scored(_ev("a")), _scored(_ev("b"))]
    assert lake.write_scored(scored) == 2
    assert lake.write_scored(scored) == 0  # same (hash, model) -> nothing new
    assert lake.read_scored().height == 2
