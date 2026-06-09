"""Sentiment health: the pure summary over raw (+ scored) frames."""

from __future__ import annotations

from datetime import UTC, datetime

from options_system.observability.sentiment_health import gather_sentiment_health
from options_system.sentiment.lake import events_to_frame, scored_to_frame
from options_system.sentiment.schema import (
    RawNewsEvent,
    ScoredNewsEvent,
    SentimentScore,
)


def _ev(source: str, source_id: str, topic: str) -> RawNewsEvent:
    return RawNewsEvent(
        source=source,
        source_id=source_id,
        title=f"t {source_id}",
        snippet_or_text="body",
        published_at=datetime(2026, 2, 3, tzinfo=UTC),
        observed_at=datetime(2026, 2, 3, tzinfo=UTC),
        ingested_at=datetime(2026, 2, 4, tzinfo=UTC),
        query_topic=topic,
        sentiment_feature_version="s1",
    )


def test_health_core_fields():
    raw = events_to_frame(
        [
            _ev("gdelt", "a", "fed"),
            _ev("gdelt", "b", "inflation"),
            _ev("sec_edgar", "c", "earnings"),
        ]
    )
    info = gather_sentiment_health(
        raw, None, source_policy={"gdelt": "free_no_auth"}, network_used=False
    )
    assert info["rows"] == 3
    assert info["rows_by_source"] == {"gdelt": 2, "sec_edgar": 1}
    assert set(info["rows_by_topic"]) == {"fed", "inflation", "earnings"}
    assert info["published_at"]["min"] is not None
    assert info["observed_at"]["max"] is not None
    assert info["duplicate_rate"] == 0.0
    assert info["missing_timestamp_count"] == 0
    assert info["network_used"] is False
    assert info["source_policy_status"]["gdelt"] == "free_no_auth"
    # sec_edgar policy not supplied -> falls back to the authoritative code registry.
    assert info["source_policy_status"]["sec_edgar"] == "free_no_auth"


def test_health_duplicate_rate():
    a = _ev("gdelt", "a", "fed")
    raw = events_to_frame([a, a])  # identical hash twice
    info = gather_sentiment_health(raw)
    assert info["rows"] == 2
    assert info["duplicate_rate"] == 0.5


def test_health_empty_frame():
    info = gather_sentiment_health(events_to_frame([]))
    assert info["rows"] == 0
    assert info["rows_by_source"] == {}
    assert info["duplicate_rate"] == 0.0


def test_health_scored_distribution():
    a = _ev("gdelt", "a", "fed")
    scored = [
        ScoredNewsEvent(
            content_hash=a.content_hash,
            source=a.source,
            query_topic=a.query_topic,
            published_at=a.published_at,
            observed_at=a.observed_at,
            sentiment_feature_version="s1",
            score=SentimentScore(
                positive_score=0.7,
                negative_score=0.1,
                neutral_score=0.2,
                sentiment_score=0.6,
                model_name="fake-lexicon-v1",
                scored_at=datetime(2026, 2, 4, 1, tzinfo=UTC),
            ),
        )
    ]
    info = gather_sentiment_health(events_to_frame([a]), scored_to_frame(scored))
    assert info["scored"]["rows"] == 1
    assert abs(info["scored"]["sentiment_score"]["mean"] - 0.6) < 1e-9
    assert info["scored"]["models"] == ["fake-lexicon-v1"]
