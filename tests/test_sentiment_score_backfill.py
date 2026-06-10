"""Phase 18 batch-scoring tests — offline, FakeScorer + tmp lake, never the network."""

from __future__ import annotations

import urllib.request
from datetime import UTC, datetime, timedelta

import pytest

from options_system.sentiment.lake import SentimentLake
from options_system.sentiment.schema import RawNewsEvent
from options_system.sentiment.score_backfill import (
    resolve_local_revision,
    score_frame,
    select_unscored,
)
from options_system.sentiment.scoring import FakeScorer

T0 = datetime(2026, 6, 1, 12, 0, 0, tzinfo=UTC)
MODEL = f"{FakeScorer.name}-{FakeScorer.version}"


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    def _boom(*args, **kwargs):
        raise AssertionError("network call attempted in an offline test")

    monkeypatch.setattr(urllib.request, "urlopen", _boom)


def _events() -> list[RawNewsEvent]:
    out = []
    for i, (title, degraded) in enumerate(
        [
            ("stocks surge on strong earnings", False),
            ("recession fears trigger selloff", False),
            ("fed holds rates steady", False),
            ("(missing title)", True),
        ]
    ):
        out.append(
            RawNewsEvent(
                source="gdelt",
                source_id=f"https://example.com/{i}",
                title=title,
                snippet_or_text=title if not degraded else "",
                published_at=T0 + timedelta(minutes=i),
                observed_at=T0 + timedelta(minutes=i),
                ingested_at=T0 + timedelta(hours=1),
                query_topic="fed",
                sentiment_feature_version="s1",
                degraded=degraded,
                error="missing seendate or title" if degraded else None,
            )
        )
    return out


def test_select_unscored_excludes_degraded_and_is_ordered(tmp_path):
    lake = SentimentLake(root=tmp_path)
    assert lake.write_raw(_events()) == 4
    unscored = select_unscored(lake.read_raw(), lake.read_scored(), MODEL)
    assert unscored.height == 3  # degraded row excluded
    observed = unscored["observed_at"].to_list()
    assert observed == sorted(observed)  # deterministic order


def test_scoring_is_idempotent_on_hash_and_model(tmp_path):
    lake = SentimentLake(root=tmp_path)
    lake.write_raw(_events())
    scorer = FakeScorer(now=T0 + timedelta(hours=2))

    unscored = select_unscored(lake.read_raw(), lake.read_scored(), MODEL)
    scored_events = score_frame(unscored, scorer)
    assert len(scored_events) == 3
    assert lake.write_scored(scored_events) == 3

    # second pass: nothing left to score, and a re-write writes nothing new
    again = select_unscored(lake.read_raw(), lake.read_scored(), MODEL)
    assert again.height == 0
    assert lake.write_scored(scored_events) == 0

    # a DIFFERENT model is a different idempotency key: all 3 select again
    other = select_unscored(lake.read_raw(), lake.read_scored(), "some-other-model")
    assert other.height == 3


def test_score_frame_carries_pit_fields_and_scores(tmp_path):
    lake = SentimentLake(root=tmp_path)
    lake.write_raw(_events())
    unscored = select_unscored(lake.read_raw(), lake.read_scored(), MODEL)
    scored_events = score_frame(unscored, FakeScorer(now=T0 + timedelta(hours=2)))
    by_title = {s.content_hash: s for s in scored_events}
    assert len(by_title) == 3
    for s in scored_events:
        assert s.observed_at >= T0
        assert s.sentiment_feature_version == "s1"
        assert s.score.model_name == MODEL
        assert -1.0 <= s.score.sentiment_score <= 1.0
    # FakeScorer is deterministic: "surge ... strong" is positive, "recession ... selloff" negative
    sentiments = {ev.content_hash: ev.score.sentiment_score for ev in scored_events}
    assert max(sentiments.values()) > 0
    assert min(sentiments.values()) < 0


def test_score_frame_empty_is_noop():
    import polars as pl

    from options_system.sentiment.lake import _RAW_SCHEMA

    empty = pl.DataFrame(schema=_RAW_SCHEMA)
    assert score_frame(empty, FakeScorer()) == []


def test_resolve_local_revision_uncached_model_is_none():
    # local_files_only lookup of a never-cached model: no network, returns None
    assert resolve_local_revision("options-system/definitely-not-cached") is None
