"""Raw event schema: content hash, point-in-time invariants, dedup, PIT filter."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from typing import Any

import pytest

from options_system.sentiment.schema import (
    RawNewsEvent,
    compute_content_hash,
    dedupe_by_hash,
    filter_point_in_time,
)


def _ev(**kw) -> RawNewsEvent:
    base: dict[str, Any] = dict(
        source="gdelt",
        source_id="id1",
        title="t",
        snippet_or_text="body",
        published_at=datetime(2026, 2, 3, 14, 30, tzinfo=UTC),
        observed_at=datetime(2026, 2, 3, 14, 30, tzinfo=UTC),
        ingested_at=datetime(2026, 2, 4, 0, 0, tzinfo=UTC),
        query_topic="fed",
        sentiment_feature_version="s1",
    )
    base.update(kw)
    return RawNewsEvent(**base)


def test_content_hash_autofilled_and_stable():
    e = _ev()
    assert e.content_hash
    assert e.content_hash == compute_content_hash("gdelt", "id1", "t", "body", e.published_at)


def test_observed_before_published_rejected():
    with pytest.raises(ValueError):
        _ev(observed_at=datetime(2026, 2, 3, 14, 29, tzinfo=UTC))


def test_ingested_before_observed_rejected():
    with pytest.raises(ValueError):
        _ev(ingested_at=datetime(2026, 2, 3, 14, 0, tzinfo=UTC))


def test_dedupe_keeps_latest_ingest():
    e1 = _ev()
    e2 = _ev(ingested_at=datetime(2026, 2, 5, 0, 0, tzinfo=UTC))  # same hash, later ingest
    out = dedupe_by_hash([e1, e2])
    assert len(out) == 1
    assert out[0].ingested_at == e2.ingested_at


def test_distinct_items_not_merged():
    a = _ev(source_id="a", title="alpha")
    b = _ev(source_id="b", title="beta")
    assert a.content_hash != b.content_hash
    assert len(dedupe_by_hash([a, b])) == 2


def test_point_in_time_filter():
    early = _ev(
        source_id="a",
        published_at=datetime(2026, 2, 3, tzinfo=UTC),
        observed_at=datetime(2026, 2, 3, tzinfo=UTC),
    )
    late = _ev(
        source_id="b",
        published_at=datetime(2026, 2, 10, tzinfo=UTC),
        observed_at=datetime(2026, 2, 10, tzinfo=UTC),
        ingested_at=datetime(2026, 2, 11, tzinfo=UTC),
    )
    degraded = _ev(source_id="c", degraded=True, error="boom")
    as_of = datetime(2026, 2, 5, tzinfo=UTC)
    kept = filter_point_in_time([early, late, degraded], as_of)
    assert {e.source_id for e in kept} == {"a"}  # late excluded (future), degraded excluded


def test_content_hash_is_timezone_independent():
    # The SAME instant expressed in different tz offsets must hash identically — the
    # digest is canonicalised to UTC so dedup works across hosts/timezones.
    inst_utc = datetime(2026, 2, 3, 14, 30, tzinfo=UTC)
    inst_tokyo = inst_utc.astimezone(timezone(timedelta(hours=9)))  # same instant, +09:00
    assert compute_content_hash("gdelt", "id1", "t", "body", inst_utc) == compute_content_hash(
        "gdelt", "id1", "t", "body", inst_tokyo
    )
    # And the event-level hash matches regardless of the input offset.
    assert (
        _ev(published_at=inst_utc).content_hash
        == _ev(published_at=inst_tokyo, observed_at=inst_tokyo).content_hash
    )
