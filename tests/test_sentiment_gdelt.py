"""GDELT adapter: fixture parse into the PIT schema + bounded URL construction."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from options_system.sentiment import gdelt
from options_system.sentiment.schema import dedupe_by_hash

_FIX = Path(__file__).parent / "fixtures" / "sentiment" / "gdelt_fed.json"
_ING = datetime(2026, 6, 1, tzinfo=UTC)


def test_parse_artlist_fixture():
    payload = json.loads(_FIX.read_text())
    events = gdelt.parse_artlist(
        payload, topic="fed", sentiment_feature_version="s1", ingested_at=_ING
    )
    assert len(events) == 5
    degraded = [e for e in events if e.degraded]
    assert len(degraded) == 1 and degraded[0].error

    good = [e for e in events if not e.degraded]
    a1 = good[0]
    assert a1.source == "gdelt"
    assert a1.published_at == datetime(2026, 2, 3, 14, 30, tzinfo=UTC)
    assert a1.observed_at == a1.published_at  # conservative: GDELT first-seen
    assert a1.ingested_at == _ING
    assert a1.sentiment_feature_version == "s1"

    # The duplicate article collapses on content_hash.
    assert len(dedupe_by_hash(events)) == 4


def test_build_query_url_is_bounded():
    url = gdelt.build_query_url(
        topic="fed",
        start=datetime(2026, 1, 1, tzinfo=UTC),
        end=datetime(2026, 1, 2, tzinfo=UTC),
        max_records=9999,  # over the 250 ceiling
        language="eng",
    )
    assert "api.gdeltproject.org" in url
    assert "maxrecords=250" in url  # hard-capped
    assert "mode=ArtList" in url
    assert "sourcelang%3Aeng" in url  # language folded into the query, url-encoded
