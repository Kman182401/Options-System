"""SEC EDGAR adapter: fixture parse into the PIT schema + CIK URL construction."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from options_system.sentiment import sec_edgar

_FIX = Path(__file__).parent / "fixtures" / "sentiment" / "sec_submissions.json"
_ING = datetime(2026, 6, 1, tzinfo=UTC)


def test_parse_submissions_fixture():
    payload = json.loads(_FIX.read_text())
    events = sec_edgar.parse_submissions(
        payload, topic="earnings", sentiment_feature_version="s1", ingested_at=_ING
    )
    assert len(events) == 2
    e0 = events[0]
    assert e0.source == "sec_edgar"
    assert e0.source_id == "0000320193-26-000010"
    assert e0.published_at == datetime(2026, 2, 1, 16, 30, tzinfo=UTC)
    assert e0.observed_at == e0.published_at  # public at acceptance time
    assert e0.ingested_at == _ING
    assert "AAPL" in e0.entities
    assert e0.source_url and "320193" in e0.source_url
    assert e0.query_topic == "earnings"


def test_build_submissions_url_pads_cik():
    assert sec_edgar.build_submissions_url(320193).endswith("CIK0000320193.json")
    assert sec_edgar.build_submissions_url("5").endswith("CIK0000000005.json")
