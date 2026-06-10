"""Phase 18 backfill orchestrator tests — ALL offline.

Every test monkeypatches ``urllib.request.urlopen`` to raise, injects a fake
fetcher/clock/sleep/rng, and writes only to tmp paths. No real network, no real
sleeps, no real lake.
"""

from __future__ import annotations

import email.message
import random
import urllib.request
from datetime import UTC, date, datetime, time, timedelta
from urllib.error import HTTPError

import pytest

from options_system.sentiment import backfill
from options_system.sentiment.backfill import (
    SUPPORTED,
    UNSUPPORTED,
    BackfillRunner,
    Manifest,
    PlanSlice,
    archive_cutoff,
    bisect_slice,
    build_request_plan,
    pending_slices,
)
from options_system.sentiment.config import SentimentConfig
from options_system.sentiment.gdelt import build_query_url
from options_system.sentiment.lake import SentimentLake
from options_system.sentiment.schema import RawNewsEvent

NOW = datetime(2026, 6, 9, 12, 0, 0, tzinfo=UTC)


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """Hard guarantee: no test in this module can touch the network."""

    def _boom(*args, **kwargs):
        raise AssertionError("network call attempted in an offline test")

    monkeypatch.setattr(urllib.request, "urlopen", _boom)


def _mk_events(n: int, *, tag: str, seen: datetime) -> list[RawNewsEvent]:
    return [
        RawNewsEvent(
            source="gdelt",
            source_id=f"https://example.com/{tag}/{i}",
            title=f"headline {tag} {i}",
            snippet_or_text=f"headline {tag} {i}",
            published_at=seen,
            observed_at=seen,
            ingested_at=seen + timedelta(hours=1),
            query_topic="fed",
            sentiment_feature_version="s1",
        )
        for i in range(n)
    ]


class _ZeroJitter(random.Random):
    """Deterministic rng: no jitter, so pacing/backoff asserts are exact."""

    def uniform(self, a: float, b: float) -> float:
        return 0.0


def _runner(tmp_path, fetcher, *, max_requests=10_000, max_wall_minutes=10_000, monotonic=None):
    sleeps: list[float] = []
    clock = monotonic if monotonic is not None else (lambda: 0.0)
    lake = SentimentLake(root=tmp_path / "lake")
    manifest = Manifest.load_or_create(tmp_path / "manifest.json", meta={"test": True})
    runner = BackfillRunner(
        lake=lake,
        manifest=manifest,
        fetcher=fetcher,
        max_requests=max_requests,
        max_wall_clock_minutes=max_wall_minutes,
        sleep=sleeps.append,
        monotonic=clock,
        rng=_ZeroJitter(),
        now_fn=lambda: NOW,
    )
    return runner, manifest, sleeps


# --- slicer ------------------------------------------------------------------- #


def test_plan_deterministic_one_day_utc_slices():
    plan_a = build_request_plan(["fed", "rates"], date(2026, 6, 1), date(2026, 6, 3), now=NOW)
    plan_b = build_request_plan(["fed", "rates"], date(2026, 6, 1), date(2026, 6, 3), now=NOW)
    assert plan_a == plan_b  # deterministic
    assert len(plan_a) == 3 * 2  # inclusive date range x topics
    for s in plan_a:
        assert s.start.tzinfo is UTC
        assert s.start.time() == time(0)
        assert s.end - s.start == timedelta(days=1)
    # date-major within region, topics in given order
    assert [s.topic for s in plan_a[:2]] == ["fed", "rates"]
    assert plan_a[0].start == datetime(2026, 6, 1, tzinfo=UTC)


def test_plan_archive_classification_supported_first():
    cutoff = archive_cutoff(NOW)
    assert cutoff == NOW - timedelta(days=92)
    plan = build_request_plan(["fed"], date(2026, 1, 26), date(2026, 6, 6), now=NOW)
    regions = {s.start.date(): s.archive_region for s in plan}
    assert regions[date(2026, 1, 26)] == UNSUPPORTED
    assert regions[date(2026, 6, 6)] == SUPPORTED
    assert regions[date(2026, 3, 8)] == UNSUPPORTED  # cutoff is 2026-03-09 12:00 UTC
    assert regions[date(2026, 3, 10)] == SUPPORTED
    # supported region ordered first so a capped run spends budget where gates live
    first_unsupported = next(i for i, s in enumerate(plan) if s.archive_region == UNSUPPORTED)
    assert all(s.archive_region == SUPPORTED for s in plan[:first_unsupported])
    assert all(s.archive_region == UNSUPPORTED for s in plan[first_unsupported:])


def test_plan_rejects_reversed_range():
    with pytest.raises(ValueError, match="precedes"):
        build_request_plan(["fed"], date(2026, 6, 2), date(2026, 6, 1), now=NOW)


def test_bisect_slice_floor_one_hour():
    day = PlanSlice(
        "fed", datetime(2026, 6, 1, tzinfo=UTC), datetime(2026, 6, 2, tzinfo=UTC), SUPPORTED
    )
    halves_day = bisect_slice(day)
    assert halves_day is not None
    left, right = halves_day
    assert left.end == right.start == datetime(2026, 6, 1, 12, tzinfo=UTC)
    assert left.archive_region == SUPPORTED
    two_h = PlanSlice("fed", day.start, day.start + timedelta(hours=2), SUPPORTED)
    halves = bisect_slice(two_h)
    assert halves is not None
    assert halves[0].end - halves[0].start == timedelta(hours=1)
    # halves would drop below 1 hour -> no further bisection
    assert (
        bisect_slice(
            PlanSlice("fed", day.start, day.start + timedelta(hours=1, minutes=59), SUPPORTED)
        )
        is None
    )


def test_query_text_overrides_topic_in_url_only():
    url_label = build_query_url(
        topic="ai_capex", start=NOW, end=NOW + timedelta(days=1), max_records=250
    )
    url_query = build_query_url(
        topic="ai_capex",
        start=NOW,
        end=NOW + timedelta(days=1),
        max_records=250,
        query_text='("AI capex" OR "AI infrastructure")',
    )
    assert "ai_capex" in url_label
    assert "ai_capex" not in url_query
    assert "AI+capex" in url_query or "AI%20capex" in url_query


# --- truncation + bisection ---------------------------------------------------- #


def test_runner_bisects_on_truncation_down_to_floor(tmp_path, monkeypatch):
    monkeypatch.setattr(backfill, "MAX_RECORDS", 5)
    seen = datetime(2026, 6, 1, 6, tzinfo=UTC)
    calls: list[str] = []

    def fetcher(topic, start, end):
        calls.append(f"{start.isoformat()}|{end.isoformat()}")
        return _mk_events(5, tag=f"{start:%H%M}-{end:%H%M}", seen=seen)

    plan = [
        PlanSlice(
            "fed", datetime(2026, 6, 1, tzinfo=UTC), datetime(2026, 6, 2, tzinfo=UTC), SUPPORTED
        )
    ]
    runner, manifest, _ = _runner(tmp_path, fetcher)
    outcome = runner.run(plan)
    assert outcome == "completed"
    # full binary tree: 24h -> 12 -> 6 -> 3 -> 1.5h floor = 1+2+4+8+16 = 31 fetches
    assert len(calls) == 31
    entries = list(manifest.data["slices"].values())
    assert len(entries) == 31
    floors = [e for e in entries if e["truncated"] and not e["bisected"]]
    parents = [e for e in entries if e["bisected"]]
    assert len(floors) == 16 and len(parents) == 15
    for e in entries:
        dur = datetime.fromisoformat(e["end"]) - datetime.fromisoformat(e["start"])
        assert dur >= timedelta(hours=1)  # bisection never goes below the 1h floor
        assert e["failed"] is None


# --- pacing + backoff ----------------------------------------------------------- #


def test_pacing_enforces_min_interval(tmp_path):
    seen = datetime(2026, 6, 1, 6, tzinfo=UTC)

    def fetcher(topic, start, end):
        return _mk_events(1, tag=start.isoformat(), seen=seen)

    plan = build_request_plan(["fed", "rates"], date(2026, 6, 1), date(2026, 6, 1), now=NOW)
    runner, _, sleeps = _runner(tmp_path, fetcher)
    runner.run(plan)
    # first request unpaced; second waits the full interval (fake clock never advances)
    assert sleeps == [pytest.approx(5.0)]


def test_backoff_sequence_retry_cap_and_run_continues(tmp_path):
    seen = datetime(2026, 6, 1, 6, tzinfo=UTC)
    attempts: list[str] = []

    def fetcher(topic, start, end):
        if topic == "fed":
            attempts.append("429")
            raise HTTPError("https://x", 429, "Too Many Requests", email.message.Message(), None)
        return _mk_events(2, tag="ok", seen=seen)

    plan = build_request_plan(["fed", "rates"], date(2026, 6, 1), date(2026, 6, 1), now=NOW)
    runner, manifest, sleeps = _runner(tmp_path, fetcher)
    outcome = runner.run(plan)
    assert outcome == "completed"
    assert len(attempts) == 5  # bounded retries per slice
    assert runner.counters.http_429 == 5
    # sleeps: 4 backoffs interleaved with pacing sleeps; backoff sequence 10,20,40,80
    backoffs = [s for s in sleeps if s > 5.0]
    assert backoffs == [10.0, 20.0, 40.0, 80.0]
    fed_key = plan[0].key
    assert manifest.entry(fed_key)["failed"] == "rate_limited"
    # the run continued: the rates slice succeeded and was written
    rates_entry = manifest.entry(plan[1].key)
    assert rates_entry["failed"] is None and rates_entry["records_written"] == 2


def test_backoff_honors_retry_after_capped():
    assert BackfillRunner._backoff_delay(1, None) == 10.0
    assert BackfillRunner._backoff_delay(2, None) == 20.0
    assert BackfillRunner._backoff_delay(5, None) == 120.0  # capped
    assert BackfillRunner._backoff_delay(1, "60") == 60.0  # Retry-After honored
    assert BackfillRunner._backoff_delay(1, "999") == 120.0  # ... but still capped
    assert BackfillRunner._backoff_delay(1, "soon") == 10.0  # non-numeric ignored


def test_non_429_http_error_marked_failed(tmp_path):
    def fetcher(topic, start, end):
        raise HTTPError("https://x", 503, "Service Unavailable", email.message.Message(), None)

    plan = build_request_plan(["fed"], date(2026, 6, 1), date(2026, 6, 1), now=NOW)
    runner, manifest, _ = _runner(tmp_path, fetcher)
    assert runner.run(plan) == "completed"
    assert manifest.entry(plan[0].key)["failed"] == "http_503"
    assert runner.counters.http_429 == 0


# --- caps (fail-closed) ----------------------------------------------------------- #


def test_request_cap_stops_cleanly_with_checkpoint(tmp_path):
    seen = datetime(2026, 6, 1, 6, tzinfo=UTC)

    def fetcher(topic, start, end):
        return _mk_events(1, tag=start.isoformat() + topic, seen=seen)

    plan = build_request_plan(["fed"], date(2026, 6, 1), date(2026, 6, 4), now=NOW)
    runner, manifest, _ = _runner(tmp_path, fetcher, max_requests=2)
    outcome = runner.run(plan)
    assert outcome == "capped_max_requests"
    assert runner.counters.requests_made == 2
    assert len(manifest.data["slices"]) == 2  # checkpoint intact
    assert manifest.data["runs"][-1]["outcome"] == "capped_max_requests"


def test_wall_clock_cap_stops_cleanly(tmp_path):
    seen = datetime(2026, 6, 1, 6, tzinfo=UTC)
    ticks = iter(range(0, 10_000_000, 600))  # +10 min per monotonic() call

    def fetcher(topic, start, end):
        return _mk_events(1, tag=start.isoformat(), seen=seen)

    plan = build_request_plan(["fed"], date(2026, 6, 1), date(2026, 6, 4), now=NOW)
    runner, manifest, _ = _runner(
        tmp_path, fetcher, max_wall_minutes=5, monotonic=lambda: float(next(ticks))
    )
    outcome = runner.run(plan)
    assert outcome == "capped_max_wall_clock_minutes"
    assert manifest.data["runs"][-1]["outcome"] == "capped_max_wall_clock_minutes"


# --- checkpoint / resume ------------------------------------------------------------ #


def test_resume_skips_completed_and_retries_failed(tmp_path):
    seen = datetime(2026, 6, 1, 6, tzinfo=UTC)
    plan = build_request_plan(["fed"], date(2026, 6, 1), date(2026, 6, 4), now=NOW)

    def fetcher_first(topic, start, end):
        return _mk_events(1, tag=start.isoformat(), seen=seen)

    runner, manifest, _ = _runner(tmp_path, fetcher_first, max_requests=2)
    assert runner.run(plan) == "capped_max_requests"

    fetched: list[str] = []

    def fetcher_second(topic, start, end):
        fetched.append(start.isoformat())
        return _mk_events(1, tag=start.isoformat(), seen=seen)

    # fresh runner over the SAME manifest file (as a real resume would)
    manifest2 = Manifest.load_or_create(tmp_path / "manifest.json", meta={})
    lake = SentimentLake(root=tmp_path / "lake")
    runner2 = BackfillRunner(
        lake=lake,
        manifest=manifest2,
        fetcher=fetcher_second,
        max_requests=100,
        max_wall_clock_minutes=100,
        sleep=lambda s: None,
        monotonic=lambda: 0.0,
        now_fn=lambda: NOW,
    )
    assert runner2.run(plan) == "completed"
    assert fetched == ["2026-06-03T00:00:00+00:00", "2026-06-04T00:00:00+00:00"]
    assert len(manifest2.data["slices"]) == 4


def test_resume_rederives_pending_bisection_children(tmp_path):
    parent = PlanSlice(
        "fed", datetime(2026, 6, 1, tzinfo=UTC), datetime(2026, 6, 2, tzinfo=UTC), SUPPORTED
    )
    manifest = Manifest.load_or_create(tmp_path / "manifest.json", meta={})
    manifest.record_slice(
        parent,
        http_status=200,
        records_returned=250,
        records_written=250,
        truncated=True,
        bisected=True,
        failed=None,
        completed_at=NOW.isoformat(),
    )
    halves = bisect_slice(parent)
    assert halves is not None
    left, right = halves
    # left child completed before the interrupt; right child still pending
    manifest.record_slice(
        left,
        http_status=200,
        records_returned=10,
        records_written=10,
        truncated=False,
        bisected=False,
        failed=None,
        completed_at=NOW.isoformat(),
    )
    pending = pending_slices([parent], manifest)
    assert pending == [right]


def test_failed_slices_are_retried_on_resume(tmp_path):
    s = PlanSlice(
        "fed", datetime(2026, 6, 1, tzinfo=UTC), datetime(2026, 6, 2, tzinfo=UTC), SUPPORTED
    )
    manifest = Manifest.load_or_create(tmp_path / "manifest.json", meta={})
    manifest.record_slice(
        s,
        http_status=429,
        records_returned=0,
        records_written=0,
        truncated=False,
        bisected=False,
        failed="rate_limited",
        completed_at=NOW.isoformat(),
    )
    assert pending_slices([s], manifest) == [s]


def test_manifest_atomic_roundtrip_and_summary(tmp_path):
    path = tmp_path / "manifest.json"
    manifest = Manifest.load_or_create(path, meta={"a": 1})
    s = PlanSlice(
        "fed", datetime(2026, 6, 1, tzinfo=UTC), datetime(2026, 6, 2, tzinfo=UTC), SUPPORTED
    )
    idx = manifest.start_run(started_at=NOW.isoformat(), kind="backfill")
    manifest.record_slice(
        s,
        http_status=200,
        records_returned=7,
        records_written=5,
        truncated=False,
        bisected=False,
        failed=None,
        completed_at=NOW.isoformat(),
    )
    manifest.end_run(
        idx, ended_at=NOW.isoformat(), outcome="completed", counters={"requests_made": 1}
    )
    reloaded = Manifest.load_or_create(path, meta={})
    assert reloaded.data["meta"] == {"a": 1}
    summary = reloaded.summarize()
    assert summary["slices_attempted"] == 1
    assert summary["records_written"] == 5
    assert summary["requests_made"] == 1
    assert summary["slices_by_region"][SUPPORTED] == 1
    assert not path.with_suffix(".json.tmp").exists()


# --- CLI ----------------------------------------------------------------------------- #


def test_plan_mode_zero_network(capsys):
    rc = backfill.main(["--plan", "--start", "2026-06-01", "--end", "2026-06-02"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "network: NONE (plan only)" in out
    assert "slices=18" in out  # 2 days x 9 topics


def test_cli_refuses_real_fetch_without_allow_network(capsys):
    rc = backfill.main(["--start", "2026-06-01", "--end", "2026-06-01"])
    assert rc == 2
    assert "BLOCKED" in capsys.readouterr().out


def test_cli_refuses_unknown_topic(capsys):
    rc = backfill.main(["--plan", "--topics", "nonsense_topic"])
    assert rc == 2
    assert "unknown topic" in capsys.readouterr().out


def test_config_backfill_block_loads_and_validates():
    cfg = SentimentConfig.load()
    assert cfg.backfill.max_requests == 2500
    assert cfg.backfill.max_wall_clock_minutes == 240
    assert set(cfg.backfill.topic_queries) <= set(cfg.query_topics)
    data = cfg.to_dict()
    data["backfill"]["topic_queries"] = {"not_a_topic": "x"}
    with pytest.raises(ValueError, match="unknown topic label"):
        SentimentConfig.model_validate(data)
