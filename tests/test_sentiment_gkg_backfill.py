"""GKG bulk-backfill runner tests — ALL offline (injected fetcher/clock/sleep, tmp lake)."""

from __future__ import annotations

import email.message
import io
import urllib.request
import zipfile
from datetime import UTC, date, datetime
from urllib.error import HTTPError

import pytest

from options_system.sentiment import gkg_backfill as gb
from options_system.sentiment.gkg_backfill import (
    FAILED,
    MISSING,
    OK,
    Counters,
    GkgBackfillRunner,
    GkgManifest,
    build_file_plan,
    file_timestamp,
    floor_to_slot,
)
from options_system.sentiment.gkg_config import GkgConfig
from options_system.sentiment.gkg_lake import GkgLake

NOW = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    def _boom(*a, **k):
        raise AssertionError("network call attempted in an offline test")

    monkeypatch.setattr(urllib.request, "urlopen", _boom)


def _gkg_text(ts: datetime, *, theme: str = "ECON_STOCKMARKET", n: int = 1) -> str:
    rows = []
    for i in range(n):
        f = [""] * 27
        f[0] = f"{file_timestamp(ts)}-{i}"
        f[1] = file_timestamp(ts)
        f[3] = "reuters.com"
        f[4] = f"https://reuters.com/{file_timestamp(ts)}/{i}"
        f[7] = theme
        f[15] = "2.0,4.0,2.0,6.0,20,5,150"
        f[26] = f"<PAGE_TITLE>headline {i}</PAGE_TITLE>"
        rows.append("\t".join(f))
    return "\n".join(rows)


def _zip_bytes(text: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("x.gkg.csv", text)
    return buf.getvalue()


def _http_error(code: int) -> HTTPError:
    return HTTPError("http://x", code, "err", email.message.Message(), None)


def _runner(tmp_path, fetcher, *, manifest=None, **caps):
    cfg = GkgConfig.load()
    lake = GkgLake(
        root=tmp_path, raw_dataset="sentiment_gkg_raw", scored_dataset="sentiment_gkg_scores"
    )
    man = manifest or GkgManifest.load_or_create(tmp_path / "manifest.json", meta={})
    defaults = dict(
        max_files=10_000,
        max_wall_clock_minutes=10_000,
        max_bytes=10**12,
        workers=1,
        politeness_delay_s=0.0,
        retry_max=cfg.backfill.retry_max,
    )
    defaults.update(caps)
    return (
        GkgBackfillRunner(
            cfg=cfg,
            lake=lake,
            manifest=man,
            fetcher=fetcher,
            sleep=lambda _s: None,
            monotonic=lambda: 0.0,
            now_fn=lambda: NOW,
            counters=Counters(),
            **defaults,
        ),
        lake,
        man,
    )


# --- plan --------------------------------------------------------------------- #


def test_plan_count_and_order():
    plan = build_file_plan(date(2024, 1, 1), date(2024, 1, 2))
    assert len(plan) == 192  # 2 days * 96 slots
    assert plan == sorted(plan)
    assert file_timestamp(plan[0]) == "20240101000000"
    assert file_timestamp(plan[-1]) == "20240102234500"


def test_plan_rejects_reversed_range():
    with pytest.raises(ValueError, match="precedes start"):
        build_file_plan(date(2024, 1, 2), date(2024, 1, 1))


def test_floor_to_slot():
    assert floor_to_slot(datetime(2024, 1, 1, 3, 47, 22, tzinfo=UTC)) == datetime(
        2024, 1, 1, 3, 45, tzinfo=UTC
    )


# --- runner ------------------------------------------------------------------- #


def test_happy_path_writes_and_records(tmp_path):
    plan = build_file_plan(date(2024, 1, 1), date(2024, 1, 1))[:3]
    bodies = {gb.file_url(ts): (200, _zip_bytes(_gkg_text(ts, n=2))) for ts in plan}

    def fetch(url):
        return bodies[url]

    runner, lake, man = _runner(tmp_path, fetch)
    outcome = runner.run(plan)
    assert outcome == "completed"
    assert runner.counters.files_ok == 3
    assert runner.counters.records_written == 6  # 2 events/file * 3
    assert lake.read_scored().height == 6
    summ = man.summarize()
    assert summ["files_by_status"][OK] == 3 and summ["records_written"] == 6


def test_missing_file_is_done_not_failed(tmp_path):
    plan = build_file_plan(date(2024, 1, 1), date(2024, 1, 1))[:1]

    def fetch(url):
        return 404, None

    runner, lake, man = _runner(tmp_path, fetch)
    runner.run(plan)
    assert runner.counters.files_missing == 1 and runner.counters.files_ok == 0
    assert man.is_done(file_timestamp(plan[0])) is True  # missing counts as done
    assert lake.read_scored().height == 0


def test_idempotent_rerun_writes_nothing_new(tmp_path):
    plan = build_file_plan(date(2024, 1, 1), date(2024, 1, 1))[:2]
    bodies = {gb.file_url(ts): (200, _zip_bytes(_gkg_text(ts))) for ts in plan}
    runner, lake, man = _runner(tmp_path, lambda u: bodies[u])
    runner.run(plan)
    first = lake.read_scored().height
    # A fresh runner over the SAME manifest skips done files entirely.
    runner2, _, _ = _runner(tmp_path, lambda u: bodies[u], manifest=man)
    runner2.run(plan)
    assert runner2.counters.files_ok == 0  # nothing re-fetched
    assert lake.read_scored().height == first


def test_failed_file_retried_on_resume(tmp_path):
    plan = build_file_plan(date(2024, 1, 1), date(2024, 1, 1))[:1]
    state = {"fail": True}

    def fetch(u):
        if state["fail"]:
            raise _http_error(500)
        return 200, _zip_bytes(_gkg_text(plan[0]))

    runner, lake, man = _runner(tmp_path, fetch)
    runner.run(plan)
    assert runner.counters.files_failed == 1
    assert man.is_done(file_timestamp(plan[0])) is False  # failed -> not done
    # Now the server "recovers"; resume retries the failed file and succeeds.
    state["fail"] = False
    runner2, _, _ = _runner(tmp_path, fetch, manifest=man)
    runner2.run(plan)
    assert runner2.counters.files_ok == 1
    assert lake.read_scored().height == 1


def test_max_files_cap_is_exact_no_overdownload(tmp_path):
    plan = build_file_plan(date(2024, 1, 1), date(2024, 1, 1))[:4]
    bodies = {gb.file_url(ts): (200, _zip_bytes(_gkg_text(ts))) for ts in plan}
    calls = {"n": 0}

    def fetch(u):
        calls["n"] += 1
        return bodies[u]

    runner, _, man = _runner(tmp_path, fetch, max_files=2)
    outcome = runner.run(plan)
    assert outcome == "capped_max_files"
    assert runner.counters.files_attempted == 2
    assert calls["n"] == 2  # the cap is EXACT: the chunk never prefetches beyond max_files
    assert man.summarize()["files_attempted"] == 2


def test_corrupt_zip_recorded_failed(tmp_path):
    plan = build_file_plan(date(2024, 1, 1), date(2024, 1, 1))[:1]

    def fetch(u):
        return 200, b"not-a-zip"

    runner, lake, man = _runner(tmp_path, fetch)
    runner.run(plan)
    assert runner.counters.files_failed == 1
    assert man.data["files"][file_timestamp(plan[0])]["status"] == FAILED
    assert lake.read_scored().height == 0


def test_resume_refuses_incompatible_manifest(tmp_path, monkeypatch):
    # A manifest built with different theme_prefixes than the current config must NOT be
    # resumed (it would skip files processed under the old config). Fail closed, exit 2.
    mpath = tmp_path / "manifest.json"
    monkeypatch.setattr(gb, "default_manifest_path", lambda: mpath)
    man = GkgManifest.load_or_create(
        mpath,
        meta={"theme_prefixes": ["SPORTS_"], "start": "2019-01-01", "end": "2019-01-01"},
    )
    man.record("20190101000000", status=OK)
    man.save()
    args = gb.build_arg_parser().parse_args(
        ["--resume", "--allow-network", "--start", "2019-01-01", "--end", "2019-01-01"]
    )
    assert gb.run(args) == 2  # blocked: config theme_prefixes (ECON_/EPU_) != manifest's


def test_resume_accepts_matching_manifest(tmp_path, monkeypatch):
    # The same config resumes cleanly (no false trip). Uses an injected no-network fetcher
    # via monkeypatching the runner factory's fetch is overkill; instead seed a fully-done
    # manifest so run() does no fetching and completes.
    from options_system.sentiment.gkg_config import GkgConfig

    cfg = GkgConfig.load()
    mpath = tmp_path / "manifest.json"
    monkeypatch.setattr(gb, "default_manifest_path", lambda: mpath)
    GkgManifest.load_or_create(
        mpath,
        meta={
            "theme_prefixes": list(cfg.theme_prefixes),
            "start": "2019-01-01",
            "end": "2019-01-01",
            "gkg_event_version": cfg.gkg_event_version,
            "tone_model_name": cfg.tone_model_name,
        },
    ).save()
    # Single-day plan with every file already "done" so no network is needed.
    plan = build_file_plan(date(2019, 1, 1), date(2019, 1, 1))
    man = GkgManifest.load_or_create(mpath, meta={})
    for ts in plan:
        man.record(file_timestamp(ts), status=MISSING)
    man.save()
    args = gb.build_arg_parser().parse_args(
        ["--resume", "--allow-network", "--start", "2019-01-01", "--end", "2019-01-01"]
    )

    def _boom(*a, **k):
        raise AssertionError("should not fetch — all files are already done")

    monkeypatch.setattr(gb, "default_fetcher", _boom)
    assert gb.run(args) == 0  # matching fingerprint resumes, nothing to fetch


def test_manifest_atomic_save_roundtrip(tmp_path):
    man = GkgManifest.load_or_create(tmp_path / "m.json", meta={"k": "v"})
    man.record("20240101000000", status=OK, n_kept=3)
    man.save()
    reloaded = GkgManifest.load_or_create(tmp_path / "m.json", meta={})
    assert reloaded.data["files"]["20240101000000"]["n_kept"] == 3
    assert reloaded.is_done("20240101000000") is True
