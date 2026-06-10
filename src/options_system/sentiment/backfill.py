"""Bounded, resumable GDELT historical backfill (Phase 18).

    uv run python -m options_system.sentiment.backfill \\
        --start 2026-01-26 --end 2026-06-06 [--topics ...] \\
        [--plan] [--probe] [--allow-network] [--resume]

Single responsibility: turn (topics x date range) into a resumable, paced, hard-capped
sequence of bounded GDELT ArtList fetches written idempotently into the existing raw
lake (:mod:`options_system.sentiment.lake`). It reuses the existing GDELT request
builder/parser (:mod:`options_system.sentiment.gdelt`) — there is no second GDELT
client — and the existing fail-closed policy gate: a real fetch requires the
``free_no_auth`` policy **and** an explicit ``--allow-network``.

Design (all fixed before any data was fetched — see docs/SENTIMENT.md Phase 18):

* **Slicing** — one UTC calendar day per topic by default. The slicer is a pure
  function (topics, dates, now) -> ordered plan.
* **Archive honesty** — GDELT's DOC 2.0 API officially supports only the most recent
  ~3 months. Every slice is classified ``supported`` / ``unsupported_archive`` at
  plan time; both are attempted, but coverage is reported per region and the
  unsupported region is never presented as guaranteed-complete. The supported region
  runs FIRST so the request budget is spent where the coverage gates are evaluated.
* **Truncation + bisection** — ArtList returns at most 250 records with no
  pagination. A slice returning exactly 250 is truncated: it is bisected and both
  halves re-fetched, recursively, until halves would drop below 1 hour; a floor
  slice still returning 250 is recorded ``truncated: true`` and not bisected.
* **Pacing/backoff** — >= 5 s + jitter between requests (GDELT enforces ~1 req/5 s
  per IP). On HTTP 429: exponential backoff 10 s doubling, capped 120 s, honoring
  ``Retry-After``, bounded retries per slice, then the slice is recorded
  ``failed: rate_limited`` and the run continues.
* **Resumability** — a JSON manifest under ``data/sentiment_backfill/`` (gitignored)
  records every slice attempt; completed slices are skipped on ``--resume``; failed
  slices are retried; pending bisection children are re-derived. The manifest is the
  audit trail the coverage report cites.
* **Hard caps (fail-closed)** — ``config/sentiment.yaml`` ``backfill.max_requests``
  and ``backfill.max_wall_clock_minutes``. Hitting either stops cleanly (exit 3)
  with the manifest intact.

The clock, sleep, RNG and fetcher are injectable so every test runs offline with no
real sleeps and no network.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import random
import time as time_module
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any
from urllib.error import HTTPError

from config.settings import Settings
from options_system.common.external_data_policy import assert_network_allowed
from options_system.sentiment import gdelt
from options_system.sentiment.config import SentimentConfig
from options_system.sentiment.lake import SentimentLake
from options_system.sentiment.schema import RawNewsEvent

# GDELT protocol facts (not tunables — changing these changes correctness):
MAX_RECORDS = 250  # ArtList hard ceiling; exactly this many returned == truncated
MIN_SLICE = timedelta(hours=1)  # bisection floor: halves never drop below 1 hour
REQUEST_INTERVAL_S = 5.0  # GDELT enforces ~1 request / 5 s per IP
JITTER_MAX_S = 1.0  # small random extra spacing on top of the interval
BACKOFF_INITIAL_S = 10.0  # first 429 backoff; doubles each retry
BACKOFF_MAX_S = 120.0  # backoff ceiling
MAX_RETRIES_PER_SLICE = 5  # attempts per slice before recording it failed
ARCHIVE_SUPPORTED_DAYS = 92  # ~3 months: GDELT's documented reliable archive depth

SUPPORTED = "supported"
UNSUPPORTED = "unsupported_archive"

#: fetcher signature: (topic_label, slice_start, slice_end) -> parsed raw events.
Fetcher = Callable[[str, datetime, datetime], list[RawNewsEvent]]


# --- pure plan ---------------------------------------------------------------- #


@dataclass(frozen=True)
class PlanSlice:
    """One bounded ArtList request: a topic label over [start, end) UTC."""

    topic: str
    start: datetime  # inclusive, UTC
    end: datetime  # exclusive, UTC
    archive_region: str  # SUPPORTED | UNSUPPORTED

    @property
    def key(self) -> str:
        return f"{self.topic}|{self.start.isoformat()}|{self.end.isoformat()}"


def archive_cutoff(now: datetime) -> datetime:
    """The oldest moment inside GDELT's officially supported ~3-month archive."""
    return now.astimezone(UTC) - timedelta(days=ARCHIVE_SUPPORTED_DAYS)


def build_request_plan(
    topics: Sequence[str],
    start: date,
    end: date,
    *,
    now: datetime,
) -> list[PlanSlice]:
    """Pure, deterministic request plan: one UTC calendar day per topic.

    ``start``..``end`` are inclusive dates. Slices are classified against the
    archive cutoff at ``now``; the SUPPORTED region is ordered FIRST (date
    ascending, topics in given order), then the unsupported region — so a
    request-capped run spends its budget where the coverage gates are evaluated.
    """
    if end < start:
        raise ValueError(f"backfill: end {end} precedes start {start}")
    cutoff = archive_cutoff(now)
    supported: list[PlanSlice] = []
    unsupported: list[PlanSlice] = []
    day = start
    while day <= end:
        lo = datetime.combine(day, time(0), UTC)
        hi = lo + timedelta(days=1)
        region = SUPPORTED if lo >= cutoff else UNSUPPORTED
        bucket = supported if region == SUPPORTED else unsupported
        for topic in topics:
            bucket.append(PlanSlice(topic=topic, start=lo, end=hi, archive_region=region))
        day += timedelta(days=1)
    return supported + unsupported


def bisect_slice(s: PlanSlice) -> tuple[PlanSlice, PlanSlice] | None:
    """Split a truncated slice in half; None when halves would drop below 1 hour.

    Children inherit the parent's archive region (a slice never spans the cutoff
    by more than its own width; the classification is a reporting label).
    """
    if (s.end - s.start) < 2 * MIN_SLICE:
        return None
    mid = s.start + (s.end - s.start) / 2
    left = PlanSlice(topic=s.topic, start=s.start, end=mid, archive_region=s.archive_region)
    right = PlanSlice(topic=s.topic, start=mid, end=s.end, archive_region=s.archive_region)
    return left, right


# --- manifest (checkpoint + audit trail) -------------------------------------- #


class Manifest:
    """JSON checkpoint/manifest under ``data/sentiment_backfill/`` (gitignored).

    One entry per attempted (topic, slice): request window, HTTP status, records
    returned/written, truncated/bisected/failed flags, timestamps. Atomic writes
    (tmp + rename) so an interrupt never corrupts it. Doubles as the audit trail
    cited by the coverage report.
    """

    def __init__(self, path: Path, data: dict[str, Any]) -> None:
        self.path = path
        self.data = data

    @classmethod
    def load_or_create(cls, path: Path, *, meta: dict[str, Any]) -> Manifest:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            data.setdefault("slices", {})
            data.setdefault("runs", [])
            return cls(path, data)
        return cls(path, {"meta": meta, "slices": {}, "runs": []})

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.data, indent=1, default=str), encoding="utf-8")
        os.replace(tmp, self.path)

    def entry(self, key: str) -> dict[str, Any] | None:
        return self.data["slices"].get(key)

    def is_done(self, key: str) -> bool:
        """Done == attempted and not failed. Failed slices are retried on resume."""
        e = self.entry(key)
        return e is not None and e.get("failed") is None

    def record_slice(self, s: PlanSlice, **fields: Any) -> None:
        self.data["slices"][s.key] = {
            "topic": s.topic,
            "start": s.start.isoformat(),
            "end": s.end.isoformat(),
            "archive_region": s.archive_region,
            **fields,
        }
        self.save()

    def start_run(self, *, started_at: str, kind: str) -> int:
        self.data["runs"].append({"kind": kind, "started_at": started_at, "outcome": "running"})
        self.save()
        return len(self.data["runs"]) - 1

    def end_run(self, idx: int, *, ended_at: str, outcome: str, counters: dict[str, int]) -> None:
        self.data["runs"][idx].update(
            {"ended_at": ended_at, "outcome": outcome, "counters": counters}
        )
        self.save()

    def summarize(self) -> dict[str, Any]:
        """Totals across all runs/slices — the numbers the coverage report cites."""
        slices = self.data["slices"].values()
        runs = self.data["runs"]
        by_region: dict[str, int] = {SUPPORTED: 0, UNSUPPORTED: 0}
        for e in slices:
            by_region[e.get("archive_region", UNSUPPORTED)] = (
                by_region.get(e.get("archive_region", UNSUPPORTED), 0) + 1
            )
        return {
            "slices_attempted": len(self.data["slices"]),
            "slices_ok": sum(1 for e in slices if e.get("failed") is None),
            "slices_failed": sum(1 for e in slices if e.get("failed") is not None),
            "slices_truncated": sum(1 for e in slices if e.get("truncated")),
            "slices_bisected": sum(1 for e in slices if e.get("bisected")),
            "slices_by_region": by_region,
            "records_returned": sum(int(e.get("records_returned") or 0) for e in slices),
            "records_written": sum(int(e.get("records_written") or 0) for e in slices),
            "requests_made": sum(
                int(r.get("counters", {}).get("requests_made") or 0) for r in runs
            ),
            "http_429": sum(int(r.get("counters", {}).get("http_429") or 0) for r in runs),
            "runs": len(runs),
        }


def pending_slices(plan: Sequence[PlanSlice], manifest: Manifest) -> list[PlanSlice]:
    """The slices a (re)run must still fetch, in order.

    Completed slices are skipped; failed slices are retried; for completed slices
    recorded ``bisected``, the two children are re-derived and checked recursively —
    so children pending at an interrupt are never lost on resume.
    """
    out: list[PlanSlice] = []
    stack: deque[PlanSlice] = deque(plan)
    while stack:
        s = stack.popleft()
        e = manifest.entry(s.key)
        if e is None or e.get("failed") is not None:
            out.append(s)
            continue
        if e.get("bisected"):
            children = bisect_slice(s)
            if children is not None:
                stack.appendleft(children[1])
                stack.appendleft(children[0])
    return out


# --- runner -------------------------------------------------------------------- #


class _CapReached(Exception):
    """Internal: a hard cap (requests / wall clock) was hit; stop cleanly."""

    def __init__(self, which: str) -> None:
        super().__init__(which)
        self.which = which


@dataclass
class Counters:
    requests_made: int = 0
    http_429: int = 0
    slices_ok: int = 0
    slices_failed: int = 0
    slices_truncated: int = 0
    slices_bisected: int = 0
    records_returned: int = 0
    records_written: int = 0

    def as_dict(self) -> dict[str, int]:
        return dict(vars(self))


@dataclass
class BackfillRunner:
    """Paced, capped, checkpointed executor for a slice plan.

    Everything nondeterministic is injectable (``fetcher``, ``sleep``,
    ``monotonic``, ``rng``, ``now_fn``) so tests run offline, instantly.
    """

    lake: SentimentLake
    manifest: Manifest
    fetcher: Fetcher
    max_requests: int
    max_wall_clock_minutes: int
    sleep: Callable[[float], None] = time_module.sleep
    monotonic: Callable[[], float] = time_module.monotonic
    rng: random.Random = field(default_factory=random.Random)
    now_fn: Callable[[], datetime] = lambda: datetime.now(UTC)
    counters: Counters = field(default_factory=Counters)
    _last_request_at: float | None = None
    _started_at: float = 0.0

    # -- pacing / caps -- #

    def _check_caps(self) -> None:
        if self.counters.requests_made >= self.max_requests:
            raise _CapReached("max_requests")
        if (self.monotonic() - self._started_at) > self.max_wall_clock_minutes * 60.0:
            raise _CapReached("max_wall_clock_minutes")

    def _pace(self) -> None:
        """Enforce >= REQUEST_INTERVAL_S (+ jitter) between consecutive requests."""
        if self._last_request_at is not None:
            target = REQUEST_INTERVAL_S + self.rng.uniform(0.0, JITTER_MAX_S)
            elapsed = self.monotonic() - self._last_request_at
            if elapsed < target:
                self.sleep(target - elapsed)
        self._last_request_at = self.monotonic()

    @staticmethod
    def _backoff_delay(attempt: int, retry_after: str | None) -> float:
        """Backoff before retry ``attempt`` (1-based): 10s doubling, capped at 120s,
        honoring a parseable Retry-After header (still capped)."""
        delay = min(BACKOFF_INITIAL_S * (2.0 ** (attempt - 1)), BACKOFF_MAX_S)
        if retry_after:
            # non-numeric Retry-After (an HTTP-date) keeps the exponential value
            with contextlib.suppress(ValueError):
                delay = max(delay, float(retry_after))
        return min(delay, BACKOFF_MAX_S)

    def _fetch_with_retries(
        self, s: PlanSlice
    ) -> tuple[list[RawNewsEvent] | None, str | None, int]:
        """Fetch one slice with pacing + bounded retries.

        Returns (events, failed_reason, http_status); raises :class:`_CapReached`
        when a hard cap is hit before/while retrying.
        """
        attempts = 0
        while True:
            self._check_caps()
            self._pace()
            attempts += 1
            self.counters.requests_made += 1
            try:
                return self.fetcher(s.topic, s.start, s.end), None, 200
            except HTTPError as exc:
                status = int(exc.code)
                if status == 429:
                    self.counters.http_429 += 1
                    if attempts >= MAX_RETRIES_PER_SLICE:
                        return None, "rate_limited", status
                    retry_after = exc.headers.get("Retry-After") if exc.headers else None
                    self.sleep(self._backoff_delay(attempts, retry_after))
                else:
                    if attempts >= MAX_RETRIES_PER_SLICE:
                        return None, f"http_{status}", status
                    self.sleep(self._backoff_delay(attempts, None))
            except Exception as exc:  # noqa: BLE001 - URLError/timeout/JSON; bounded retries
                if attempts >= MAX_RETRIES_PER_SLICE:
                    return None, f"error_{type(exc).__name__}", 0
                self.sleep(self._backoff_delay(attempts, None))

    # -- main loop -- #

    def run(self, plan: Sequence[PlanSlice], *, kind: str = "backfill") -> str:
        """Execute the pending part of ``plan``. Returns the outcome:
        ``completed`` | ``capped_max_requests`` | ``capped_max_wall_clock_minutes``."""
        self._started_at = self.monotonic()
        queue: deque[PlanSlice] = deque(pending_slices(plan, self.manifest))
        total = len(queue)
        run_idx = self.manifest.start_run(started_at=self.now_fn().isoformat(), kind=kind)
        outcome = "completed"
        done = 0
        try:
            while queue:
                s = queue.popleft()
                events, failed, status = self._fetch_with_retries(s)
                done += 1
                if failed is not None or events is None:
                    self.counters.slices_failed += 1
                    self.manifest.record_slice(
                        s,
                        http_status=status,
                        records_returned=0,
                        records_written=0,
                        truncated=False,
                        bisected=False,
                        failed=failed or "unknown",
                        completed_at=self.now_fn().isoformat(),
                    )
                    print(f"[backfill] {done}/{total} {s.key} FAILED: {failed}", flush=True)
                    continue
                written = self.lake.write_raw(events)
                truncated = len(events) >= MAX_RECORDS
                children = bisect_slice(s) if truncated else None
                if children is not None:
                    queue.appendleft(children[1])
                    queue.appendleft(children[0])
                    total += 2
                    self.counters.slices_bisected += 1
                if truncated:
                    self.counters.slices_truncated += 1
                self.counters.slices_ok += 1
                self.counters.records_returned += len(events)
                self.counters.records_written += written
                self.manifest.record_slice(
                    s,
                    http_status=status,
                    records_returned=len(events),
                    records_written=written,
                    truncated=truncated,
                    bisected=children is not None,
                    failed=None,
                    completed_at=self.now_fn().isoformat(),
                )
                note = (
                    " TRUNCATED->bisect"
                    if children is not None
                    else (" TRUNCATED (floor)" if truncated else "")
                )
                print(
                    f"[backfill] {done}/{total} {s.key} ok: {len(events)} records "
                    f"({written} new){note}",
                    flush=True,
                )
        except _CapReached as cap:
            outcome = f"capped_{cap.which}"
            print(f"[backfill] STOPPED: hard cap {cap.which} reached — checkpoint intact.")
        finally:
            self.manifest.end_run(
                run_idx,
                ended_at=self.now_fn().isoformat(),
                outcome=outcome,
                counters=self.counters.as_dict(),
            )
        return outcome


# --- CLI ------------------------------------------------------------------------ #


def default_manifest_path() -> Path:
    return Settings().data_dir / "sentiment_backfill" / "manifest.json"


def _make_fetcher(cfg: SentimentConfig) -> Fetcher:
    """The one real fetcher: the existing gdelt adapter, label->query via config."""
    queries = cfg.backfill.topic_queries

    def fetch(topic: str, start: datetime, end: datetime) -> list[RawNewsEvent]:
        return gdelt.fetch_artlist(
            topic=topic,
            start=start,
            end=end,
            max_records=MAX_RECORDS,
            sentiment_feature_version=cfg.sentiment_feature_version,
            ingested_at=datetime.now(UTC),
            allow_network=True,
            language=cfg.fetch_limits.default_language,
            query_text=queries.get(topic),
        )

    return fetch


def _print_plan(plan: Sequence[PlanSlice], topics: Sequence[str], now: datetime) -> None:
    per_topic = {t: sum(1 for s in plan if s.topic == t) for t in topics}
    n_sup = sum(1 for s in plan if s.archive_region == SUPPORTED)
    n_unsup = len(plan) - n_sup
    est_min = len(plan) * REQUEST_INTERVAL_S / 60.0
    print(f"[sentiment.backfill] PLAN slices={len(plan)} topics={len(topics)}")
    print(
        f"  archive: supported={n_sup} unsupported_archive={n_unsup} "
        f"(cutoff {archive_cutoff(now).date()} — supported region runs first)"
    )
    print(f"  per-topic slices: {per_topic}")
    print(
        f"  estimated minimum duration at {REQUEST_INTERVAL_S:.0f}s spacing: "
        f"~{est_min:.0f} min (before bisection/backoff)"
    )
    print("  network: NONE (plan only)")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sentiment.backfill", description=__doc__)
    p.add_argument("--start", help="YYYY-MM-DD inclusive (default: config windows.start)")
    p.add_argument("--end", help="YYYY-MM-DD inclusive (default: config windows.end)")
    p.add_argument("--topics", nargs="+", default=None, help="subset of config query_topics")
    p.add_argument("--plan", action="store_true", help="print the request plan; zero network")
    p.add_argument(
        "--probe",
        action="store_true",
        help="run exactly one slice (first topic, most recent UTC day) as a live canary",
    )
    p.add_argument("--allow-network", action="store_true", help="REQUIRED for any real fetch")
    p.add_argument("--resume", action="store_true", help="continue from an existing manifest")
    return p


def run(args: argparse.Namespace) -> int:
    cfg = SentimentConfig.load()
    topics: Sequence[str] = args.topics or list(cfg.query_topics)
    unknown = sorted(set(topics) - set(cfg.query_topics))
    if unknown:
        print(f"BLOCKED: unknown topic(s) {unknown}; config topics: {list(cfg.query_topics)}")
        return 2
    start = date.fromisoformat(args.start) if args.start else cfg.windows.start
    end = date.fromisoformat(args.end) if args.end else cfg.windows.end
    now = datetime.now(UTC)

    if args.probe:
        day = now.date() - timedelta(days=1)
        lo = datetime.combine(day, time(0), UTC)
        plan: list[PlanSlice] = [
            PlanSlice(
                topic=topics[0], start=lo, end=lo + timedelta(days=1), archive_region=SUPPORTED
            )
        ]
    else:
        plan = build_request_plan(topics, start, end, now=now)

    if args.plan:
        _print_plan(plan, topics, now)
        return 0

    # Real fetches from here. Fail-closed: free_no_auth + explicit --allow-network only.
    try:
        assert_network_allowed(gdelt.SOURCE, allow_network=args.allow_network)
    except Exception as exc:  # noqa: BLE001 - surface the refusal cleanly
        print(f"BLOCKED: {exc}")
        return 2

    manifest_path = default_manifest_path()
    if manifest_path.exists() and not (args.resume or args.probe):
        print(
            f"BLOCKED: manifest already exists at {manifest_path}. Pass --resume to "
            f"continue it (completed slices are skipped) or move it aside to start fresh."
        )
        return 2
    manifest = Manifest.load_or_create(
        manifest_path,
        meta={
            "created_at": now.isoformat(),
            "start": str(start),
            "end": str(end),
            "topics": list(topics),
            "sentiment_feature_version": cfg.sentiment_feature_version,
            "archive_cutoff": archive_cutoff(now).isoformat(),
            "archive_supported_days": ARCHIVE_SUPPORTED_DAYS,
        },
    )

    lake = SentimentLake(
        raw_dataset=cfg.storage.raw_dataset, scored_dataset=cfg.storage.scored_dataset
    )
    runner = BackfillRunner(
        lake=lake,
        manifest=manifest,
        fetcher=_make_fetcher(cfg),
        max_requests=cfg.backfill.max_requests,
        max_wall_clock_minutes=cfg.backfill.max_wall_clock_minutes,
    )
    kind = "probe" if args.probe else "backfill"
    print(
        f"[sentiment.backfill] {kind} starting: {len(plan)} planned slices, "
        f"caps: {cfg.backfill.max_requests} requests / {cfg.backfill.max_wall_clock_minutes} min. "
        f"manifest={manifest_path}"
    )
    outcome = runner.run(plan, kind=kind)
    summary = manifest.summarize()
    print(f"[sentiment.backfill] outcome={outcome} summary={json.dumps(summary)}")
    if args.probe:
        entry = manifest.entry(plan[0].key)
        if entry is None or entry.get("failed") is not None:
            return 1
        return 0
    return 0 if outcome == "completed" else 3


def main(argv: list[str] | None = None) -> int:
    return run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
