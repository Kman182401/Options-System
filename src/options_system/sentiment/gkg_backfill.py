"""Resumable GDELT GKG bulk-archive backfill (System A).

    uv run python -m options_system.sentiment.gkg_backfill \\
        --start 2019-01-01 --end 2026-06-16 \\
        [--plan] [--probe] [--allow-network] [--resume] [--workers N]

Single responsibility: turn a date range into a resumable, paced, hard-capped sequence
of 15-minute GKG file downloads, each parsed + theme-filtered (:mod:`gkg`) and written
idempotently into a SEPARATE lake (``sentiment_gkg_*``) via the existing
:class:`~options_system.sentiment.lake.SentimentLake`. GDELT's precomputed tone is the
score — no FinBERT, no DOC API, no per-query cap, no 1-req/5s throttle.

Why this exists: the DOC 2.0 ArtList path (``backfill.py``) is officially limited to a
~3-month archive and is rate-limited, so it can only ever cover a sliver of history. The
bulk file archive serves every 15-minute file back to 2015 as a plain download, which is
the only way to backfill years of coverage.

Design (mirrors ``backfill.py`` so the two read the same way):

* **Plan** — a pure function: every 15-minute UTC slot in ``[start, end]`` -> a file URL.
  Deterministic; archive gaps (a genuinely missing file) are recorded ``missing``, never
  treated as a failure.
* **Concurrency** — downloads (I/O-bound) run in a bounded thread pool; parsing, lake
  writes and the manifest stay on the main thread (single-writer, atomic). Tests inject
  ``workers=1`` for a fully deterministic serial path.
* **Resumability** — an atomic JSON manifest under ``data/sentiment_gkg/`` records every
  file attempt; ``ok``/``missing`` files are skipped on ``--resume``, ``failed`` retried.
* **Hard caps (fail-closed)** — ``max_files`` / ``max_wall_clock_minutes`` / ``max_bytes``
  from ``config/sentiment_gkg.yaml``; hitting any stops cleanly (exit 3) with the manifest
  intact.
* **Fail-closed gate** — a real fetch requires the ``free_no_auth`` policy for ``gdelt_gkg``
  **and** an explicit ``--allow-network``.

The clock, sleep and fetcher are injectable so every test runs offline, instantly.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import zipfile
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from config.settings import Settings
from options_system.common.external_data_policy import assert_network_allowed
from options_system.sentiment import gkg
from options_system.sentiment.gkg_config import GkgConfig
from options_system.sentiment.gkg_lake import GkgLake

# GDELT protocol facts (not tunables — changing these changes correctness):
GKG_BASE = "http://data.gdeltproject.org/gdeltv2"
SLOT = timedelta(minutes=15)  # GKG cadence: files at :00 :15 :30 :45
PROBE_LAG = timedelta(minutes=90)  # a slot this far back is reliably already published

OK = "ok"
MISSING = "missing"
FAILED = "failed"

#: fetcher signature: url -> (http_status, body_bytes_or_None). 404 returns (404, None)
#: (a genuinely absent file). Transient errors (5xx, timeouts) RAISE for bounded retry.
Fetcher = Callable[[str], tuple[int, bytes | None]]


# --- pure plan ---------------------------------------------------------------- #


def file_timestamp(ts: datetime) -> str:
    return ts.astimezone(UTC).strftime("%Y%m%d%H%M%S")


def file_url(ts: datetime) -> str:
    return f"{GKG_BASE}/{file_timestamp(ts)}.gkg.csv.zip"


def build_file_plan(start: date, end: date) -> list[datetime]:
    """Every 15-minute UTC slot in ``[start 00:00, end 23:45]`` (inclusive), ascending."""
    if end < start:
        raise ValueError(f"gkg backfill: end {end} precedes start {start}")
    lo = datetime.combine(start, time(0), UTC)
    hi = datetime.combine(end, time(0), UTC) + timedelta(days=1)  # exclusive next-midnight
    out: list[datetime] = []
    cur = lo
    while cur < hi:
        out.append(cur)
        cur += SLOT
    return out


def floor_to_slot(ts: datetime) -> datetime:
    ts = ts.astimezone(UTC)
    minute = (ts.minute // 15) * 15
    return ts.replace(minute=minute, second=0, microsecond=0)


# --- default network fetcher -------------------------------------------------- #


def default_fetcher(url: str, *, timeout: float = 60.0) -> tuple[int, bytes | None]:
    """GET ``url`` -> (status, bytes). 404 -> (404, None); other HTTP errors raise."""
    req = Request(url, headers={"User-Agent": "Options-System/research (GKG bulk backfill)"})
    try:
        with urlopen(req, timeout=timeout) as resp:  # noqa: S310 - http GDELT bulk archive
            return int(getattr(resp, "status", 200) or 200), resp.read()
    except HTTPError as exc:
        if int(exc.code) == 404:
            return 404, None
        raise


# --- manifest (checkpoint + audit trail) -------------------------------------- #


class GkgManifest:
    """Atomic JSON checkpoint under ``data/sentiment_gkg/`` (gitignored).

    One entry per attempted file timestamp: status, http code, rows/kept/malformed,
    bytes, completed_at. Atomic writes (tmp + rename) so an interrupt never corrupts it.
    """

    def __init__(self, path: Path, data: dict[str, Any]) -> None:
        self.path = path
        self.data = data

    @classmethod
    def load_or_create(cls, path: Path, *, meta: dict[str, Any]) -> GkgManifest:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            data.setdefault("files", {})
            data.setdefault("runs", [])
            return cls(path, data)
        return cls(path, {"meta": meta, "files": {}, "runs": []})

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.data, indent=1, default=str), encoding="utf-8")
        os.replace(tmp, self.path)

    def is_done(self, key: str) -> bool:
        """Done == attempted and not failed. ``missing`` counts as done (no such file)."""
        e = self.data["files"].get(key)
        return e is not None and e.get("status") != FAILED

    def record(self, key: str, **fields: Any) -> None:
        self.data["files"][key] = fields

    def start_run(self, *, started_at: str) -> int:
        self.data["runs"].append({"started_at": started_at, "outcome": "running"})
        self.save()
        return len(self.data["runs"]) - 1

    def end_run(self, idx: int, *, ended_at: str, outcome: str, counters: dict[str, int]) -> None:
        self.data["runs"][idx].update(
            {"ended_at": ended_at, "outcome": outcome, "counters": counters}
        )
        self.save()

    def summarize(self) -> dict[str, Any]:
        files = self.data["files"].values()
        by_status: dict[str, int] = {OK: 0, MISSING: 0, FAILED: 0}
        for e in files:
            st = e.get("status", FAILED)
            by_status[st] = by_status.get(st, 0) + 1
        return {
            "files_attempted": len(self.data["files"]),
            "files_by_status": by_status,
            "rows_seen": sum(int(e.get("n_rows") or 0) for e in files),
            "events_kept": sum(int(e.get("n_kept") or 0) for e in files),
            "rows_malformed": sum(int(e.get("n_malformed") or 0) for e in files),
            "records_written": sum(int(e.get("records_written") or 0) for e in files),
            "bytes_downloaded": sum(int(e.get("n_bytes") or 0) for e in files),
            "runs": len(self.data["runs"]),
        }


# --- runner ------------------------------------------------------------------- #


class _CapReached(Exception):
    def __init__(self, which: str) -> None:
        super().__init__(which)
        self.which = which


@dataclass
class Counters:
    files_attempted: int = 0
    files_ok: int = 0
    files_missing: int = 0
    files_failed: int = 0
    bytes_downloaded: int = 0
    events_kept: int = 0
    records_written: int = 0

    def as_dict(self) -> dict[str, int]:
        return dict(vars(self))


@dataclass(frozen=True)
class FileResult:
    """Outcome of fetching one file (never raises out of the worker)."""

    ts: datetime
    status: str  # OK | MISSING | FAILED
    http_status: int
    data: bytes | None
    error: str | None
    n_bytes: int


@dataclass
class GkgBackfillRunner:
    """Paced, capped, checkpointed executor for a GKG file plan.

    Everything nondeterministic is injectable (``fetcher``, ``sleep``, ``monotonic``,
    ``now_fn``) so tests run offline, instantly, with ``workers=1`` (serial).
    """

    cfg: GkgConfig
    lake: GkgLake
    manifest: GkgManifest
    fetcher: Fetcher
    max_files: int
    max_wall_clock_minutes: int
    max_bytes: int
    workers: int
    politeness_delay_s: float
    retry_max: int
    sleep: Callable[[float], None]
    monotonic: Callable[[], float]
    now_fn: Callable[[], datetime] = lambda: datetime.now(UTC)
    counters: Counters = field(default_factory=Counters)
    _started_at: float = 0.0

    # -- caps -- #

    def _check_caps(self) -> None:
        if self.counters.files_attempted >= self.max_files:
            raise _CapReached("max_files")
        if self.counters.bytes_downloaded >= self.max_bytes:
            raise _CapReached("max_bytes")
        if (self.monotonic() - self._started_at) > self.max_wall_clock_minutes * 60.0:
            raise _CapReached("max_wall_clock_minutes")

    # -- per-file fetch with bounded retry (runs in worker threads) -- #

    def _fetch_one(self, ts: datetime) -> FileResult:
        url = file_url(ts)
        attempts = 0
        while True:
            attempts += 1
            try:
                status, data = self.fetcher(url)
            except HTTPError as exc:  # 5xx etc (404 is returned, not raised)
                if attempts > self.retry_max:
                    return FileResult(ts, FAILED, int(exc.code), None, f"http_{exc.code}", 0)
                self.sleep(min(2.0 ** (attempts - 1), 30.0))
                continue
            except Exception as exc:  # noqa: BLE001 - URLError/timeout/etc; bounded retry
                if attempts > self.retry_max:
                    return FileResult(ts, FAILED, 0, None, f"error_{type(exc).__name__}", 0)
                self.sleep(min(2.0 ** (attempts - 1), 30.0))
                continue
            if status == 404 or data is None:
                return FileResult(ts, MISSING, 404, None, None, 0)
            return FileResult(ts, OK, status, data, None, len(data))

    def _download_chunk(self, chunk: Sequence[datetime]) -> list[FileResult]:
        if self.workers <= 1:
            return [self._fetch_one(ts) for ts in chunk]
        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            return list(ex.map(self._fetch_one, chunk))

    # -- parse + write one downloaded file (main thread) -- #

    def _process(self, res: FileResult) -> None:
        key = file_timestamp(res.ts)
        self.counters.files_attempted += 1
        self.counters.bytes_downloaded += res.n_bytes
        completed = self.now_fn().isoformat()

        if res.status == MISSING:
            self.counters.files_missing += 1
            self.manifest.record(key, status=MISSING, http_status=404, completed_at=completed)
            return
        if res.status == FAILED or res.data is None:
            self.counters.files_failed += 1
            self.manifest.record(
                key,
                status=FAILED,
                http_status=res.http_status,
                error=res.error,
                completed_at=completed,
            )
            return

        try:
            with zipfile.ZipFile(io.BytesIO(res.data)) as zf:
                text = zf.read(zf.namelist()[0]).decode("utf-8", "replace")
        except (zipfile.BadZipFile, IndexError, OSError) as exc:
            self.counters.files_failed += 1
            self.manifest.record(
                key,
                status=FAILED,
                http_status=res.http_status,
                error=f"unzip_{type(exc).__name__}",
                n_bytes=res.n_bytes,
                completed_at=completed,
            )
            return

        parsed = gkg.parse_gkg_file(
            text,
            theme_prefixes=self.cfg.theme_prefixes,
            query_topic=self.cfg.query_topic,
            event_version=self.cfg.gkg_event_version,
            tone_model_name=self.cfg.tone_model_name,
            ingested_at=self.now_fn(),
        )
        written = self.lake.write_file(res.ts, parsed.raw, parsed.scored)
        self.counters.files_ok += 1
        self.counters.events_kept += parsed.n_kept
        self.counters.records_written += written
        self.manifest.record(
            key,
            status=OK,
            http_status=res.http_status,
            n_rows=parsed.n_rows,
            n_kept=parsed.n_kept,
            n_malformed=parsed.n_malformed,
            records_written=written,
            n_bytes=res.n_bytes,
            completed_at=completed,
        )

    # -- main loop -- #

    def run(self, plan: Sequence[datetime]) -> str:
        """Execute the pending part of ``plan``. Returns ``completed`` or ``capped_<which>``."""
        self._started_at = self.monotonic()
        pending = [ts for ts in plan if not self.manifest.is_done(file_timestamp(ts))]
        run_idx = self.manifest.start_run(started_at=self.now_fn().isoformat())
        outcome = "completed"
        chunk_size = max(1, self.workers * 4)
        done = 0
        total = len(pending)
        idx = 0
        try:
            while idx < total:
                self._check_caps()
                # Bound the chunk to the remaining FILE budget so concurrent prefetch can
                # never download beyond max_files (the cap is exact, not chunk-granular).
                remaining = self.max_files - self.counters.files_attempted
                this_chunk = min(chunk_size, remaining, total - idx)
                if this_chunk <= 0:
                    raise _CapReached("max_files")
                if self.politeness_delay_s > 0:
                    self.sleep(self.politeness_delay_s)
                chunk = pending[idx : idx + this_chunk]
                idx += this_chunk
                results = self._download_chunk(chunk)
                for res in results:
                    self._process(res)
                    done += 1
                    self._check_caps()
                self.manifest.save()
                print(
                    f"[gkg.backfill] {done}/{total} files "
                    f"(ok={self.counters.files_ok} missing={self.counters.files_missing} "
                    f"failed={self.counters.files_failed}) "
                    f"kept={self.counters.events_kept} "
                    f"~{self.counters.bytes_downloaded / 1e9:.2f}GB",
                    flush=True,
                )
        except _CapReached as cap:
            outcome = f"capped_{cap.which}"
            print(f"[gkg.backfill] STOPPED: hard cap {cap.which} reached — checkpoint intact.")
        finally:
            self.manifest.save()
            self.manifest.end_run(
                run_idx,
                ended_at=self.now_fn().isoformat(),
                outcome=outcome,
                counters=self.counters.as_dict(),
            )
        return outcome


# --- CLI ---------------------------------------------------------------------- #


def default_manifest_path() -> Path:
    return Settings().data_dir / "sentiment_gkg" / "manifest.json"


def _current_fingerprint(cfg: GkgConfig, start: date, end: date) -> dict[str, Any]:
    """The config/run identity stamped in the manifest. ``--resume`` refuses a manifest
    whose stored fingerprint disagrees, so a window/theme/version change can never quietly
    skip files that were processed under a different configuration."""
    return {
        "start": str(start),
        "end": str(end),
        "gkg_event_version": cfg.gkg_event_version,
        "tone_model_name": cfg.tone_model_name,
        "theme_prefixes": list(cfg.theme_prefixes),
        "query_topic": cfg.query_topic,
        "raw_dataset": cfg.storage.raw_dataset,
        "scored_dataset": cfg.storage.scored_dataset,
    }


def _make_runner(cfg: GkgConfig, *, workers: int, manifest: GkgManifest) -> GkgBackfillRunner:
    import time as _time

    lake = GkgLake(raw_dataset=cfg.storage.raw_dataset, scored_dataset=cfg.storage.scored_dataset)
    return GkgBackfillRunner(
        cfg=cfg,
        lake=lake,
        manifest=manifest,
        fetcher=default_fetcher,
        max_files=cfg.backfill.max_files,
        max_wall_clock_minutes=cfg.backfill.max_wall_clock_minutes,
        max_bytes=cfg.backfill.max_bytes,
        workers=workers,
        politeness_delay_s=cfg.backfill.politeness_delay_s,
        retry_max=cfg.backfill.retry_max,
        sleep=_time.sleep,
        monotonic=_time.monotonic,
    )


def _print_plan(plan: Sequence[datetime]) -> None:
    n = len(plan)
    est_gb = n * 5.0 / 1024.0  # ~5 MB/file
    print(f"[gkg.backfill] PLAN files={n} (15-min slots)")
    if plan:
        print(f"  range: {plan[0].isoformat()} .. {plan[-1].isoformat()}")
    print(f"  estimated transfer: ~{est_gb:.0f} GB (filtered store is a small fraction)")
    print("  network: NONE (plan only)")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sentiment.gkg_backfill", description=__doc__)
    p.add_argument("--start", help="YYYY-MM-DD inclusive (default: config window.start)")
    p.add_argument("--end", help="YYYY-MM-DD inclusive (default: config window.end)")
    p.add_argument("--plan", action="store_true", help="print the file plan; zero network")
    p.add_argument(
        "--probe",
        action="store_true",
        help="fetch exactly one recent file (~90 min ago) as a live canary",
    )
    p.add_argument("--allow-network", action="store_true", help="REQUIRED for any real fetch")
    p.add_argument("--resume", action="store_true", help="continue from an existing manifest")
    p.add_argument("--workers", type=int, default=None, help="override config download workers")
    return p


def run(args: argparse.Namespace) -> int:
    cfg = GkgConfig.load()
    start = date.fromisoformat(args.start) if args.start else cfg.window.start
    end = date.fromisoformat(args.end) if args.end else cfg.window.end
    workers = args.workers if args.workers is not None else cfg.backfill.workers

    if args.probe:
        plan = [floor_to_slot(datetime.now(UTC) - PROBE_LAG)]
    else:
        plan = build_file_plan(start, end)

    if args.plan:
        _print_plan(plan)
        return 0

    # Real fetches from here. Fail-closed: free_no_auth + explicit --allow-network only.
    try:
        assert_network_allowed(GkgConfig.SOURCE, allow_network=args.allow_network)
    except Exception as exc:  # noqa: BLE001 - surface the refusal cleanly
        print(f"BLOCKED: {exc}")
        return 2

    now = datetime.now(UTC)
    manifest_path = default_manifest_path()
    existed = manifest_path.exists()
    if existed and not (args.resume or args.probe):
        print(
            f"BLOCKED: manifest already exists at {manifest_path}. Pass --resume to continue "
            f"(ok/missing files are skipped) or move it aside to start fresh."
        )
        return 2
    fingerprint = _current_fingerprint(cfg, start, end)
    if existed and args.resume and not args.probe:
        try:
            stored = json.loads(manifest_path.read_text(encoding="utf-8")).get("meta", {})
        except (OSError, json.JSONDecodeError):
            stored = {}
        # Compare only fields present in the stored meta (older manifests carry a subset),
        # so a legitimate resume never false-trips while a real config drift is caught.
        mismatched = {
            k: {"manifest": stored[k], "current": v}
            for k, v in fingerprint.items()
            if k in stored and stored[k] != v
        }
        if mismatched:
            print(
                f"BLOCKED: --resume manifest at {manifest_path} was built with a different "
                f"configuration {mismatched}. Move it aside to start a fresh backfill (or restore "
                f"the matching window/theme/version)."
            )
            return 2
    manifest = GkgManifest.load_or_create(
        manifest_path, meta={"created_at": now.isoformat(), **fingerprint}
    )
    runner = _make_runner(cfg, workers=max(1, workers), manifest=manifest)
    kind = "probe" if args.probe else "backfill"
    print(
        f"[gkg.backfill] {kind} starting: {len(plan)} files, workers={runner.workers}, "
        f"caps: {cfg.backfill.max_files} files / {cfg.backfill.max_wall_clock_minutes} min / "
        f"{cfg.backfill.max_bytes / 1e9:.0f} GB. manifest={manifest_path}"
    )
    outcome = runner.run(plan)
    print(f"[gkg.backfill] outcome={outcome} summary={json.dumps(manifest.summarize())}")
    if args.probe:
        entry = manifest.data["files"].get(file_timestamp(plan[0]))
        return 0 if (entry is not None and entry.get("status") in (OK, MISSING)) else 1
    return 0 if outcome == "completed" else 3


def main(argv: list[str] | None = None) -> int:
    return run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
