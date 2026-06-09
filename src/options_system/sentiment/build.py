"""Sentiment build CLI — fixture/local-only by default; network is opt-in and gated.

    uv run python -m options_system.sentiment.build [flags]

Flags
-----
--source gdelt|sec_edgar|fixture   which adapter (``fixture`` = load a pre-shaped file)
--start / --end  YYYY-MM-DD        bounds (default: config window)
--topic TOPIC                      query topic label (default: first config topic)
--max-records N                    hard cap on records (default: config fetch_limits)
--allow-network                    REQUIRED for any real fetch; only works for a
                                   free_no_auth source. Without it, runs offline.
--score                            score parsed events with the deterministic FakeScorer
--dry-run                          parse/plan only; never write, never network
--fixture PATH                     parse this local fixture file (offline)

Safety: the access decision is a pure function (:func:`decide_access`) that fails
closed — paid/unknown sources are refused outright, and a network fetch is only
authorized when the source is ``free_no_auth`` **and** ``--allow-network`` is passed.
The default does nothing over the network.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, date, datetime, time
from pathlib import Path

from options_system.common.external_data_policy import (
    SourcePolicy,
    assert_network_allowed,
    assert_source_usable,
)
from options_system.sentiment import gdelt, sec_edgar
from options_system.sentiment.config import SentimentConfig
from options_system.sentiment.lake import SentimentLake
from options_system.sentiment.schema import (
    RawNewsEvent,
    ScoredNewsEvent,
    dedupe_by_hash,
)
from options_system.sentiment.scoring import FakeScorer, Scorer
from options_system.sentiment.sources import FIXTURE_SOURCE


@dataclass(frozen=True)
class AccessPlan:
    """Resolved, fail-closed decision about how a build may access its source."""

    mode: str  # "fixture" | "dry_run" | "network"
    network: bool
    source: str


def decide_access(
    source: str,
    *,
    allow_network: bool,
    fixture: str | None,
    dry_run: bool,
) -> AccessPlan:
    """Decide (purely) how this build may reach its source. Fails closed.

    * paid/unknown source -> raises (refused outright).
    * fixture given (or ``source == 'fixture'``) -> offline parse, no network.
    * ``local_only`` source -> dry-run (nothing to fetch over the network).
    * ``free_no_auth`` source -> network ONLY with ``allow_network=True``; otherwise
      a default dry-run (no network).
    """
    src = source.strip().lower()
    if src == FIXTURE_SOURCE or fixture is not None:
        if src != FIXTURE_SOURCE:
            assert_source_usable(src)  # refuse paid/unknown even for offline parse
        return AccessPlan(mode="fixture", network=False, source=src)

    policy = assert_source_usable(src)  # raises for paid_blocked / unknown_blocked
    if policy is SourcePolicy.LOCAL_ONLY:
        return AccessPlan(mode="dry_run", network=False, source=src)
    # free_no_auth from here
    if allow_network and not dry_run:
        assert_network_allowed(src, allow_network=True)  # belt-and-suspenders
        return AccessPlan(mode="network", network=True, source=src)
    return AccessPlan(mode="dry_run", network=False, source=src)


def _ingested_now() -> datetime:
    return datetime.now(UTC)


def parse_fixture(
    path: str,
    *,
    source: str,
    topic: str,
    sentiment_feature_version: str,
    ingested_at: datetime,
) -> list[RawNewsEvent]:
    """Parse a local fixture file into raw events (offline)."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    src = source.strip().lower()
    if src == "gdelt" or (isinstance(payload, dict) and "articles" in payload):
        return gdelt.parse_artlist(
            payload,
            topic=topic,
            sentiment_feature_version=sentiment_feature_version,
            ingested_at=ingested_at,
        )
    if src == "sec_edgar" or (isinstance(payload, dict) and "filings" in payload):
        return sec_edgar.parse_submissions(
            payload,
            topic=topic,
            sentiment_feature_version=sentiment_feature_version,
            ingested_at=ingested_at,
        )
    # Generic: a list of pre-shaped RawNewsEvent dicts.
    if isinstance(payload, list):
        return [RawNewsEvent.model_validate(d) for d in payload]
    raise ValueError(
        f"unrecognised fixture shape for source {source!r}: expected GDELT 'articles', "
        f"SEC 'filings', or a list of raw events."
    )


def _score_events(events: list[RawNewsEvent], scorer: Scorer) -> list[ScoredNewsEvent]:
    out: list[ScoredNewsEvent] = []
    for ev in events:
        if ev.degraded:
            continue
        score = scorer.score_text(ev.snippet_or_text or ev.title)
        out.append(
            ScoredNewsEvent(
                content_hash=ev.content_hash,
                source=ev.source,
                query_topic=ev.query_topic,
                published_at=ev.published_at,
                observed_at=ev.observed_at,
                sentiment_feature_version=ev.sentiment_feature_version,
                score=score,
            )
        )
    return out


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sentiment.build", description=__doc__)
    p.add_argument("--source", default=FIXTURE_SOURCE, help="gdelt | sec_edgar | fixture")
    p.add_argument("--start", help="YYYY-MM-DD (default: config window.start)")
    p.add_argument("--end", help="YYYY-MM-DD (default: config window.end)")
    p.add_argument("--topic", help="query topic label (default: first config topic)")
    p.add_argument("--max-records", type=int, default=None, help="hard cap on records")
    p.add_argument("--allow-network", action="store_true", help="permit a real fetch (gated)")
    p.add_argument("--score", action="store_true", help="score events with the FakeScorer")
    p.add_argument("--dry-run", action="store_true", help="parse/plan only; never write/network")
    p.add_argument("--fixture", default=None, help="parse this local fixture file (offline)")
    return p


def run(args: argparse.Namespace) -> int:
    cfg = SentimentConfig.load()
    topic = args.topic or cfg.query_topics[0]
    start = date.fromisoformat(args.start) if args.start else cfg.windows.start
    end = date.fromisoformat(args.end) if args.end else cfg.windows.end
    max_records = args.max_records if args.max_records is not None else cfg.fetch_limits.max_records
    sfv = cfg.sentiment_feature_version

    try:
        plan = decide_access(
            args.source,
            allow_network=args.allow_network,
            fixture=args.fixture,
            dry_run=args.dry_run,
        )
    except Exception as exc:  # noqa: BLE001 - surface the refusal cleanly
        print(f"BLOCKED: {exc}")
        return 2

    print(
        f"[sentiment.build] source={plan.source} mode={plan.mode} network={plan.network} "
        f"topic={topic!r} window={start}..{end} max_records={max_records} sfv={sfv}"
    )

    if plan.mode == "network":
        # Real bounded fetch — only reachable with --allow-network on a free source.
        lo = datetime.combine(start, time(0), UTC)
        hi = datetime.combine(end, time(0), UTC)
        ingested = _ingested_now()
        if plan.source == "gdelt":
            events = gdelt.fetch_artlist(
                topic=topic,
                start=lo,
                end=hi,
                max_records=max_records,
                sentiment_feature_version=sfv,
                ingested_at=ingested,
                allow_network=True,
                language=cfg.fetch_limits.default_language,
            )
        else:  # sec_edgar real fetch needs a CIK; out of scope for this scaffold
            print("BLOCKED: real sec_edgar fetch needs a CIK and is out of scaffold scope.")
            return 2
        return _finish(events, args, cfg, write=True)

    if plan.mode == "dry_run" and args.fixture is None:
        # Nothing to parse offline: show the bounded request that WOULD be made, no call.
        if plan.source == "gdelt":
            lo = datetime.combine(start, time(0), UTC)
            hi = datetime.combine(end, time(0), UTC)
            url = gdelt.build_query_url(
                topic=topic,
                start=lo,
                end=hi,
                max_records=max_records,
                language=cfg.fetch_limits.default_language,
            )
            print(f"  would fetch (NOT executed): {url}")
        print("  dry run — no network, no writes. Pass --fixture for offline parse.")
        return 0

    # fixture / dry-run-with-fixture path
    ingested = _ingested_now()
    if args.fixture is None:
        print("BLOCKED: --fixture PATH is required for an offline parse of this source.")
        return 2
    events = parse_fixture(
        args.fixture,
        source=plan.source,
        topic=topic,
        sentiment_feature_version=sfv,
        ingested_at=ingested,
    )
    write = not args.dry_run
    return _finish(events, args, cfg, write=write)


def _finish(
    events: list[RawNewsEvent], args: argparse.Namespace, cfg: SentimentConfig, *, write: bool
) -> int:
    events = dedupe_by_hash(events)
    n_degraded = sum(1 for e in events if e.degraded)
    print(f"  parsed {len(events)} unique events ({n_degraded} degraded)")
    scored: list[ScoredNewsEvent] = []
    if args.score:
        scored = _score_events(events, FakeScorer())
        print(f"  scored {len(scored)} events with FakeScorer")
    if write:
        lake = SentimentLake(
            raw_dataset=cfg.storage.raw_dataset, scored_dataset=cfg.storage.scored_dataset
        )
        n_raw = lake.write_raw(events)
        print(f"  wrote {n_raw} new raw rows to {cfg.storage.raw_dataset}")
        if scored:
            n_sc = lake.write_scored(scored)
            print(f"  wrote {n_sc} new scored rows to {cfg.storage.scored_dataset}")
    else:
        print("  dry run — not written.")
    return 0


def main(argv: list[str] | None = None) -> int:
    return run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
