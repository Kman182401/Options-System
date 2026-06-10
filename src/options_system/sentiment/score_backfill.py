"""Batch-score unscored raw sentiment events with the LOCAL FinBERT (Phase 18).

    uv run python -m options_system.sentiment.score_backfill \\
        [--batch-size 64] [--limit N] [--fake] [--dry-run]

Reads non-degraded raw events from the lake, selects the ones not yet scored by the
target model, scores them in deterministic order, and writes the scores back
**idempotently on ``(content_hash, model_name)``** — a re-run writes nothing new.

Local-only by design: the scorer loads weights with ``local_files_only=True`` and
**never downloads anything**. If the weights are absent it fails with instructions
(one-time explicit download: ``uv run python scripts/download_finbert.py``). The
``model_version_or_hash`` stamped on every score is the resolved local snapshot
revision (git commit hash), so a future re-score with different weights is
distinguishable in the lake.

``--fake`` swaps in the deterministic :class:`FakeScorer` (tests / no-weights runs);
``--dry-run`` selects and reports but never scores or writes.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from options_system.sentiment.config import SentimentConfig
from options_system.sentiment.lake import SentimentLake
from options_system.sentiment.schema import ScoredNewsEvent, SentimentScore
from options_system.sentiment.scoring import (
    FakeScorer,
    FinbertScorer,
    FinbertWeightsMissing,
    Scorer,
)


def select_unscored(raw: pl.DataFrame, scored: pl.DataFrame, model_name: str) -> pl.DataFrame:
    """Non-degraded raw rows not yet scored by ``model_name``, deterministically ordered.

    Pure. Idempotency key is ``(content_hash, model_name)``: a hash already scored by
    this model is excluded; the same hash scored by a *different* model is not.
    """
    candidates = raw.filter(~pl.col("degraded"))
    if scored.height:
        done = (
            scored.filter(pl.col("model_name") == model_name)
            .get_column("content_hash")
            .unique()
            .to_list()
        )
        candidates = candidates.filter(~pl.col("content_hash").is_in(done))
    return candidates.sort(["observed_at", "content_hash"])


def score_frame(
    unscored: pl.DataFrame, scorer: Scorer, *, batch_size: int = 64
) -> list[ScoredNewsEvent]:
    """Score a selected raw frame -> scored events (input order preserved).

    Uses the scorer's batched path when it has one (FinBERT); otherwise scores one
    text at a time (FakeScorer). The scored text is ``snippet_or_text`` falling back
    to ``title`` — the same convention as the build CLI.
    """
    if unscored.height == 0:
        return []
    texts = [
        (snippet or title)
        for snippet, title in zip(
            unscored["snippet_or_text"].to_list(), unscored["title"].to_list(), strict=True
        )
    ]
    batch = getattr(scorer, "score_batch", None)
    scores: list[SentimentScore] = (
        batch(texts, batch_size=batch_size)
        if callable(batch)
        else [scorer.score_text(t) for t in texts]
    )
    out: list[ScoredNewsEvent] = []
    rows = unscored.select(
        "content_hash",
        "source",
        "query_topic",
        "published_at",
        "observed_at",
        "sentiment_feature_version",
    ).iter_rows(named=True)
    for row, score in zip(rows, scores, strict=True):
        out.append(
            ScoredNewsEvent(
                content_hash=row["content_hash"],
                source=row["source"],
                query_topic=row["query_topic"],
                published_at=row["published_at"],
                observed_at=row["observed_at"],
                sentiment_feature_version=row["sentiment_feature_version"],
                score=score,
            )
        )
    return out


def resolve_local_revision(model_name: str) -> str | None:
    """The locally-cached snapshot's git revision hash (no network; None if absent)."""
    try:
        from huggingface_hub import snapshot_download

        path = snapshot_download(model_name, local_files_only=True)
    except Exception:  # noqa: BLE001 - not cached / hub unavailable -> unknown revision
        return None
    return Path(path).name


def _pick_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001 - torch missing/broken -> let the scorer report it
        return "cpu"


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sentiment.score_backfill", description=__doc__)
    p.add_argument("--batch-size", type=int, default=64, help="texts per inference batch")
    p.add_argument("--limit", type=int, default=None, help="score at most N rows (debug)")
    p.add_argument("--fake", action="store_true", help="use the deterministic FakeScorer")
    p.add_argument("--dry-run", action="store_true", help="select + report only; never write")
    return p


def run(args: argparse.Namespace) -> int:
    cfg = SentimentConfig.load()
    lake = SentimentLake(
        raw_dataset=cfg.storage.raw_dataset, scored_dataset=cfg.storage.scored_dataset
    )
    raw = lake.read_raw()
    scored_existing = lake.read_scored()

    if args.fake:
        scorer: Scorer = FakeScorer()
        model_name = f"{FakeScorer.name}-{FakeScorer.version}"
        device = "cpu"
        revision: str | None = FakeScorer.version
    else:
        device = _pick_device()
        revision = resolve_local_revision(cfg.scoring.model)
        scorer = FinbertScorer(cfg.scoring.model, device=device, version_hash=revision)
        model_name = cfg.scoring.model

    unscored = select_unscored(raw, scored_existing, model_name)
    if args.limit is not None:
        unscored = unscored.head(args.limit)
    print(
        f"[sentiment.score_backfill] model={model_name} device={device} "
        f"revision={revision} raw_rows={raw.height} already_scored="
        f"{scored_existing.height} to_score={unscored.height} batch_size={args.batch_size}"
    )
    if args.dry_run:
        print("  dry run — nothing scored, nothing written.")
        return 0
    if unscored.height == 0:
        print("  nothing to score (idempotent re-run).")
        return 0

    t0 = time.monotonic()
    try:
        scored_events = score_frame(unscored, scorer, batch_size=args.batch_size)
    except FinbertWeightsMissing as exc:
        print(f"BLOCKED: {exc}")
        return 2
    elapsed = time.monotonic() - t0
    written = lake.write_scored(scored_events)
    rate = len(scored_events) / elapsed if elapsed > 0 else float("inf")
    print(
        f"  scored {len(scored_events)} rows in {elapsed:.1f}s ({rate:.1f} rows/s); "
        f"wrote {written} new scored rows (idempotent on (content_hash, model_name))"
    )

    # Health summary over the full lake state after the write (existing observability).
    from options_system.observability.sentiment_health import gather_sentiment_health

    health = gather_sentiment_health(
        lake.read_raw(),
        lake.read_scored(),
        source_policy={k: v.value for k, v in cfg.source_policy.items()},
        network_used=False,
    )
    print(f"  health={json.dumps(health, default=str)}")
    print(f"  finished_at={datetime.now(UTC).isoformat()}")
    return 0


def main(argv: list[str] | None = None) -> int:
    return run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
