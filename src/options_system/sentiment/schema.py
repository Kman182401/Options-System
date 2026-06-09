"""Point-in-time-correct schema for raw news/text events and dedup/PIT helpers.

The sentiment layer's defining risk is **look-ahead leakage**: scoring a headline
into a feature that a model then "sees" at a timestamp *before our system could have
known the headline existed*. This schema makes that impossible to do by accident by
forcing every event to carry three distinct timestamps and validating their order:

* ``published_at``  — when the *source* says the item was published.
* ``observed_at``   — the earliest moment our system could first have known it. For
  live capture this is the fetch time; for a historical backfill it must be set
  **conservatively** from source metadata (e.g. GDELT's first-seen ``seendate``, or
  an SEC filing's public acceptance datetime) and never earlier than the source
  supports. Feature generation may only use an event where ``observed_at <= t`` for
  the label/event time ``t``.
* ``ingested_at``   — when *we* stored it.

Invariant enforced here: ``published_at <= observed_at <= ingested_at``. You cannot
observe something before it was published, nor store it before observing it.

Deduplication is by ``content_hash`` (a stable hash of the identifying fields) or a
stable ``source_id`` — re-ingesting the same item never creates a duplicate feature.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Fields that define an item's identity for content-hash dedup. Two events with the
# same source, source_id, title, body and publish time are the same item.
_HASH_FIELDS = ("source", "source_id", "title", "snippet_or_text", "published_at")


def compute_content_hash(
    source: str,
    source_id: str,
    title: str,
    snippet_or_text: str,
    published_at: datetime,
) -> str:
    """Stable SHA-256 hex digest identifying one news/text item (order-fixed)."""
    parts = [
        source.strip().lower(),
        source_id.strip(),
        title.strip(),
        snippet_or_text.strip(),
        # Canonicalise to UTC so the digest is host-timezone-independent (a naive
        # datetime is treated as UTC). Without this, the same item hashes differently
        # on hosts in different timezones and dedup silently fails.
        (
            published_at.astimezone(UTC)
            if published_at.tzinfo
            else published_at.replace(tzinfo=UTC)
        ).isoformat(),
    ]
    joined = "\x1f".join(parts)  # unit-separator so fields can't collide on join
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


class RawNewsEvent(BaseModel):
    """One raw, unscored news/text item with point-in-time-correct timestamps."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source: str  # e.g. "gdelt", "sec_edgar"
    source_id: str  # stable id from the source (url, accession number, doc id)
    source_url: str | None = None
    title: str
    snippet_or_text: str = ""
    published_at: datetime
    observed_at: datetime
    ingested_at: datetime
    query_topic: str
    language: str | None = None
    entities: tuple[str, ...] = ()
    content_hash: str = ""  # filled in validation if not supplied
    sentiment_feature_version: str
    # Degraded path: set when the adapter could not fully parse an item but we still
    # want to record that we tried (e.g. a malformed record). ``degraded`` rows must
    # be excluded from feature generation; ``error`` carries the reason.
    degraded: bool = False
    error: str | None = None

    @model_validator(mode="after")
    def _fill_hash_and_check_pit(self) -> RawNewsEvent:
        # Canonicalise every timestamp to UTC (a naive datetime is assumed UTC) so the
        # point-in-time comparisons below and the content hash never depend on the host
        # timezone, and a naive/aware mix can't raise on comparison.
        for field in ("published_at", "observed_at", "ingested_at"):
            dt: datetime = getattr(self, field)
            norm = dt.astimezone(UTC) if dt.tzinfo else dt.replace(tzinfo=UTC)
            object.__setattr__(self, field, norm)
        # Fill the content hash deterministically if the adapter did not.
        if not self.content_hash:
            object.__setattr__(
                self,
                "content_hash",
                compute_content_hash(
                    self.source,
                    self.source_id,
                    self.title,
                    self.snippet_or_text,
                    self.published_at,
                ),
            )
        # Point-in-time order: cannot observe before publish, nor store before observe.
        if self.observed_at < self.published_at:
            raise ValueError(
                f"observed_at ({self.observed_at}) is before published_at ({self.published_at}): "
                "a backfill must not claim to have observed an item earlier than the source "
                "metadata supports."
            )
        if self.ingested_at < self.observed_at:
            raise ValueError(
                f"ingested_at ({self.ingested_at}) is before observed_at ({self.observed_at})."
            )
        return self


class SentimentScore(BaseModel):
    """The output of scoring one item's text (model-agnostic)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    positive_score: float = Field(ge=0.0, le=1.0)
    negative_score: float = Field(ge=0.0, le=1.0)
    neutral_score: float = Field(ge=0.0, le=1.0)
    sentiment_score: float  # positive_score - negative_score, in [-1, 1]
    model_name: str
    model_version_or_hash: str | None = None
    scored_at: datetime


class ScoredNewsEvent(BaseModel):
    """A raw event joined to its sentiment score (the scored-lake row)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    content_hash: str
    source: str
    query_topic: str
    published_at: datetime
    observed_at: datetime
    sentiment_feature_version: str
    score: SentimentScore


def dedupe_by_hash(events: Iterable[RawNewsEvent]) -> list[RawNewsEvent]:
    """Drop duplicate items by ``content_hash``; the latest-ingested copy wins.

    Deterministic: among rows sharing a hash, keep the one with the greatest
    ``ingested_at`` (ties broken by keeping the later-seen item), mirroring the
    latest-ingest-wins rule used by the price/microstructure lakes.
    """
    best: dict[str, RawNewsEvent] = {}
    for ev in events:
        cur = best.get(ev.content_hash)
        if cur is None or ev.ingested_at >= cur.ingested_at:
            best[ev.content_hash] = ev
    return list(best.values())


def filter_point_in_time(events: Sequence[RawNewsEvent], as_of: datetime) -> list[RawNewsEvent]:
    """Keep only events knowable at ``as_of`` (``observed_at <= as_of``), non-degraded.

    This is the single gate that prevents look-ahead: a feature computed for event
    time ``as_of`` may use the returned events and no others.
    """
    # Coerce as_of to UTC (naive == UTC) so the comparison never raises on a
    # naive/aware mismatch and the gate is timezone-consistent with stored events.
    as_of = as_of.astimezone(UTC) if as_of.tzinfo else as_of.replace(tzinfo=UTC)
    return [ev for ev in events if not ev.degraded and ev.observed_at <= as_of]
