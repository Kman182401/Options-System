"""GDELT GKG 2.1 bulk-file adapter — parse + filter + map to PIT events (System A).

The GKG 2.1 "translingual"-English stream publishes one tab-separated file every 15
minutes (``YYYYMMDDHHMMSS.gkg.csv.zip``) at http://data.gdeltproject.org/gdeltv2/ . Each
row is one news article GDELT processed, with a **precomputed tone** (the V1.5/V2 ``Tone``
field) — so, unlike the DOC ArtList path, we do NOT run FinBERT here: GDELT's tone IS the
score. This module turns those rows into the existing point-in-time event schema
(:class:`~options_system.sentiment.schema.RawNewsEvent` +
:class:`~options_system.sentiment.schema.ScoredNewsEvent`) so the rest of the lake /
aggregation machinery is reused unchanged.

Column layout (verified against a live file, 27 tab-separated fields; 0-indexed):

* ``[1]``  DATE — ``YYYYMMDDHHMMSS`` UTC, GDELT first-seen → conservative ``observed_at``.
* ``[3]``  SourceCommonName — the publisher domain.
* ``[4]``  DocumentIdentifier — the article URL (stable id).
* ``[7]``  V1Themes — ``;``-separated GKG theme tokens (the row filter reads these).
* ``[15]`` V1.5Tone — ``,``-separated: ``tone,positive,negative,polarity,...`` (percent units).
* ``[26]`` Extras — XML; may carry ``<PAGE_TITLE>...</PAGE_TITLE>``.

Point-in-time stance (identical to :mod:`options_system.sentiment.gdelt`): GKG ``DATE`` is
the earliest moment our system could have known the item, so ``published_at`` and
``observed_at`` are both set to it. The schema validator enforces
``published_at <= observed_at <= ingested_at``.
"""

from __future__ import annotations

import html
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from options_system.sentiment.schema import (
    RawNewsEvent,
    ScoredNewsEvent,
    SentimentScore,
)

SOURCE = "gdelt_gkg"

# Verified 0-indexed field positions (see module docstring).
_F_DATE = 1
_F_SOURCE = 3
_F_DOCID = 4
_F_V1THEMES = 7
_F_TONE = 15
_F_EXTRAS = 26
# We read up to the tone field; rows shorter than this are malformed and skipped.
_MIN_FIELDS = _F_TONE + 1

_DATE_FMT = "%Y%m%d%H%M%S"
_PAGE_TITLE_RE = re.compile(r"<PAGE_TITLE>(.*?)</PAGE_TITLE>", re.IGNORECASE | re.DOTALL)


def parse_gkg_datetime(raw: str) -> datetime:
    """``YYYYMMDDHHMMSS`` -> tz-aware UTC datetime (raises ValueError if malformed)."""
    return datetime.strptime(raw.strip(), _DATE_FMT).replace(tzinfo=UTC)


def parse_tone(field: str) -> tuple[float, float, float] | None:
    """The V1.5 Tone field -> ``(tone, positive, negative)`` in percent units.

    Returns None when the field is empty or its first three components are not numbers
    (the row then cannot be scored and is skipped — not an error).
    """
    s = field.strip()
    if not s:
        return None
    parts = s.split(",")
    if len(parts) < 3:
        return None
    try:
        return float(parts[0]), float(parts[1]), float(parts[2])
    except ValueError:
        return None


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def tone_to_score(
    tone: float, positive: float, negative: float, *, model_name: str, scored_at: datetime
) -> SentimentScore:
    """Map GDELT tone (percent units) to the system's :class:`SentimentScore`.

    GDELT ``positive``/``negative`` are percentages of words (0..100) and ``tone`` is
    ``positive - negative``; dividing by 100 puts them on the system's [0,1] / [-1,1]
    scales, exactly matching ``sentiment_score == positive_score - negative_score``.
    Values are clamped defensively so a stray out-of-range component cannot violate the
    score schema.
    """
    pos = _clamp(positive / 100.0, 0.0, 1.0)
    neg = _clamp(negative / 100.0, 0.0, 1.0)
    neu = _clamp(1.0 - pos - neg, 0.0, 1.0)
    return SentimentScore(
        positive_score=pos,
        negative_score=neg,
        neutral_score=neu,
        sentiment_score=_clamp(tone / 100.0, -1.0, 1.0),
        model_name=model_name,
        model_version_or_hash=None,
        scored_at=scored_at,
    )


def extract_title(extras_field: str) -> str:
    """Pull ``<PAGE_TITLE>`` from the Extras XML (HTML-unescaped); ``""`` if absent."""
    m = _PAGE_TITLE_RE.search(extras_field or "")
    return html.unescape(m.group(1)).strip() if m else ""


def matches_themes(v1themes_field: str, prefixes: Sequence[str]) -> bool:
    """True if any ``;``-separated theme token starts with any allowed prefix (ci)."""
    if not v1themes_field:
        return False
    ups = [p.upper() for p in prefixes]
    for token in v1themes_field.split(";"):
        tok = token.strip().upper()
        if tok and any(tok.startswith(p) for p in ups):
            return True
    return False


@dataclass(frozen=True)
class GkgFileParse:
    """Result of parsing one GKG file: kept events + per-file counters for the manifest."""

    raw: list[RawNewsEvent]
    scored: list[ScoredNewsEvent]
    n_rows: int  # total data rows in the file
    n_kept: int  # rows that matched the theme filter AND scored
    n_malformed: int  # rows too short / bad date (skipped, recorded for visibility)


def parse_gkg_file(
    text: str,
    *,
    theme_prefixes: Sequence[str],
    query_topic: str,
    event_version: str,
    tone_model_name: str,
    ingested_at: datetime,
) -> GkgFileParse:
    """Parse one decoded GKG file into kept (raw, scored) events + counters.

    A row is **kept** only if it matches the theme filter and carries a parseable tone.
    Theme-miss and empty-tone rows are simply dropped (the normal majority). Rows too
    short to read the tone field, or with an unparseable DATE, are counted as malformed
    and skipped. Nothing is ever stored that cannot be made point-in-time correct.
    """
    raw: list[RawNewsEvent] = []
    scored: list[ScoredNewsEvent] = []
    n_rows = 0
    n_malformed = 0

    for line in text.splitlines():
        if not line.strip():
            continue
        n_rows += 1
        fields = line.split("\t")
        if len(fields) < _MIN_FIELDS:
            n_malformed += 1
            continue
        if not matches_themes(fields[_F_V1THEMES], theme_prefixes):
            continue
        tone = parse_tone(fields[_F_TONE])
        if tone is None:
            continue
        try:
            when = parse_gkg_datetime(fields[_F_DATE])
        except ValueError:
            n_malformed += 1
            continue

        url = fields[_F_DOCID].strip()
        record_id = fields[0].strip()
        source_id = url or record_id or f"{fields[_F_SOURCE].strip()}|{fields[_F_DATE].strip()}"
        extras = fields[_F_EXTRAS] if len(fields) > _F_EXTRAS else ""
        title = extract_title(extras) or fields[_F_SOURCE].strip()

        event = RawNewsEvent(
            source=SOURCE,
            source_id=source_id,
            source_url=url or None,
            title=title,
            snippet_or_text=title,  # GKG carries no body; the title is the available text
            published_at=when,
            observed_at=when,  # conservative: GDELT first-seen
            ingested_at=ingested_at,
            query_topic=query_topic,
            language="eng",
            sentiment_feature_version=event_version,
        )
        score = tone_to_score(*tone, model_name=tone_model_name, scored_at=ingested_at)
        raw.append(event)
        scored.append(
            ScoredNewsEvent(
                content_hash=event.content_hash,
                source=SOURCE,
                query_topic=query_topic,
                published_at=when,
                observed_at=when,
                sentiment_feature_version=event_version,
                score=score,
            )
        )

    return GkgFileParse(
        raw=raw, scored=scored, n_rows=n_rows, n_kept=len(raw), n_malformed=n_malformed
    )
