"""GDELT DOC 2.0 adapter — fixture-parse now, bounded real fetch later (gated).

GDELT (https://www.gdeltproject.org) is a free/open global news index. The DOC 2.0
``ArtList`` endpoint returns article metadata (url, title, the crawl time
``seendate``, domain, language) as JSON — no API key, no account, no card.

This phase only **parses fixtures** into the point-in-time event schema; it does not
run a broad ingestion. A real fetch is implemented (:func:`fetch_artlist`) but is
guarded by :func:`external_data_policy.assert_network_allowed` and is never invoked
here. Point-in-time stance: GDELT's ``seendate`` is when GDELT first *saw* the item,
which is the earliest our system could have known of it — so both ``published_at``
and ``observed_at`` are set conservatively to ``seendate``.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from datetime import UTC, datetime

from options_system.common.external_data_policy import assert_network_allowed
from options_system.sentiment.schema import RawNewsEvent

SOURCE = "gdelt"
_DOC_ENDPOINT = "https://api.gdeltproject.org/api/v2/doc/doc"
_SEENDATE_FMT = "%Y%m%dT%H%M%SZ"


def build_query_url(
    *,
    topic: str,
    start: datetime,
    end: datetime,
    max_records: int,
    language: str | None = None,
    query_text: str | None = None,
) -> str:
    """Construct a bounded GDELT DOC 2.0 ArtList request URL (no network performed).

    ``max_records`` is hard-capped at GDELT's 250 ceiling. ``language`` (a GDELT lang
    code like ``eng``) is appended to the query when given. ``query_text`` overrides
    the search text actually sent (some stable topic LABELS, e.g. ``ai_capex``, are
    not usable as literal GDELT tokens); when None the topic label itself is queried.
    """
    query = (query_text or topic).strip()
    if language:
        query = f"{query} sourcelang:{language}"
    params = {
        "query": query,
        "mode": "ArtList",
        "format": "json",
        "maxrecords": str(max(1, min(int(max_records), 250))),
        "startdatetime": start.astimezone(UTC).strftime("%Y%m%d%H%M%S"),
        "enddatetime": end.astimezone(UTC).strftime("%Y%m%d%H%M%S"),
    }
    return f"{_DOC_ENDPOINT}?{urllib.parse.urlencode(params)}"


def _parse_seendate(raw: str) -> datetime:
    return datetime.strptime(raw.strip(), _SEENDATE_FMT).replace(tzinfo=UTC)


def parse_artlist(
    payload: dict,
    *,
    topic: str,
    sentiment_feature_version: str,
    ingested_at: datetime,
) -> list[RawNewsEvent]:
    """Parse a GDELT ArtList JSON payload into raw events.

    Malformed articles (missing ``seendate``/``title``) become a single ``degraded``
    event carrying the error, so the failure is recorded but is excluded from feature
    generation by :func:`schema.filter_point_in_time`.
    """
    articles = payload.get("articles") or []
    out: list[RawNewsEvent] = []
    for art in articles:
        title = (art.get("title") or "").strip()
        seendate = art.get("seendate")
        url = (art.get("url") or "").strip()
        if not seendate or not title:
            out.append(
                RawNewsEvent(
                    source=SOURCE,
                    source_id=url or f"degraded:{title[:40]}",
                    source_url=url or None,
                    title=title or "(missing title)",
                    snippet_or_text=title,
                    published_at=ingested_at,
                    observed_at=ingested_at,
                    ingested_at=ingested_at,
                    query_topic=topic,
                    language=(art.get("language") or None),
                    sentiment_feature_version=sentiment_feature_version,
                    degraded=True,
                    error="missing seendate or title",
                )
            )
            continue
        seen = _parse_seendate(seendate)
        out.append(
            RawNewsEvent(
                source=SOURCE,
                source_id=url or title,
                source_url=url or None,
                title=title,
                # ArtList carries no article body; the title is the available text.
                snippet_or_text=title,
                published_at=seen,
                observed_at=seen,  # conservative: GDELT first-seen
                ingested_at=ingested_at,
                query_topic=topic,
                language=(art.get("language") or None),
                sentiment_feature_version=sentiment_feature_version,
            )
        )
    return out


def fetch_artlist(
    *,
    topic: str,
    start: datetime,
    end: datetime,
    max_records: int,
    sentiment_feature_version: str,
    ingested_at: datetime,
    allow_network: bool,
    language: str | None = None,
    timeout: float = 30.0,
    query_text: str | None = None,
) -> list[RawNewsEvent]:
    """Bounded real fetch — GATED. Refuses unless ``allow_network`` is explicitly True.

    Not exercised in the scaffold phase or in tests. Kept so the network path exists
    and is provably behind the policy gate. ``query_text`` overrides the search text
    sent to GDELT; the ``topic`` label is what gets stamped on the parsed events.
    """
    assert_network_allowed(SOURCE, allow_network=allow_network)  # fail-closed
    url = build_query_url(
        topic=topic,
        start=start,
        end=end,
        max_records=max_records,
        language=language,
        query_text=query_text,
    )
    # Send a standard JSON client's headers. GDELT rate-limits (HTTP 429) header-less
    # requests far more aggressively than well-formed ones; an explicit Accept and an
    # identity Accept-Encoding (urllib does not auto-decompress gzip) match a normal
    # client and avoid a decode error on the response body.
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Options-System/research",
            "Accept": "application/json",
            "Accept-Encoding": "identity",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - https GDELT API
        payload = json.loads(resp.read().decode("utf-8"))
    return parse_artlist(
        payload,
        topic=topic,
        sentiment_feature_version=sentiment_feature_version,
        ingested_at=ingested_at,
    )
