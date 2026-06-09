"""SEC EDGAR adapter — minimal scaffold (fixture-parse now, gated fetch later).

SEC EDGAR (https://data.sec.gov) exposes filing metadata with **no API key** — the
only requirement is a descriptive ``User-Agent`` header and respect for the published
rate guidance. This adapter parses the ``submissions`` JSON shape into the
point-in-time event schema. It does NOT build a ticker universe and does NOT download
in this phase; the real fetch (:func:`fetch_submissions`) is guarded by
:func:`external_data_policy.assert_network_allowed`.

EDGAR is **future context** for index / earnings / megacap-tech themes, not yet a
trading signal. Point-in-time stance: a filing's ``acceptanceDateTime`` is when it
became public, so both ``published_at`` and ``observed_at`` are set to it (falling back
to ``filingDate`` at day start when acceptance time is absent).
"""

from __future__ import annotations

import json
import urllib.request
from datetime import UTC, datetime, time

from options_system.common.external_data_policy import assert_network_allowed
from options_system.sentiment.schema import RawNewsEvent

SOURCE = "sec_edgar"
_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"


def build_submissions_url(cik: int | str) -> str:
    """URL for a company's submissions JSON (CIK zero-padded to 10 digits)."""
    return _SUBMISSIONS_URL.format(cik=int(cik))


def _at(seq: list, i: int) -> str:
    """Safe positional read of a parallel-array column ('' when absent/None)."""
    return str(seq[i]) if i < len(seq) and seq[i] is not None else ""


def _parse_acceptance(raw: str | None, filing_date: str) -> datetime:
    """Best-effort public-availability timestamp for one filing (UTC)."""
    if raw:
        s = raw.strip()
        # Accept both "...Z" and naive forms; treat naive as UTC.
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=UTC)
    # Fall back to the filing date at day start (conservative — never earlier).
    d = datetime.fromisoformat(filing_date.strip()).date()
    return datetime.combine(d, time(0), UTC)


def parse_submissions(
    payload: dict,
    *,
    topic: str,
    sentiment_feature_version: str,
    ingested_at: datetime,
) -> list[RawNewsEvent]:
    """Parse an EDGAR ``submissions`` JSON payload's ``filings.recent`` into events."""
    name = (payload.get("name") or "").strip()
    cik = payload.get("cik")
    tickers = tuple(str(t) for t in (payload.get("tickers") or []))
    recent = (payload.get("filings") or {}).get("recent") or {}

    accession = recent.get("accessionNumber") or []
    filing_dates = recent.get("filingDate") or []
    acceptance = recent.get("acceptanceDateTime") or []
    forms = recent.get("form") or []
    primary_doc = recent.get("primaryDocument") or []
    primary_desc = recent.get("primaryDocDescription") or []
    items = recent.get("items") or []

    n = len(accession)
    out: list[RawNewsEvent] = []
    for i in range(n):
        form = _at(forms, i)
        desc = _at(primary_desc, i)
        item_str = _at(items, i)
        filing_date = _at(filing_dates, i)
        acc = _at(accession, i)
        title = f"{name} {form}".strip() + (f" — {desc}" if desc else "")
        body = " ".join(p for p in (form, desc, item_str) if p)
        published = _parse_acceptance(_at(acceptance, i) or None, filing_date)
        acc_nodash = acc.replace("-", "")
        url = (
            f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nodash}/{_at(primary_doc, i)}"
            if cik and acc_nodash
            else None
        )
        out.append(
            RawNewsEvent(
                source=SOURCE,
                source_id=acc or f"{cik}:{i}",
                source_url=url,
                title=title or f"{name} filing",
                snippet_or_text=body,
                published_at=published,
                observed_at=published,  # public at acceptance time
                ingested_at=ingested_at,
                query_topic=topic,
                language="eng",
                entities=(*tickers, name) if name else tickers,
                sentiment_feature_version=sentiment_feature_version,
            )
        )
    return out


def fetch_submissions(
    cik: int | str,
    *,
    topic: str,
    sentiment_feature_version: str,
    ingested_at: datetime,
    user_agent: str,
    allow_network: bool,
    timeout: float = 30.0,
) -> list[RawNewsEvent]:
    """Bounded real fetch of one company's submissions — GATED. Not used in scaffold.

    Refuses unless ``allow_network`` is explicitly True; sends the required SEC
    ``User-Agent`` (no API key).
    """
    assert_network_allowed(SOURCE, allow_network=allow_network)  # fail-closed
    if not user_agent or "set-in-env" in user_agent:
        raise ValueError(
            "SEC EDGAR requires a real descriptive User-Agent (name + contact email). "
            "Set fetch_limits.sec_user_agent before any real fetch."
        )
    url = build_submissions_url(cik)
    req = urllib.request.Request(url, headers={"User-Agent": user_agent})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - https SEC API
        payload = json.loads(resp.read().decode("utf-8"))
    return parse_submissions(
        payload,
        topic=topic,
        sentiment_feature_version=sentiment_feature_version,
        ingested_at=ingested_at,
    )
