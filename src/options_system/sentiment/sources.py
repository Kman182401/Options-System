"""Registry of sentiment source adapters and their safety policy.

Thin descriptor layer over :mod:`options_system.common.external_data_policy`. It
names the sources this layer knows how to *shape* (parse into the raw event schema)
and records, for each, the authoritative policy and whether a network fetch needs an
HTTP ``User-Agent`` header (SEC EDGAR requires one; it is still no-auth / no-key).

Resolving a source goes through :func:`get_source_spec`, which refuses paid/unknown
sources up front via :func:`external_data_policy.assert_source_usable` — so a blocked
source is rejected before any adapter code runs, not just at the network boundary.
"""

from __future__ import annotations

from dataclasses import dataclass

from options_system.common.external_data_policy import (
    SourcePolicy,
    assert_source_usable,
    classify,
)


@dataclass(frozen=True)
class SourceSpec:
    """Static description of a sentiment data source."""

    name: str
    policy: SourcePolicy
    description: str
    needs_user_agent: bool = False


#: Sources this layer can parse into RawNewsEvent. Policy is taken from the
#: authoritative registry via :func:`classify`, never hard-coded here, so the two can
#: never drift.
SENTIMENT_SOURCES: dict[str, SourceSpec] = {
    "gdelt": SourceSpec(
        name="gdelt",
        policy=classify("gdelt"),
        description="GDELT DOC 2.0 article list (free/open global news metadata + tone).",
        needs_user_agent=False,
    ),
    "sec_edgar": SourceSpec(
        name="sec_edgar",
        policy=classify("sec_edgar"),
        description="SEC EDGAR submissions / companyfacts (data.sec.gov; no key, UA required).",
        needs_user_agent=True,
    ),
    "finbert_local": SourceSpec(
        name="finbert_local",
        policy=classify("finbert_local"),
        description="Local FinBERT scorer (runs on local weights; never networks).",
        needs_user_agent=False,
    ),
}

#: The pseudo-source the CLI accepts to load a pre-shaped RawNewsEvent fixture offline.
FIXTURE_SOURCE = "fixture"


def get_source_spec(name: str) -> SourceSpec:
    """Return the spec for ``name``, refusing paid/unknown sources outright."""
    assert_source_usable(name)  # raises for paid_blocked / unknown_blocked
    key = name.strip().lower()
    spec = SENTIMENT_SOURCES.get(key)
    if spec is None:
        raise KeyError(
            f"no sentiment adapter registered for source {name!r}; "
            f"known: {sorted(SENTIMENT_SOURCES)}"
        )
    return spec


def known_sources() -> list[str]:
    """Names of sources with a registered adapter (excludes the fixture pseudo-source)."""
    return sorted(SENTIMENT_SOURCES)
