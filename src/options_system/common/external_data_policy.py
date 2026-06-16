"""Fail-closed safety policy for NON-Databento external data sources.

This is the generic companion to :mod:`options_system.common.databento_guard`.
``databento_guard`` is a single-purpose, environment-gated kill switch for the
one source that already cost real money (see the 2026-06-09 incident). This module
generalises the *idea* — never touch an external source we have not explicitly
classified as safe — to every other source the system might reach (news, filings,
local models, future paid APIs).

It does NOT replace ``databento_guard``: Databento keeps its dedicated, stricter
env-attestation gate. This module is the default policy for everything else, and
it classifies Databento (and other paid APIs) as ``PAID_BLOCKED`` so a careless
new caller is refused here too.

Policy classes
--------------
* ``FREE_NO_AUTH`` — free, open, no API key / no account / no card (GDELT DOC, the
  GDELT GKG bulk archive, SEC EDGAR). Network is permitted, but ONLY when the caller
  explicitly opts in (``allow_network=True``, surfaced as a ``--allow-network`` CLI
  flag). Default stays offline.
* ``FREE_AUTH`` — free, but requires a free API key / account (no card, no charge) —
  e.g. FRED/ALFRED. Treated at the network gate exactly like ``FREE_NO_AUTH`` (an
  explicit ``allow_network=True`` is still required); the *key* requirement is enforced
  by the caller (it reads the key from config), since this module deliberately knows
  nothing about secrets.
* ``LOCAL_ONLY`` — runs entirely on this machine (local FinBERT weights). It must
  never perform network access; asking it to is a programming error and is refused.
* ``PAID_BLOCKED`` — costs money / needs a card or paid subscription (Databento,
  Finnhub). Always refused by this module.
* ``UNKNOWN_BLOCKED`` — anything not in the registry. **Fails closed.** A source we
  have not deliberately vetted is treated as unsafe.

The whole point is that the *default* — an unrecognised source, or a recognised
free source without an explicit network opt-in — does nothing over the network.
"""

from __future__ import annotations

from enum import StrEnum


class SourcePolicy(StrEnum):
    """How an external data source is allowed to be accessed."""

    PAID_BLOCKED = "paid_blocked"
    FREE_NO_AUTH = "free_no_auth"
    FREE_AUTH = "free_auth"
    LOCAL_ONLY = "local_only"
    UNKNOWN_BLOCKED = "unknown_blocked"


#: Canonical, authoritative source -> policy mapping. This module — NOT a YAML file
#: — is the source of truth, so a typo or an omission in config can never widen
#: access. Config may *declare* the same mapping for documentation, but it is
#: cross-checked against this, never trusted over it.
DEFAULT_REGISTRY: dict[str, SourcePolicy] = {
    "gdelt": SourcePolicy.FREE_NO_AUTH,
    "gdelt_gkg": SourcePolicy.FREE_NO_AUTH,  # bulk GKG 15-min file archive (no key)
    "sec_edgar": SourcePolicy.FREE_NO_AUTH,
    "fred": SourcePolicy.FREE_AUTH,  # free, but needs a free FRED API key (no card)
    "finbert_local": SourcePolicy.LOCAL_ONLY,
    "finnhub": SourcePolicy.PAID_BLOCKED,
    "databento": SourcePolicy.PAID_BLOCKED,
}


class ExternalAccessNotAuthorized(RuntimeError):
    """Raised when an external source is accessed in a way the policy forbids."""


def _normalise(source: str) -> str:
    return source.strip().lower()


def classify(source: str, registry: dict[str, SourcePolicy] | None = None) -> SourcePolicy:
    """Return the policy for ``source``. Unknown sources fail closed (``UNKNOWN_BLOCKED``)."""
    reg = registry if registry is not None else DEFAULT_REGISTRY
    return reg.get(_normalise(source), SourcePolicy.UNKNOWN_BLOCKED)


def requires_network(source: str, registry: dict[str, SourcePolicy] | None = None) -> bool:
    """Whether reaching ``source`` for fresh data inherently involves the network.

    Only ``FREE_NO_AUTH`` and ``FREE_AUTH`` sources fetch over the network in this
    system. Local-only sources never do; paid/unknown sources are blocked outright and
    so are treated as not network-eligible here.
    """
    return classify(source, registry) in (SourcePolicy.FREE_NO_AUTH, SourcePolicy.FREE_AUTH)


def assert_network_allowed(
    source: str,
    *,
    allow_network: bool,
    registry: dict[str, SourcePolicy] | None = None,
) -> None:
    """Authorize (or refuse) a network fetch from ``source``.

    Call this at the top of any code path that is about to make a real network
    request. It permits the request ONLY when the source is allow-listed as
    ``FREE_NO_AUTH`` **and** the caller explicitly passed ``allow_network=True``.
    Every other case — local-only (must stay offline), paid (blocked), or unknown
    (fail closed) — raises :class:`ExternalAccessNotAuthorized`.
    """
    policy = classify(source, registry)
    name = _normalise(source)
    if policy in (SourcePolicy.FREE_NO_AUTH, SourcePolicy.FREE_AUTH):
        if not allow_network:
            raise ExternalAccessNotAuthorized(
                f"REFUSED: network fetch from free source '{name}' is blocked by default. "
                f"Re-run with an explicit --allow-network (allow_network=True) to permit a "
                f"single bounded fetch. Default behaviour is offline/fixture-only."
            )
        # FREE_AUTH additionally needs a key, but that is the caller's responsibility
        # (this module knows nothing about secrets). The network gate itself is identical.
        return
    if policy is SourcePolicy.LOCAL_ONLY:
        raise ExternalAccessNotAuthorized(
            f"REFUSED: source '{name}' is local-only and must never perform network access. "
            f"Run it entirely on local resources (e.g. local model weights)."
        )
    if policy is SourcePolicy.PAID_BLOCKED:
        raise ExternalAccessNotAuthorized(
            f"REFUSED: source '{name}' is a paid/credentialed source and is hard-blocked here. "
            f"Paid data requires a separate, explicit operator authorization (and, for "
            f"Databento, the OPTIONS_DATABENTO_SPEND_OK gate). It is never reached by this path."
        )
    raise ExternalAccessNotAuthorized(
        f"REFUSED: source '{name}' is unknown / unvetted and fails closed. Only sources "
        f"explicitly allow-listed in external_data_policy.DEFAULT_REGISTRY may be accessed."
    )


def assert_source_usable(
    source: str, registry: dict[str, SourcePolicy] | None = None
) -> SourcePolicy:
    """Refuse paid/unknown sources outright (even for offline/fixture use).

    Returns the resolved policy for allowed sources (free or local-only) so callers
    can branch; raises :class:`ExternalAccessNotAuthorized` for paid or unknown ones.
    Use this before scaffolding *any* work for a source, so paid/unknown sources are
    rejected up front — not just at the network boundary.
    """
    policy = classify(source, registry)
    if policy in (SourcePolicy.PAID_BLOCKED, SourcePolicy.UNKNOWN_BLOCKED):
        raise ExternalAccessNotAuthorized(
            f"REFUSED: source '{_normalise(source)}' has policy {policy.value}; it is blocked. "
            f"Only free-no-auth or local-only sources are usable."
        )
    return policy
