"""Fail-closed authorization gate for ALL Databento *paid* downloads.

WHY THIS EXISTS (2026-06-09 incident): Databento bills per byte against whatever
payment method the API key's ACCOUNT has on file. The code selects an account only
by which API key resolves (``pass databento/api_key_2`` for the microstructure
ingest, ``OPTIONS_DATABENTO_API_KEY`` for the daily loader) — it CANNOT tell from
the API whether a download will draw on the free trial credits or charge a real
card. The cost guard only ever enforced a *dollar cap*, not the *funding source*.
So once the original free-credit key (``databento/api_key``, ~$125 trial credit) was
exhausted and removed, the active key billed a real card while every guard still
showed green.

This gate closes that hole: every real download path calls
:func:`assert_spend_authorized` BEFORE a single byte is fetched, and it refuses
unless the operator has explicitly attested — per process, via the environment —
that spending is authorized AND the account is safe (drawing on free credits, or
with a hard spend limit / no card on file on the Databento dashboard).

Default behaviour is BLOCKED. Dry-run cost estimation (``metadata.get_cost`` /
``get_billable_size``) is free and is NOT gated — only the actual ``get_range`` /
batch downloads are. To authorize one real download:

    OPTIONS_DATABENTO_SPEND_OK=1 uv run python -m options_system.microstructure.ingest ... --confirm

Leaving the variable unset (the default) makes it impossible for this code — or any
future run — to bill the account again.
"""

from __future__ import annotations

import os

#: Environment variable that must be explicitly set to authorize a paid download.
SPEND_ENV = "OPTIONS_DATABENTO_SPEND_OK"
_TRUTHY = frozenset({"1", "true", "yes", "on"})


class DatabentoSpendNotAuthorized(RuntimeError):
    """Raised when a real Databento download is attempted without explicit authorization."""


def spend_authorized() -> bool:
    """True only if the operator has explicitly opted in via :data:`SPEND_ENV`."""
    return os.environ.get(SPEND_ENV, "").strip().lower() in _TRUTHY


def assert_spend_authorized(context: str) -> None:
    """Refuse a real (billable) Databento download unless explicitly authorized.

    Call this at the top of every code path that hits ``timeseries.get_range`` or a
    batch download — never on the free dry-run estimate. ``context`` is a short human
    description of the download (symbols / window) for the error message.
    """
    if spend_authorized():
        return
    raise DatabentoSpendNotAuthorized(
        f"REFUSED: Databento paid download blocked for [{context}].\n"
        f"Databento bills per byte to the API key's account payment method, and this "
        f"code cannot verify whether that is free credits or a real card. To prevent "
        f"another accidental card charge, real downloads are fail-closed.\n"
        f"Before authorizing, confirm on the Databento dashboard that the account "
        f"either (a) draws on free trial credits, (b) has a hard spend limit at/below "
        f"the remaining free credit, or (c) has NO card on file. Then set "
        f"{SPEND_ENV}=1 for the single run that should spend (e.g. "
        f"`{SPEND_ENV}=1 uv run python -m ... --confirm`). Leaving it unset keeps all "
        f"Databento spending blocked."
    )
