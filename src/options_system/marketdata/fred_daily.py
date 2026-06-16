"""Daily FRED observation fetch (System B) — stdlib only, no new dependency.

Pulls a daily series' **final** observations (FRED ``output_type=1``). For the market
series used here — VIX/VXN, Treasury constant-maturity yields, WTI, the dollar index,
the high-yield OAS — the EOD value is **not revised** (the same rationale the macro layer
uses for the fed-funds target via :func:`macro.ingest.fetch_rate_series`), so the latest
observation equals the first print and there is no revision leak. Point-in-time safety is
then enforced downstream by stamping each value's ``observed_at`` at the END of its
observation day (see :mod:`options_system.marketdata.lake`).
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from datetime import date

import polars as pl

from ..common.logging import get_logger

logger = get_logger(__name__)

_FRED_BASE = "https://api.stlouisfed.org/fred"
_NA = {".", "", "NA", "NaN", "null", None}


def _fred_request(endpoint: str, params: dict[str, str], *, timeout: float = 30.0) -> dict:
    """GET ``{_FRED_BASE}/{endpoint}`` with ``params`` -> parsed JSON (stdlib urllib)."""
    url = f"{_FRED_BASE}/{endpoint}?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "options-system/marketdata"})
    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - https FRED
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_exc = exc
            logger.warning(f"FRED request failed (attempt {attempt + 1}/3): {exc}")
    raise RuntimeError(f"FRED request to {endpoint} failed after retries") from last_exc


def fetch_daily_series(
    series_id: str,
    api_key: str,
    *,
    observation_start: date,
    observation_end: date | None = None,
) -> pl.DataFrame:
    """Final daily observations for ``series_id`` -> ``[obs_date (Date), value (Float64)]``.

    Missing values (FRED ``"."``) are dropped. Sorted ascending by date. An empty,
    correctly-typed frame is returned when FRED has no observations in the window.
    """
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": observation_start.isoformat(),
        "observation_end": (observation_end or date(2100, 1, 1)).isoformat(),
    }
    payload = _fred_request("series/observations", params)
    rows = [
        {"obs_date": date.fromisoformat(o["date"]), "value": float(o["value"])}
        for o in payload.get("observations", [])
        if o.get("value") not in _NA
    ]
    if not rows:
        return pl.DataFrame(schema={"obs_date": pl.Date, "value": pl.Float64})
    return pl.DataFrame(rows).sort("obs_date")
