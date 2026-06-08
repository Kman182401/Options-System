"""Point-in-time macro-event ingestion from FRED / ALFRED — free, key-gated.

This is the *raw external input* for the Phase-6 macro layer: a calendar of
high-impact US economic releases, 2019→2026, each stamped with **what was known
at release time** and never a later revision.

Point-in-time, the event-specific way (the whole leakage game lives here):

* **Scheduled timing is public in advance** — the release date+time of the next
  CPI / FOMC is knowable before it happens, so timing features (built later in
  :mod:`options_system.features.macro_features`) may legitimately look ahead at
  the *schedule*.
* **Values are known only at release** — so every ``actual_pit`` is the
  **first-print** value (FRED ``output_type=4`` = "Observations, Initial Release
  Only"), and its ``event_time`` is the publication date (the observation's
  ``realtime_start``) combined with the standard release clock time (08:30 ET for
  data, 14:00 ET for the FOMC statement), converted to UTC. A later revision is
  never used.

``surprise`` (actual vs consensus) is left **null**: FRED carries no consensus
series and we do not fabricate one. The downstream outcome feature is the change
vs the prior *first-print* release (``actual_pit - prior``), not a surprise.

Network access uses only the Python standard library (``urllib``) — no new
dependency. The whole thing is **key-gated**: with ``OPTIONS_FRED_API_KEY`` unset
it no-ops with a clear message and the rest of the system runs unchanged.

Storage: a ``macro_events`` table in the lake at
``data/macro_events/date=<YYYY-MM-DD>/part-<uuid>.parquet`` (partitioned by the
event-time date, **not** by symbol — macro context is instrument-independent),
append-only and idempotent on the natural key ``(event_type, event_time)``.
"""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, date, datetime, time
from glob import glob as _glob
from pathlib import Path
from uuid import uuid4
from zoneinfo import ZoneInfo

import polars as pl

from config.settings import Settings

from ..common.logging import get_logger
from ..data.store import DuckStore
from .config import MacroConfig

logger = get_logger(__name__)

_DATASET = "macro_events"
_FRED_BASE = "https://api.stlouisfed.org/fred"
_NA = {".", "", "NA", "NaN", "null", None}

# Canonical macro_events schema (column -> polars dtype), in on-disk order.
_TS = pl.Datetime("us", "UTC")
_SCHEMA: dict[str, pl.DataType] = {
    "event_time": _TS,  # UTC release timestamp (release date @ standard clock time)
    "event_type": pl.Utf8(),  # cpi / core_cpi / ... / fomc
    "ref_period": pl.Date(),  # economic period the value refers to (FOMC: decision date)
    "actual_pit": pl.Float64(),  # FIRST-PRINT value as known at release (never a revision)
    "prior": pl.Float64(),  # previous release's first-print value (null for the first)
    "surprise": pl.Float64(),  # actual vs consensus — NULL (no free consensus source)
    "source": pl.Utf8(),  # e.g. "FRED:CPIAUCSL"
    "fred_series_id": pl.Utf8(),
    "ingest_ts": _TS,
    "macro_version": pl.Utf8(),
}

# How far before the bar window to start pulling releases, so the earliest bars
# already have a populated "most recent release / time since last event" context.
_INGEST_START = date(2018, 6, 1)


# --------------------------------------------------------------------------- #
# FRED HTTP (stdlib only)
# --------------------------------------------------------------------------- #
def _fred_request(endpoint: str, params: dict[str, str], *, timeout: float = 30.0) -> dict:
    """GET ``{_FRED_BASE}/{endpoint}`` with ``params`` → parsed JSON (stdlib urllib)."""
    url = f"{_FRED_BASE}/{endpoint}?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "options-system/macro"})
    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - https FRED only
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_exc = exc
            logger.warning(f"FRED request failed (attempt {attempt + 1}/3): {exc}")
    raise RuntimeError(f"FRED request to {endpoint} failed after retries") from last_exc


def fetch_first_print(
    series_id: str,
    api_key: str,
    *,
    observation_start: date = _INGEST_START,
    observation_end: date | None = None,
) -> pl.DataFrame:
    """First-print observations for ``series_id`` via FRED ``output_type=4``.

    Returns columns ``[ref_period (Date), value (Float64), release_date (Date)]``
    where ``value`` is the value as **first released** and ``release_date`` is that
    first vintage's ``realtime_start`` (the publication date). Missing values
    (FRED ``"."``) are dropped. ``output_type=4`` = "Observations, Initial Release
    Only", so no later revision can leak in.
    """
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "output_type": "4",  # initial release only (first print)
        "observation_start": observation_start.isoformat(),
        "observation_end": (observation_end or date(2100, 1, 1)).isoformat(),
        # output_type=4 requires an explicit real-time window spanning the vintages;
        # the default (today only) errors "no vintage dates exist". A wide window
        # returns every first-print value with realtime_start = its publication date.
        "realtime_start": "2000-01-01",
        "realtime_end": "9999-12-31",
    }
    payload = _fred_request("series/observations", params)
    rows = []
    for o in payload.get("observations", []):
        raw = o.get("value")
        if raw in _NA:
            continue
        rows.append(
            {
                "ref_period": date.fromisoformat(o["date"]),
                "value": float(raw),
                "release_date": date.fromisoformat(o["realtime_start"]),
            }
        )
    if not rows:
        return pl.DataFrame(
            schema={"ref_period": pl.Date, "value": pl.Float64, "release_date": pl.Date}
        )
    return pl.DataFrame(rows).sort("ref_period")


def fetch_rate_series(
    series_id: str,
    api_key: str,
    *,
    observation_start: date = _INGEST_START,
) -> pl.DataFrame:
    """Daily target-rate observations (``[ref_period, value]``, sorted).

    Used for the FOMC outcome (DFEDTARU). We deliberately use the standard
    (latest) observations rather than ``output_type=4``: the federal-funds target
    rate is **never revised**, so the latest value equals the first print — and a
    daily series has too many ALFRED vintages (>2000) for the initial-release file
    type anyway. Leak-free because the value is only ever read as-of a meeting date
    (the rate is announced in that meeting's 14:00-ET statement).
    """
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": observation_start.isoformat(),
    }
    payload = _fred_request("series/observations", params)
    rows = [
        {"ref_period": date.fromisoformat(o["date"]), "value": float(o["value"])}
        for o in payload.get("observations", [])
        if o.get("value") not in _NA
    ]
    if not rows:
        return pl.DataFrame(schema={"ref_period": pl.Date, "value": pl.Float64})
    return pl.DataFrame(rows).sort("ref_period")


# --------------------------------------------------------------------------- #
# event_time construction
# --------------------------------------------------------------------------- #
def _event_time_utc(release_day: date, release_et: time, tz: ZoneInfo) -> datetime:
    """Combine a release DATE with its standard ET clock time → a UTC timestamp."""
    local = datetime.combine(release_day, release_et, tzinfo=tz)
    return local.astimezone(UTC)


def _asof_value(frame: pl.DataFrame, when: date) -> float | None:
    """Latest ``value`` in ``frame`` (sorted by ``ref_period``) with ref_period <= ``when``."""
    sub = frame.filter(pl.col("ref_period") <= when)
    if sub.is_empty():
        return None
    return float(sub["value"][-1])


# --------------------------------------------------------------------------- #
# Build the macro_events frame
# --------------------------------------------------------------------------- #
def _data_events(cfg: MacroConfig, api_key: str, tz: ZoneInfo, ingest_ts: datetime) -> pl.DataFrame:
    """Ingest every FRED economic-data release type into macro_events rows."""
    frames: list[pl.DataFrame] = []
    for etype, spec in cfg.events.items():
        fp = fetch_first_print(spec.series_id, api_key)
        if fp.is_empty():
            logger.warning(f"[{etype}] FRED returned no first-print observations; skipping")
            continue
        # prior = the previous period's first-print value (as known at this release).
        fp = fp.with_columns(pl.col("value").shift(1).alias("prior"))
        rows = fp.select(
            pl.col("release_date")
            .map_elements(
                lambda d, rt=spec.release_et: _event_time_utc(d, rt, tz),
                return_dtype=_TS,
            )
            .alias("event_time"),
            pl.lit(etype).alias("event_type"),
            pl.col("ref_period"),
            pl.col("value").alias("actual_pit"),
            pl.col("prior"),
            pl.lit(None, dtype=pl.Float64).alias("surprise"),
            pl.lit(f"FRED:{spec.series_id}").alias("source"),
            pl.lit(spec.series_id).alias("fred_series_id"),
            pl.lit(ingest_ts).cast(_TS).alias("ingest_ts"),
            pl.lit(cfg.macro_version).alias("macro_version"),
        )
        frames.append(rows)
        logger.info(f"[{etype}] {rows.height} first-print releases ({spec.series_id})")
    return pl.concat(frames) if frames else _empty()


def _fomc_events(cfg: MacroConfig, api_key: str, tz: ZoneInfo, ingest_ts: datetime) -> pl.DataFrame:
    """FOMC scheduled meetings → events; rate outcome from the target-upper series.

    ``actual_pit`` = federal-funds target upper limit set at the meeting (the rate
    in effect just after it), ``prior`` = the rate just before. The new target is
    announced in the 14:00-ET statement, so attaching it at the decision-day
    statement time introduces no look-ahead. Meetings beyond the rate series'
    coverage (e.g. tentative future dates) get a null outcome but a valid
    ``event_time`` so timing features still work.
    """
    fomc = cfg.fomc
    rates = fetch_rate_series(fomc.rate_series_id, api_key)
    rows = []
    for d in fomc.decision_dates:
        actual = (
            _asof_value(rates, date.fromordinal(d.toordinal() + 3))
            if not rates.is_empty()
            else None
        )
        prior = (
            _asof_value(rates, date.fromordinal(d.toordinal() - 3))
            if not rates.is_empty()
            else None
        )
        rows.append(
            {
                "event_time": _event_time_utc(d, fomc.release_et, tz),
                "event_type": "fomc",
                "ref_period": d,
                "actual_pit": actual,
                "prior": prior,
                "surprise": None,
                "source": f"FRED:{fomc.rate_series_id}",
                "fred_series_id": fomc.rate_series_id,
                "ingest_ts": ingest_ts,
                "macro_version": cfg.macro_version,
            }
        )
    logger.info(f"[fomc] {len(rows)} scheduled meetings ({fomc.rate_series_id} outcome)")
    return pl.DataFrame(rows, schema=_SCHEMA) if rows else _empty()


def _empty() -> pl.DataFrame:
    return pl.DataFrame(schema=_SCHEMA)


def build_events(cfg: MacroConfig, api_key: str) -> pl.DataFrame:
    """Assemble the full macro_events frame (all data releases + FOMC), sorted."""
    tz = ZoneInfo(cfg.timezone)
    ingest_ts = datetime.now(UTC)
    data = _data_events(cfg, api_key, tz, ingest_ts)
    fomc = _fomc_events(cfg, api_key, tz, ingest_ts)
    frame = pl.concat([data.select(list(_SCHEMA)), fomc.select(list(_SCHEMA))])
    return frame.sort(["event_time", "event_type"])


# --------------------------------------------------------------------------- #
# Storage (own idempotent writer; partitioned by event-time date, no symbol)
# --------------------------------------------------------------------------- #
def _root() -> Path:
    return Settings().data_dir / _DATASET


def partition_glob() -> str:
    return str(_root() / "date=*" / "*.parquet")


def _existing_keys(part_dir: Path) -> set[tuple[str, datetime]]:
    if not part_dir.exists() or not any(part_dir.glob("*.parquet")):
        return set()
    seen = pl.read_parquet(part_dir / "*.parquet").select("event_type", "event_time")
    return set(zip(seen["event_type"].to_list(), seen["event_time"].to_list(), strict=True))


def write_macro_events(frame: pl.DataFrame) -> int:
    """Append ``frame`` to the macro_events lake; idempotent on (event_type, event_time)."""
    if frame.is_empty():
        return 0
    frame = frame.select(list(_SCHEMA)).with_columns(pl.col("event_time").dt.date().alias("_date"))
    written = 0
    for (d,), group in frame.group_by(["_date"], maintain_order=True):
        part_dir = _root() / f"date={d}"
        seen = _existing_keys(part_dir)
        if seen:
            keys = list(
                zip(group["event_type"].to_list(), group["event_time"].to_list(), strict=True)
            )
            keep = [k not in seen for k in keys]
            group = group.filter(pl.Series(keep))
        if group.is_empty():
            continue
        part_dir.mkdir(parents=True, exist_ok=True)
        group.drop("_date").write_parquet(
            part_dir / f"part-{uuid4().hex}.parquet", compression="zstd"
        )
        written += group.height
    return written


def read_macro_events(
    start: datetime | None = None,
    end: datetime | None = None,
    *,
    store: DuckStore | None = None,
) -> pl.DataFrame:
    """All macro events with ``event_time`` in ``[start, end]`` (UTC), latest-ingest wins.

    Returns an empty (correctly-typed) frame if nothing has been ingested. Read via
    the DuckDB store so the dedup is pushed down; mirrors the feature reader.
    """
    files = _glob(partition_glob())
    if not files:
        return _empty()
    own = store is None
    store = store or DuckStore()
    try:
        lo = start or datetime(2000, 1, 1, tzinfo=UTC)
        hi = end or datetime(2100, 1, 1, tzinfo=UTC)
        return store.con.execute(
            f"""
            SELECT * EXCLUDE (rn) FROM (
                SELECT *, row_number() OVER (
                    PARTITION BY event_type, event_time ORDER BY ingest_ts DESC
                ) AS rn
                FROM read_parquet('{partition_glob()}', hive_partitioning=false)
                WHERE event_time >= ? AND event_time <= ?
            ) WHERE rn = 1
            ORDER BY event_time, event_type
            """,
            [lo, hi],
        ).pl()
    finally:
        if own:
            store.close()


# --------------------------------------------------------------------------- #
# Top-level ingest (key-gated)
# --------------------------------------------------------------------------- #
def ingest(cfg: MacroConfig | None = None, settings: Settings | None = None) -> dict[str, int]:
    """Ingest the macro-event calendar to the lake. Key-gated; returns rows written per type.

    With ``OPTIONS_FRED_API_KEY`` unset this **no-ops** with a clear message and
    returns ``{}`` — the rest of the system is unaffected.
    """
    cfg = cfg or MacroConfig.load()
    settings = settings or Settings()
    if settings.fred_api_key is None:
        logger.warning(
            "OPTIONS_FRED_API_KEY is not set — macro ingestion is a no-op. "
            "Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html "
            "and set OPTIONS_FRED_API_KEY (env/.env) or bridge it from `pass`."
        )
        return {}
    api_key = settings.fred_api_key.get_secret_value()
    frame = build_events(cfg, api_key)
    written = write_macro_events(frame)
    by_type = dict(frame.group_by("event_type").len().iter_rows()) if not frame.is_empty() else {}
    logger.info(
        f"macro ingest: {written} new rows written (built {frame.height}); by_type={by_type}"
    )
    return {
        "_written": written,
        "_built": frame.height,
        **{str(k): int(v) for k, v in by_type.items()},
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="macro.ingest", description=__doc__)
    p.add_argument("--show", action="store_true", help="print a sample of ingested events and exit")
    args = p.parse_args(argv)

    cfg = MacroConfig.load()
    print(f"macro_version={cfg.macro_version} events={list(cfg.events)} + fomc")
    result = ingest(cfg)
    if not result:
        print("  (no FRED key — nothing ingested)")
        return 0
    print(f"  built={result.get('_built', 0)} written={result.get('_written', 0)}")
    for k, v in sorted(result.items()):
        if not k.startswith("_"):
            print(f"    {k}: {v}")
    if args.show:
        ev = read_macro_events()
        print(ev.tail(12))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
