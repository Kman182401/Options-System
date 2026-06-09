"""Databento MBP-1 ingestion -> dollar bars -> Parquet lake. Hard budget guard.

Streams the cheap top-of-book (``mbp-1`` = best bid/offer updates + trades) for
the deep e-mini parents (``ES``, ``NQ`` front-month continuous, volume roll),
reduces each trading day to dollar bars with the single-level order-flow feature
family, and writes them to the lake at::

    data/micro_bars/symbol=<SYM>/date=<YYYY-MM-DD>/part-<uuid>.parquet

stamped ``microstructure_feature_version`` and the source instrument. Idempotent
on ``ts_event`` per partition (re-running never duplicates), mirroring the price
feature store.

COST GUARD (Databento credits are real money — billed per byte):

* **No-ops** (no network, exit 0) when no API key is available.
* Estimates the cost of **every** day chunk with the free ``metadata.get_cost``
  BEFORE downloading it, and tracks a running total.
* **Aborts** (downloads nothing more) the instant the running total + the next
  chunk would exceed ``databento_budget_usd_cap`` (override with ``--cap``). The
  guard is never circumvented by splitting work.
* Logs **actual billable bytes** (``metadata.get_billable_size``) per chunk and
  in total.
* Without ``--confirm`` it only estimates and prints — nothing is downloaded.

RAM: one trading day is downloaded to a temp DBN file, **stream-read** record by
record into the O(1)-memory reducer, then the temp file is deleted.
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
import subprocess
import tempfile
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from datetime import time as dtime
from pathlib import Path
from typing import cast
from uuid import uuid4
from zoneinfo import ZoneInfo

import polars as pl

from config.settings import Settings

from ..common.logging import get_logger
from ..data.store import DuckStore
from .bars import BookEvent, assemble_features, build_dollar_bars, feature_names, from_records
from .config import Instrument, MicrostructureConfig

logger = get_logger(__name__)

_DATASET = "micro_bars"


class BudgetExceededError(RuntimeError):
    """Raised when a planned download would exceed ``databento_budget_usd_cap``."""


def check_budget(running_usd: float, chunk_usd: float, cap_usd: float) -> None:
    """Guard one chunk against the cap. Raises :class:`BudgetExceededError` if the
    running total plus this chunk would exceed ``cap_usd`` (pure; unit-tested)."""
    projected = running_usd + chunk_usd
    if projected > cap_usd:
        raise BudgetExceededError(
            f"next chunk would bring spend to ${projected:.2f}, over cap ${cap_usd:.2f}"
        )


# --- secret sourcing -------------------------------------------------------- #


# pass entries tried in order: the live key first, then a reserved backup slot.
# The original free key (databento/api_key) ran out of funds and was removed
# 2026-06-08; api_key_2 is the live one. To roll in the next backup when api_key_2
# is depleted, just `pass insert databento/api_key_3` — no code change needed (a
# missing path is skipped).
_PASS_KEY_PATHS = ("databento/api_key_2", "databento/api_key_3")


def _get_api_key(settings: Settings) -> str | None:
    """Databento key, sourced securely. Prefers ``Settings`` (env/.env, the
    existing convention); falls back to the ``pass`` store so the key need never be
    written to ``.env``. Never logged, never on argv."""
    if settings.databento_api_key is not None:
        return settings.databento_api_key.get_secret_value()
    for path in _PASS_KEY_PATHS:
        try:
            out = subprocess.run(["pass", "show", path], capture_output=True, text=True, check=True)
            key = out.stdout.strip()
            if key:
                return key
        except Exception:  # noqa: BLE001 - missing entry -> try the next path
            continue
    return None


# --- time helpers ----------------------------------------------------------- #


def _trading_days(start: date, end: date) -> list[date]:
    """Weekday calendar days in ``[start, end)`` (holidays just return no records)."""
    out: list[date] = []
    cur = start
    while cur < end:
        if cur.weekday() < 5:
            out.append(cur)
        cur += timedelta(days=1)
    return out


def _fetch_window(cfg: MicrostructureConfig, day: date) -> tuple[datetime, datetime]:
    """UTC ``[start, end)`` to fetch for one day. RTH-only fetches just the RTH
    window (≈1/3 the data/credits of full Globex); otherwise the whole UTC day."""
    if not cfg.session.rth_only:
        return (
            datetime.combine(day, dtime(0), UTC),
            datetime.combine(day + timedelta(days=1), dtime(0), UTC),
        )
    et = ZoneInfo(cfg.session.tz)
    o = datetime.combine(
        day, dtime(cfg.session.rth_open_min // 60, cfg.session.rth_open_min % 60), et
    )
    c = datetime.combine(
        day, dtime(cfg.session.rth_close_min // 60, cfg.session.rth_close_min % 60), et
    )
    return o.astimezone(UTC), c.astimezone(UTC)


# --- cost estimate ---------------------------------------------------------- #


def _chunk_cost(client, cfg: MicrostructureConfig, inst: Instrument, day: date) -> float:
    """Free USD cost estimate for one day's fetch window (matches what we download)."""
    lo, hi = _fetch_window(cfg, day)
    return float(
        client.metadata.get_cost(
            dataset=cfg.dataset,
            symbols=[inst.continuous_symbol],
            schema=cfg.schema_,
            start=lo.isoformat(),
            end=hi.isoformat(),
            stype_in="continuous",
        )
    )


def _chunk_bytes(client, cfg: MicrostructureConfig, inst: Instrument, day: date) -> int:
    """Free billable-bytes estimate for one day's fetch window."""
    lo, hi = _fetch_window(cfg, day)
    return int(
        client.metadata.get_billable_size(
            dataset=cfg.dataset,
            symbols=[inst.continuous_symbol],
            schema=cfg.schema_,
            start=lo.isoformat(),
            end=hi.isoformat(),
            stype_in="continuous",
        )
    )


def estimate_cost(
    client, cfg: MicrostructureConfig, symbols: list[str], start: date, end: date
) -> dict[str, float]:
    """Per-symbol USD estimate for the requested window, summed over day chunks
    using the exact fetch windows we would download (so it matches actual spend)."""
    out: dict[str, float] = {}
    for sym in symbols:
        inst = cfg.instrument(sym)
        out[sym] = sum(_chunk_cost(client, cfg, inst, d) for d in _trading_days(start, end))
    return out


# --- storage ---------------------------------------------------------------- #


def _root() -> Path:
    return Settings().data_dir / _DATASET


def partition_glob(symbol: str | None = None) -> str:
    sym = "*" if symbol is None else f"symbol={symbol}"
    return str(_root() / sym / "date=*" / "*.parquet")


def _existing_keys(part_dir: Path) -> set:
    if not part_dir.exists() or not any(part_dir.glob("*.parquet")):
        return set()
    keys = cast(
        "pl.DataFrame", pl.scan_parquet(part_dir / "*.parquet").select("ts_event").collect()
    )
    return set(keys["ts_event"])


def write_micro_bars(df: pl.DataFrame, symbol: str) -> int:
    """Append a symbol's bar frame; idempotent on ``ts_event`` per date partition."""
    if df.is_empty():
        return 0
    df = df.with_columns(
        pl.lit(datetime.now(UTC)).cast(pl.Datetime("us", "UTC")).alias("ts_ingest"),
        pl.col("ts_event").dt.date().alias("_date"),
    )
    written = 0
    for (day,), group in df.group_by(["_date"], maintain_order=True):
        part_dir = _root() / f"symbol={symbol}" / f"date={day}"
        seen = _existing_keys(part_dir)
        new = group.filter(~pl.col("ts_event").is_in(list(seen))) if seen else group
        if new.is_empty():
            continue
        part_dir.mkdir(parents=True, exist_ok=True)
        new.drop("_date").write_parquet(
            part_dir / f"part-{uuid4().hex}.parquet", compression="zstd"
        )
        written += new.height
    return written


def read_micro_bars(
    symbol: str, start: datetime, end: datetime, store: DuckStore | None = None
) -> pl.DataFrame:
    """Bar rows for ``symbol`` in ``[start, end]`` (UTC, inclusive); latest-ingest wins."""
    from glob import glob as _glob

    own = store is None
    store = store or DuckStore()
    try:
        if not _glob(partition_glob(symbol)):
            return pl.DataFrame()
        glob_str = partition_glob(symbol)
        return store.con.execute(
            f"""
            SELECT * EXCLUDE (rn) FROM (
                SELECT *, row_number() OVER (PARTITION BY ts_event ORDER BY ts_ingest DESC) AS rn
                FROM read_parquet('{glob_str}', hive_partitioning=false)
                WHERE ts_event >= ? AND ts_event <= ?
            ) WHERE rn = 1
            ORDER BY ts_event
            """,
            [start, end],
        ).pl()
    finally:
        if own:
            store.close()


# --- one-day reduce --------------------------------------------------------- #


def _resolve_contracts(store, instrument_ids: set[int], day: date) -> dict[int, str]:
    """Best-effort instrument_id -> raw contract symbol via Databento's symbology
    map. Falls back to ``id<n>`` when unavailable; never raises."""
    try:
        from databento.common.symbology import InstrumentMap

        imap = InstrumentMap()
        imap.insert_metadata(store.metadata)
        out: dict[int, str] = {}
        for iid in instrument_ids:
            sym = imap.resolve(iid, day)
            if sym:
                out[iid] = sym
        return out
    except Exception as exc:  # noqa: BLE001 - mapping is a nicety, not required
        logger.debug(f"contract symbol resolution unavailable: {exc}")
        return {}


@dataclass(frozen=True)
class DayWorkUnit:
    """One **independent** (symbol, session-day) reduction unit — the atom of
    parallelism. Exactly one record source is set:

    * ``dbn_path`` — a local DBN file (the production path; the worker stream-reads
      it, so only the path crosses the process boundary, never the raw records);
    * ``events`` — an in-memory tuple of :class:`BookEvent` (the synthetic /
      offline-test path).

    A unit is one **whole** session-day stream: it is never split across workers
    (order-flow continuity inside a session matters), and two days are never merged
    into one unit (the reducer already severs flow at session / contract seams).
    Frozen + only-picklable fields, so it ships safely to a worker process.
    """

    symbol: str
    session_date: date
    instrument: Instrument
    cfg: MicrostructureConfig
    dbn_path: str | None = None
    events: tuple[BookEvent, ...] | None = None
    contract_map: dict[int, str] | None = None

    def __post_init__(self) -> None:
        if (self.dbn_path is None) == (self.events is None):
            raise ValueError("DayWorkUnit needs exactly one of dbn_path / events")


def _download_day(client, cfg: MicrostructureConfig, inst: Instrument, day: date) -> Path:
    """Download one trading day for one instrument to a temp DBN file; return its
    path. This is the **only** network step (Databento ``get_range``), with retry /
    backoff. Raises on persistent failure. The caller owns the temp file lifecycle
    (delete it after reducing). Kept separate from the reducer so the CPU-heavy
    reduction can run in a worker process while this stays serial + budget-gated.
    """
    lo, hi = _fetch_window(cfg, day)
    tmp = Path(tempfile.gettempdir()) / f"mbp1-{inst.symbol}-{day}-{uuid4().hex}.dbn.zst"
    last: Exception | None = None
    for attempt in range(1, cfg.ingest.retries + 1):
        try:
            data = client.timeseries.get_range(
                dataset=cfg.dataset,
                symbols=[inst.continuous_symbol],
                schema=cfg.schema_,
                start=lo.isoformat(),
                end=hi.isoformat(),
                stype_in="continuous",
            )
            data.to_file(str(tmp))
            last = None
            break
        except Exception as exc:  # noqa: BLE001 - transient network/rate-limit, retried
            last = exc
            if attempt < cfg.ingest.retries:
                logger.warning(f"{inst.symbol} {day} fetch failed ({exc}); retry {attempt}")
                time.sleep(cfg.ingest.backoff_s * attempt)
    if last is not None:
        raise last
    return tmp


def reduce_work_unit(unit: DayWorkUnit) -> pl.DataFrame:
    """Reduce ONE independent (symbol, session-day) unit to its assembled feature
    frame. This is the **process-pool worker**: module-level + picklable, network-
    free (the ``dbn_path`` branch only *reads* a local file — no Databento API
    call), and it never touches an Executor/Future. It calls the same unchanged
    reducer the serial path does, so its output is bit-identical to serial.
    """
    inst = unit.instrument
    cfg = unit.cfg
    if unit.events is not None:
        raw_bars = build_dollar_bars(unit.events, instrument=inst, session=cfg.session)
        cmap: dict[int, str] = unit.contract_map or {}
    else:
        import databento as db

        assert unit.dbn_path is not None  # guaranteed by __post_init__
        store = db.DBNStore.from_file(unit.dbn_path)
        raw_bars = build_dollar_bars(
            from_records(store, inst), instrument=inst, session=cfg.session
        )
        cmap = (
            unit.contract_map
            if unit.contract_map is not None
            else _resolve_contracts(
                store, {b["instrument_id"] for b in raw_bars}, unit.session_date
            )
        )
    return assemble_features(raw_bars, symbol=inst.symbol, cfg=cfg, contract_map=cmap)


def ingest_day(client, cfg: MicrostructureConfig, inst: Instrument, day: date) -> pl.DataFrame:
    """Download one trading day for one instrument, reduce to the bar+feature frame.

    The **serial reference path**: download the DBN to a temp file, stream-read it
    into the O(1) reducer, delete the temp file. Output is identical to the parallel
    path — both funnel through :func:`reduce_work_unit`. Returns an (assembled)
    frame; empty if the day had no records.
    """
    tmp = _download_day(client, cfg, inst, day)
    try:
        return reduce_work_unit(
            DayWorkUnit(
                symbol=inst.symbol,
                session_date=day,
                instrument=inst,
                cfg=cfg,
                dbn_path=str(tmp),
            )
        )
    finally:
        tmp.unlink(missing_ok=True)


# --- parallel reduction across independent (symbol, session-day) units ------ #

# Beyond this many processes MBP-1 day reduction is bound by DBN I/O + result
# pickling, not CPU, so more workers only add fork + serialisation overhead.
_MAX_AUTO_WORKERS = 8


def _auto_workers(n_tasks: int) -> int:
    """Conservative auto worker count: ``min(cpus, n_tasks, _MAX_AUTO_WORKERS)``,
    floored at 1. Uses ``os.process_cpu_count`` (affinity-aware, Python ≥3.13) when
    present, else ``os.cpu_count`` (this host is on 3.12)."""
    cpu_fn = getattr(os, "process_cpu_count", None) or os.cpu_count
    cpu = cpu_fn() or 1
    return max(1, min(cpu, max(1, n_tasks), _MAX_AUTO_WORKERS))


def reduce_units(units: Sequence[DayWorkUnit], *, workers: int = 1) -> list[pl.DataFrame]:
    """Reduce independent units to per-unit feature frames, returned in **input
    order** — deterministic and independent of worker completion order.

    * ``workers <= 1`` → the serial reference path (direct :func:`reduce_work_unit`).
    * ``workers > 1`` → a :class:`~concurrent.futures.ProcessPoolExecutor` across
      units; each result is placed back by its input index, so the output never
      depends on which worker finished first. Memory stays bounded: at most
      ``workers`` units (each one session-day) are reduced concurrently.

    A worker exception propagates unchanged (via ``Future.result()``).
    """
    n = len(units)
    if n == 0:
        return []
    if workers <= 1:
        return [reduce_work_unit(u) for u in units]
    workers = min(workers, n)
    results: list[pl.DataFrame | None] = [None] * n
    # Use 'spawn', NOT the Linux-default 'fork': the reducer calls into polars, whose
    # Rayon threadpool deadlocks when inherited across a fork() in a worker. Spawn
    # starts each worker as a fresh interpreter (no inherited threadpool, no
    # deadlock); the small startup cost is negligible against per-day reduction work.
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
        fut_to_idx = {ex.submit(reduce_work_unit, u): i for i, u in enumerate(units)}
        for fut in as_completed(fut_to_idx):
            results[fut_to_idx[fut]] = fut.result()
    out: list[pl.DataFrame] = []
    for r in results:
        assert r is not None  # every index filled, or fut.result() raised above
        out.append(r)
    return out


def reduce_units_to_frame(
    units: Sequence[DayWorkUnit],
    *,
    workers: int = 1,
    cfg: MicrostructureConfig | None = None,
) -> pl.DataFrame:
    """Reduce units and concatenate into ONE feature frame in a **canonical** order
    that is independent of both worker-completion order and unit input order: sorted
    by ``(symbol, ts_event, con_id, ts_open)`` (a full ordering — no ties). Useful
    as a single equivalence surface; the lake writer still partitions per
    (symbol, date) exactly as before. Empty when every unit reduced to no bars."""
    frames = reduce_units(units, workers=workers)
    nonempty = [f for f in frames if f.height > 0]
    if not nonempty:
        c = cfg if cfg is not None else (units[0].cfg if units else None)
        if c is None:
            raise ValueError("reduce_units_to_frame: no units and no cfg to build empty schema")
        return assemble_features([], symbol="", cfg=c)
    combined = pl.concat(nonempty, how="vertical")
    return combined.sort(["symbol", "ts_event", "con_id", "ts_open"])


# --- driver ----------------------------------------------------------------- #


def run_ingest(
    cfg: MicrostructureConfig,
    symbols: list[str],
    start: date,
    end: date,
    *,
    api_key: str,
    cap: float,
    workers: int = 1,
) -> dict:
    """Ingest ``[start, end)`` for ``symbols`` under the budget cap. Estimates each
    day chunk before downloading and aborts (cleanly) if the cap would be breached.

    ``workers`` controls the **reduction** stage only: ``1`` (default) is the
    unchanged serial path; ``>1`` reduces downloaded days in parallel processes via
    :func:`_run_ingest_parallel`. Budget gating + downloads stay strictly sequential
    in both — the cap is a running total that must be honoured in order, and the
    per-day reduced output is identical either way.

    Returns ``{symbols: {...}, "_totals": {...}}``.
    """
    if workers > 1:
        return _run_ingest_parallel(
            cfg, symbols, start, end, api_key=api_key, cap=cap, workers=workers
        )

    import databento as db

    client = db.Historical(api_key)
    days = _trading_days(start, end)
    per_symbol: dict[str, dict] = {}
    running_usd = 0.0
    running_bytes = 0
    aborted = False
    for sym in symbols:
        inst = cfg.instrument(sym)
        s = {
            "rows_written": 0,
            "days_requested": len(days),
            "days_with_data": 0,
            "dollar_threshold": inst.dollar_threshold,
            "est_usd": 0.0,
            "billable_bytes": 0,
        }
        for day in days:
            chunk_usd = _chunk_cost(client, cfg, inst, day)
            try:
                check_budget(running_usd, chunk_usd, cap)
            except BudgetExceededError as exc:
                logger.warning(f"budget cap reached before {sym} {day}: {exc}")
                aborted = True
                break
            running_usd += chunk_usd
            s["est_usd"] += chunk_usd
            chunk_bytes = _chunk_bytes(client, cfg, inst, day)
            running_bytes += chunk_bytes
            s["billable_bytes"] += chunk_bytes
            df = ingest_day(client, cfg, inst, day)
            if df.is_empty():
                logger.info(f"{sym} {day}: no bars")
                continue
            n = write_micro_bars(df, sym)
            s["rows_written"] += n
            s["days_with_data"] += 1
            logger.info(
                f"{sym} {day}: +{n} bars (running ${running_usd:.2f}, {running_bytes / 1e6:.1f} MB)"
            )
        per_symbol[sym] = s
        if aborted:
            break
    return {
        **per_symbol,
        "_totals": {
            "est_usd": running_usd,
            "billable_bytes": running_bytes,
            "cap_usd": cap,
            "aborted": aborted,
        },
    }


def _run_ingest_parallel(
    cfg: MicrostructureConfig,
    symbols: list[str],
    start: date,
    end: date,
    *,
    api_key: str,
    cap: float,
    workers: int,
) -> dict:
    """Parallel-reduce variant of :func:`run_ingest`. Budget gating + downloads stay
    STRICTLY SEQUENTIAL in the same (symbol-major, day-ascending) order as the serial
    path — only the CPU-heavy per-day reduction is parallelised, in **waves** of
    ``workers`` days so at most ``workers`` temp DBN files / day-frames are ever in
    flight (memory + disk stay bounded). The reduced output and lake writes are
    identical to serial; only wall-clock differs.

    The returned stats dict pre-includes every requested symbol (zero-filled) even
    past a budget abort — a cosmetic difference from serial (which omits unreached
    symbols); the bars written to the lake are identical.
    """
    import databento as db

    client = db.Historical(api_key)
    days = _trading_days(start, end)
    per_symbol: dict[str, dict] = {
        sym: {
            "rows_written": 0,
            "days_requested": len(days),
            "days_with_data": 0,
            "dollar_threshold": cfg.instrument(sym).dollar_threshold,
            "est_usd": 0.0,
            "billable_bytes": 0,
        }
        for sym in symbols
    }
    # Flat task list in the exact serial order, so budget gating aborts at the same
    # cumulative point regardless of worker count.
    tasks = [(sym, day) for sym in symbols for day in days]
    running_usd = 0.0
    running_bytes = 0
    aborted = False
    i = 0
    while i < len(tasks) and not aborted:
        wave: list[DayWorkUnit] = []
        while len(wave) < workers and i < len(tasks):
            sym, day = tasks[i]
            inst = cfg.instrument(sym)
            chunk_usd = _chunk_cost(client, cfg, inst, day)
            try:
                check_budget(running_usd, chunk_usd, cap)
            except BudgetExceededError as exc:
                logger.warning(f"budget cap reached before {sym} {day}: {exc}")
                aborted = True
                break
            running_usd += chunk_usd
            per_symbol[sym]["est_usd"] += chunk_usd
            chunk_bytes = _chunk_bytes(client, cfg, inst, day)
            running_bytes += chunk_bytes
            per_symbol[sym]["billable_bytes"] += chunk_bytes
            tmp = _download_day(client, cfg, inst, day)
            wave.append(
                DayWorkUnit(
                    symbol=sym,
                    session_date=day,
                    instrument=inst,
                    cfg=cfg,
                    dbn_path=str(tmp),
                )
            )
            i += 1
        if not wave:
            break
        frames = reduce_units(wave, workers=workers)
        for unit, df in zip(wave, frames, strict=True):
            try:
                if df.is_empty():
                    logger.info(f"{unit.symbol} {unit.session_date}: no bars")
                    continue
                n = write_micro_bars(df, unit.symbol)
                per_symbol[unit.symbol]["rows_written"] += n
                per_symbol[unit.symbol]["days_with_data"] += 1
                logger.info(
                    f"{unit.symbol} {unit.session_date}: +{n} bars "
                    f"(running ${running_usd:.2f}, {running_bytes / 1e6:.1f} MB)"
                )
            finally:
                if unit.dbn_path is not None:
                    Path(unit.dbn_path).unlink(missing_ok=True)
    return {
        **per_symbol,
        "_totals": {
            "est_usd": running_usd,
            "billable_bytes": running_bytes,
            "cap_usd": cap,
            "aborted": aborted,
        },
    }


def log_ingest_stats(cfg: MicrostructureConfig, stats: dict, start: date, end: date) -> str | None:
    """Log dataset-level stats (incl. cost) to the local MLflow file store (best-effort)."""
    try:
        import os

        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
        import mlflow
    except Exception as exc:  # noqa: BLE001 - tracking optional
        logger.warning(f"mlflow unavailable ({exc}); skipping ingest tracking")
        return None
    totals = stats.get("_totals", {})
    syms = [k for k in stats if k != "_totals"]
    mlflow.set_tracking_uri((Settings().data_dir / "mlruns").as_uri())
    mlflow.set_experiment("microstructure-data")
    with mlflow.start_run(run_name=f"ingest-mbp1-{start}_{end}") as run:
        mlflow.log_params(
            {
                "microstructure_feature_version": cfg.microstructure_feature_version,
                "dataset": cfg.dataset,
                "schema": cfg.schema_,
                "window_start": start.isoformat(),
                "window_end": end.isoformat(),
                "rth_only": cfg.session.rth_only,
                "n_features": len(feature_names(cfg)),
                "symbols": ",".join(syms),
                "budget_cap_usd": cfg.databento_budget_usd_cap,
            }
        )
        mlflow.log_metrics(
            {
                "total_est_usd": float(totals.get("est_usd", 0.0)),
                "total_billable_bytes": float(totals.get("billable_bytes", 0)),
            }
        )
        for sym in syms:
            mlflow.log_metrics({f"{sym}_rows_written": float(stats[sym]["rows_written"])})
        mlflow.log_dict({"feature_list": feature_names(cfg), "stats": stats}, "ingest.json")
        return run.info.run_id


def _workers_arg(v: str) -> int | str:
    """argparse type for ``--workers``: a positive int, or the literal ``auto``."""
    if v.strip().lower() == "auto":
        return "auto"
    try:
        iv = int(v)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"--workers must be a positive int or 'auto', got {v!r}"
        ) from exc
    if iv < 1:
        raise argparse.ArgumentTypeError("--workers must be >= 1")
    return iv


def _parse(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="microstructure.ingest", description=__doc__)
    p.add_argument("--symbols", nargs="+", default=None, help="default: all config instruments")
    p.add_argument("--start", help="YYYY-MM-DD (UTC); default: config window.start")
    p.add_argument("--end", help="YYYY-MM-DD (UTC, exclusive); default: config window.end")
    p.add_argument("--confirm", action="store_true", help="actually download (consumes credits)")
    p.add_argument("--cap", type=float, default=None, help="override databento_budget_usd_cap")
    p.add_argument(
        "--workers",
        type=_workers_arg,
        default=1,
        help="reduction processes: 1 (default, serial-equivalent) | N>1 (parallel) | 'auto'",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)
    cfg = MicrostructureConfig.load()
    settings = Settings()

    symbols = args.symbols or cfg.symbols()
    start = date.fromisoformat(args.start) if args.start else cfg.window.start
    end = date.fromisoformat(args.end) if args.end else cfg.window.end
    cap = args.cap if args.cap is not None else cfg.databento_budget_usd_cap
    n_tasks = len(symbols) * len(_trading_days(start, end))
    workers = _auto_workers(n_tasks) if args.workers == "auto" else int(args.workers)

    api_key = _get_api_key(settings)
    if api_key is None:
        print(
            "No Databento API key (Settings or `pass databento/api_key`) — ingestion "
            "disabled (no-op). No network call was made."
        )
        return 0

    import databento as db

    client = db.Historical(api_key)
    est = estimate_cost(client, cfg, symbols, start, end)
    total = sum(est.values())
    print(f"Estimated cost {cfg.schema_} {symbols} {start}..{end} (RTH={cfg.session.rth_only}):")
    for sym, c in est.items():
        print(f"  {sym}: ${c:.2f}")
    print(f"  TOTAL: ${total:.2f}  |  budget cap ${cap:.2f}")

    if total > cap:
        print(f"REFUSING: estimate ${total:.2f} exceeds cap ${cap:.2f}. Narrow the window.")
        return 2
    if not args.confirm:
        print(
            f"Dry run — nothing downloaded. Re-run with --confirm to ingest "
            f"(consumes credits). [workers={workers}]"
        )
        return 0

    stats = run_ingest(cfg, symbols, start, end, api_key=api_key, cap=cap, workers=workers)
    run_id = log_ingest_stats(cfg, stats, start, end)
    totals = stats["_totals"]
    for sym in symbols:
        s = stats.get(sym, {})
        print(
            f"  {sym}: +{s.get('rows_written', 0):,} bars over "
            f"{s.get('days_with_data', 0)}/{s.get('days_requested', 0)} days  "
            f"(${s.get('est_usd', 0.0):.2f}, {s.get('billable_bytes', 0) / 1e6:.1f} MB)"
        )
    print(
        f"TOTAL spend ${totals['est_usd']:.2f} / cap ${totals['cap_usd']:.2f}  "
        f"({totals['billable_bytes'] / 1e6:.1f} MB billable)  aborted={totals['aborted']}"
    )
    print(f"MLflow run: {run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
