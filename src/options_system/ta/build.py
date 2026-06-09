"""Build, store, and retrieve TA feature tables — versioned, idempotent, leak-free.

This is the ``feature_version = v2`` lake, a sibling of the v1 price lake. Features
are computed over the **full** continuous history of each symbol (so the EWM/rolling
state is correctly seeded and the value at ``t`` never depends on where a build
window starts), then written to the lake at::

    data/ta_features/symbol=<SYM>/date=<YYYY-MM-DD>/part-<uuid>.parquet

partitioned, zstd, idempotent on the natural key ``ts_event``. Re-running never
duplicates. ``--start/--end`` bound only what is *written*, never what is computed,
so stored values are identical regardless of the write window.

Everything is computed locally over the existing ``bars_1m`` data — no Databento
calls, no cost guard, no external TA service. Retrieval (:func:`read_ta`,
:func:`ta_asof`) goes through the DuckDB store and its ``asof_join``, preserving
the no-look-ahead guarantee: a bar at ``t`` only ever sees a feature row with
``ts_event <= t``.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from glob import glob as _glob
from pathlib import Path
from typing import cast
from uuid import uuid4

import polars as pl

from config.settings import Settings

from ..data.store import DuckStore
from .compute import compute_ta, ta_feature_names
from .config import TaConfig

_DATASET = "ta_features"
_WIDE_START = datetime(2000, 1, 1, tzinfo=UTC)
_WIDE_END = datetime(2100, 1, 1, tzinfo=UTC)


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


def _write_symbol(frame: pl.DataFrame, symbol: str) -> int:
    """Append a symbol's TA frame; idempotent on ts_event per date partition."""
    if frame.is_empty():
        return 0
    frame = frame.with_columns(pl.col("ts_event").dt.date().alias("_date"))
    written = 0
    for (date,), group in frame.group_by(["_date"], maintain_order=True):
        part_dir = _root() / f"symbol={symbol}" / f"date={date}"
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


def _continuous(store: DuckStore, symbol: str) -> pl.DataFrame:
    """Full back-adjusted continuous (outright) series for a symbol."""
    return store.get_bars(symbol, _WIDE_START, _WIDE_END, freq="1m", continuous=True)


def build_ta(
    symbols: list[str],
    *,
    write_start: datetime | None = None,
    write_end: datetime | None = None,
    cfg: TaConfig | None = None,
    store: DuckStore | None = None,
) -> dict[str, int]:
    """Compute TA features over full history for each symbol and write the lake tables.

    ``write_start/write_end`` bound only the rows persisted (computation always uses
    full history). Returns ``{symbol: rows_written}``.
    """
    cfg = cfg or TaConfig.load()
    own_store = store is None
    store = store or DuckStore()
    try:
        ts_ingest = datetime.now(UTC)
        result: dict[str, int] = {}
        for sym in symbols:
            feats = compute_ta(_continuous(store, sym), cfg)
            feats = feats.with_columns(
                pl.lit(ts_ingest).cast(pl.Datetime("us", "UTC")).alias("ts_ingest"),
                pl.lit(sym).alias("symbol"),
            )
            if write_start is not None:
                feats = feats.filter(pl.col("ts_event") >= write_start)
            if write_end is not None:
                feats = feats.filter(pl.col("ts_event") <= write_end)
            # canonical column order: keys first, then features, then flags
            ordered = [
                "ts_event",
                "ts_ingest",
                "symbol",
                "session",
                *ta_feature_names(cfg),
                "degraded",
                "ta_feature_version",
            ]
            result[sym] = _write_symbol(feats.select(ordered), sym)
        return result
    finally:
        if own_store:
            store.close()


# --- retrieval (point-in-time correct) ------------------------------------- #


def read_ta(
    symbol: str, start: datetime, end: datetime, store: DuckStore | None = None
) -> pl.DataFrame:
    """TA feature rows for ``symbol`` in ``[start, end]`` (inclusive, UTC), latest-ingest wins."""
    own = store is None
    store = store or DuckStore()
    try:
        files = _glob(partition_glob(symbol))
        if not files:
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


def ta_asof(bars: pl.DataFrame, symbol: str, *, store: DuckStore | None = None) -> pl.DataFrame:
    """Attach, to each row of ``bars`` (must have ``ts_event``), the latest TA feature
    row with ``feature.ts_event <= bar.ts_event`` — leak-free via the store's ASOF JOIN.
    """
    if bars.is_empty():
        return bars
    own = store is None
    store = store or DuckStore()
    try:
        lo = cast("datetime", bars["ts_event"].min())
        hi = cast("datetime", bars["ts_event"].max())
        feats = read_ta(symbol, lo, hi, store=store)
        if feats.is_empty():
            return bars
        feats = feats.drop("symbol")  # avoid clashing with a left symbol column
        return store.asof_join(bars, feats, on="ts_event")
    finally:
        if own:
            store.close()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="ta.build", description=__doc__)
    p.add_argument("--symbols", nargs="+", default=None, help="default: settings.record_symbols")
    p.add_argument("--start", help="write-window start YYYY-MM-DD (UTC); compute is always full")
    p.add_argument("--end", help="write-window end YYYY-MM-DD (UTC)")
    args = p.parse_args(argv)

    symbols = args.symbols or Settings().record_symbols

    def _parse(d: str | None) -> datetime | None:
        return datetime.fromisoformat(d).replace(tzinfo=UTC) if d else None

    cfg = TaConfig.load()
    print(f"ta_feature_version={cfg.ta_feature_version} symbols={symbols}")
    written = build_ta(symbols, write_start=_parse(args.start), write_end=_parse(args.end), cfg=cfg)
    for sym, n in written.items():
        print(f"  {sym}: +{n:,} TA feature rows written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
