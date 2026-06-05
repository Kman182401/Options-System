"""The Parquet lake: canonical on-disk schemas + an append-only, idempotent writer.

Layout (under ``settings.data_dir``, gitignored)::

    data/<dataset>/symbol=<SYM>/date=<YYYY-MM-DD>/part-<uuid>.parquet

Datasets: ``bars_1m``, ``bars_5s``, ``quotes_l1``, ``trades``, ``roll_events``.

Two non-negotiables, enforced here:

* **Dual timestamps, always UTC.** Every market-data row carries ``ts_event``
  (exchange/publish time) and ``ts_ingest`` (when we received it). This is the
  foundation for point-in-time-correct retrieval — no look-ahead, ever.
* **Append-only + idempotent.** The writer never overwrites an existing
  partition. Re-writing the same rows is a no-op: each dataset has a natural key
  and already-present keys are skipped. Re-recording a day cannot create
  duplicates. (Files accumulate as small parts; ``compact`` merges them.)

The writer does not repair, forward-fill, or synthesize data. Gaps stay gaps.
"""

from __future__ import annotations

from glob import glob as _glob
from pathlib import Path
from uuid import uuid4

import polars as pl

from config.settings import Settings

SCHEMA_VERSION = 1

# Timestamps are microsecond-resolution, timezone-aware UTC.
_TS = pl.Datetime("us", "UTC")

# --- Canonical schemas (column -> Polars dtype), in on-disk order. ---
_BARS: dict[str, pl.DataType] = {
    "ts_event": _TS,
    "ts_ingest": _TS,
    "symbol": pl.Utf8,
    "contract_id": pl.Utf8,
    "con_id": pl.Int64,
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Float64,
    "wap": pl.Float64,
    "n_trades": pl.Int64,
    "session": pl.Utf8,
    "source": pl.Utf8,
    "schema_version": pl.Int32,
}

_QUOTES_L1: dict[str, pl.DataType] = {
    "ts_event": _TS,
    "ts_ingest": _TS,
    "symbol": pl.Utf8,
    "contract_id": pl.Utf8,
    "con_id": pl.Int64,
    "bid": pl.Float64,
    "ask": pl.Float64,
    "bid_size": pl.Float64,
    "ask_size": pl.Float64,
    "last": pl.Float64,
    "last_size": pl.Float64,
    "session": pl.Utf8,
    "source": pl.Utf8,
    "schema_version": pl.Int32,
}

_TRADES: dict[str, pl.DataType] = {
    "ts_event": _TS,
    "ts_ingest": _TS,
    "symbol": pl.Utf8,
    "contract_id": pl.Utf8,
    "con_id": pl.Int64,
    "price": pl.Float64,
    "size": pl.Float64,
    "session": pl.Utf8,
    "source": pl.Utf8,
    "schema_version": pl.Int32,
}

_ROLL_EVENTS: dict[str, pl.DataType] = {
    "ts_event": _TS,
    "ts_ingest": _TS,
    "symbol": pl.Utf8,
    "from_contract_id": pl.Utf8,
    "to_contract_id": pl.Utf8,
    "from_con_id": pl.Int64,
    "to_con_id": pl.Int64,
    "rule": pl.Utf8,
    "adj_factor": pl.Float64,
    "note": pl.Utf8,
    "source": pl.Utf8,
    "schema_version": pl.Int32,
}

# dataset -> (schema, natural-key columns used for dedupe)
DATASETS: dict[str, tuple[dict[str, pl.DataType], tuple[str, ...]]] = {
    "bars_1m": (_BARS, ("contract_id", "ts_event")),
    "bars_5s": (_BARS, ("contract_id", "ts_event")),
    "quotes_l1": (_QUOTES_L1, ("contract_id", "ts_event")),
    "trades": (_TRADES, ("contract_id", "ts_event", "price", "size")),
    "roll_events": (_ROLL_EVENTS, ("symbol", "to_contract_id")),
}

# Columns that every market-data row MUST carry (the writer refuses without them).
_REQUIRED = ("ts_event", "ts_ingest", "symbol")


def schema(dataset: str) -> dict[str, pl.DataType]:
    """Return the canonical Polars schema for ``dataset``."""
    return dict(DATASETS[_check(dataset)][0])


def empty_frame(dataset: str) -> pl.DataFrame:
    """An empty DataFrame with the dataset's canonical schema."""
    return pl.DataFrame(schema=schema(dataset))


def _check(dataset: str) -> str:
    if dataset not in DATASETS:
        raise KeyError(f"unknown dataset {dataset!r}; known: {sorted(DATASETS)}")
    return dataset


def _key_expr(key_cols: tuple[str, ...]) -> pl.Expr:
    """Stable string key from the natural-key columns (for dedupe)."""
    return pl.concat_str([pl.col(c).cast(pl.Utf8) for c in key_cols], separator="\x1f")


class Lake:
    """Append-only Parquet store rooted at ``settings.data_dir``."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root is not None else Settings().data_dir
        # Lazily-populated per-partition set of natural keys already on disk /
        # written this session, so re-writes are cheap and idempotent.
        self._seen: dict[Path, set[str]] = {}

    # -- paths --------------------------------------------------------------
    def dataset_dir(self, dataset: str) -> Path:
        return self.root / _check(dataset)

    def partition_glob(self, dataset: str, symbol: str | None = None) -> str:
        sym = "*" if symbol is None else f"symbol={symbol}"
        return str(self.dataset_dir(dataset) / sym / "date=*" / "*.parquet")

    # -- coercion -----------------------------------------------------------
    def _coerce(self, dataset: str, df: pl.DataFrame) -> pl.DataFrame:
        sch = schema(dataset)
        missing_required = [c for c in _REQUIRED if c not in df.columns]
        if missing_required:
            raise ValueError(f"{dataset}: missing required columns {missing_required}")

        cols: list[pl.Expr] = []
        for name, dtype in sch.items():
            if name in df.columns:
                cols.append(pl.col(name).cast(dtype, strict=False).alias(name))
            elif name == "schema_version":
                cols.append(pl.lit(SCHEMA_VERSION, dtype=dtype).alias(name))
            else:
                cols.append(pl.lit(None, dtype=dtype).alias(name))
        out = df.select(cols)
        # ts columns must be UTC-aware; if a naive datetime slipped in, treat as UTC.
        for ts_col in ("ts_event", "ts_ingest"):
            if out.schema[ts_col].time_zone is None:  # type: ignore[union-attr]
                out = out.with_columns(pl.col(ts_col).dt.replace_time_zone("UTC"))
        return out

    # -- existing keys ------------------------------------------------------
    def _load_existing_keys(self, part_dir: Path, key_cols: tuple[str, ...]) -> set[str]:
        if not part_dir.exists() or not any(part_dir.glob("*.parquet")):
            return set()
        lf = pl.scan_parquet(part_dir / "*.parquet")
        keys = lf.select(_key_expr(key_cols).alias("_key")).collect()["_key"]
        return set(keys.to_list())

    # -- write --------------------------------------------------------------
    def write(self, dataset: str, df: pl.DataFrame) -> int:
        """Append ``df`` to ``dataset``. Returns the number of NEW rows written.

        Append-only and idempotent: rows whose natural key already exists in the
        target partition are skipped, so calling this repeatedly with overlapping
        data never duplicates. Never overwrites an existing partition file.
        """
        _check(dataset)
        if df.is_empty():
            return 0
        key_cols = DATASETS[dataset][1]
        frame = self._coerce(dataset, df)
        frame = frame.with_columns(pl.col("ts_event").dt.date().alias("_date"))

        written = 0
        for (symbol, date), group in frame.group_by(["symbol", "_date"], maintain_order=True):
            part_dir = self.dataset_dir(dataset) / f"symbol={symbol}" / f"date={date}"
            seen = self._seen.get(part_dir)
            if seen is None:
                seen = self._load_existing_keys(part_dir, key_cols)
                self._seen[part_dir] = seen

            group = group.with_columns(_key_expr(key_cols).alias("_key"))
            new = group.filter(~pl.col("_key").is_in(list(seen)))
            if new.is_empty():
                continue

            seen.update(new["_key"].to_list())
            part_dir.mkdir(parents=True, exist_ok=True)
            out = new.drop("_key", "_date")
            out.write_parquet(part_dir / f"part-{uuid4().hex}.parquet", compression="zstd")
            written += new.height
        return written

    # -- read (lazy) --------------------------------------------------------
    def scan(self, dataset: str, symbol: str | None = None) -> pl.LazyFrame:
        """Lazily scan a dataset (optionally one symbol). Empty if nothing on disk."""
        _check(dataset)
        # stdlib glob handles the absolute partition pattern (Path.glob rejects
        # non-relative patterns on 3.12); matches the reader in store.py.
        files = _glob(self.partition_glob(dataset, symbol))
        if not files:
            return empty_frame(dataset).lazy()
        # No hive parsing: `symbol` already lives in the data; avoids a column clash.
        return pl.scan_parquet(files)

    # -- maintenance --------------------------------------------------------
    def compact(self, dataset: str, symbol: str | None = None) -> int:
        """Merge a dataset's many small part files into one per partition.

        Deduplicates on the natural key while doing so. Returns partitions
        compacted. Safe to run any time; purely a storage optimization.
        """
        _check(dataset)
        key_cols = DATASETS[dataset][1]
        base = self.dataset_dir(dataset)
        pattern = (f"symbol={symbol}" if symbol else "symbol=*", "date=*")
        compacted = 0
        for part_dir in sorted(base.glob(str(Path(*pattern)))):
            parts = sorted(part_dir.glob("*.parquet"))
            if len(parts) <= 1:
                continue
            merged = (
                pl.read_parquet(part_dir / "*.parquet")
                .with_columns(_key_expr(key_cols).alias("_key"))
                .unique(subset="_key", keep="last")
                .drop("_key")
            )
            tmp = part_dir / f".compact-{uuid4().hex}.parquet"
            merged.write_parquet(tmp, compression="zstd")
            for old in parts:
                old.unlink()
            tmp.rename(part_dir / f"part-{uuid4().hex}.parquet")
            self._seen.pop(part_dir, None)
            compacted += 1
        return compacted
