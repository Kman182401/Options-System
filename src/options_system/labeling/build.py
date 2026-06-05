"""Build, store, and retrieve label tables — versioned, idempotent, leak-aware.

Labels are computed over the **full** continuous history of each symbol (same
series the features use), then written to the lake at::

    data/labels/symbol=<SYM>/date=<YYYY-MM-DD>/part-<uuid>.parquet

partitioned by the event date (``t0``), zstd, idempotent on the natural key
``t0``. Re-running never duplicates. ``--start/--end`` bound only what is
*written*, never what is computed, so stored labels are identical regardless of
the write window.

Retrieval keeps the point-in-time guarantee end-to-end:

* :func:`read_labels` — versioned label rows for a window (latest-ingest wins).
* :func:`labels_with_features` — attaches the **as-of** feature row at each
  ``t0`` through the store's ``asof_join`` (``feature.ts_event <= t0``), so the
  next phase gets an aligned ``(features@t0, label, t1, weight)`` matrix with no
  look-ahead. A label may look forward; the *features* attached to it never do.

The meta-labeling hook (:func:`apply_meta_labeling`) is structure only — the
schema carries ``side`` / ``meta_label`` columns and the API exists, but the
secondary model is deferred (see ``docs/LABELING.md``).
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from glob import glob as _glob
from pathlib import Path
from typing import cast
from uuid import uuid4

import numpy as np
import polars as pl

from config.settings import Settings

from ..data.store import DuckStore
from ..features.config import FeatureConfig
from .config import LabelConfig
from .triple_barrier import generate_labels
from .weights import sample_weights

_DATASET = "labels"
_WIDE_START = datetime(2000, 1, 1, tzinfo=UTC)
_WIDE_END = datetime(2100, 1, 1, tzinfo=UTC)

# Persisted column order: keys first, then outcome, weights, meta hook, stamps.
_COLUMNS = (
    "t0",
    "t1",
    "symbol",
    "ret",
    "label",
    "barrier",
    "sigma",
    "n_bars",
    "contract_id",
    "roll_crossed",
    "session",
    "degraded",
    "avg_uniqueness",
    "weight",
    "side",
    "meta_label",
    "label_version",
    "ts_ingest",
)


def _root() -> Path:
    return Settings().data_dir / _DATASET


def partition_glob(symbol: str | None = None) -> str:
    sym = "*" if symbol is None else f"symbol={symbol}"
    return str(_root() / sym / "date=*" / "*.parquet")


def _existing_keys(part_dir: Path) -> set:
    if not part_dir.exists() or not any(part_dir.glob("*.parquet")):
        return set()
    keys = cast("pl.DataFrame", pl.scan_parquet(part_dir / "*.parquet").select("t0").collect())
    return set(keys["t0"])


def _write_symbol(frame: pl.DataFrame, symbol: str) -> int:
    """Append a symbol's label frame; idempotent on ``t0`` per date partition."""
    if frame.is_empty():
        return 0
    frame = frame.with_columns(pl.col("t0").dt.date().alias("_date"))
    written = 0
    for (date,), group in frame.group_by(["_date"], maintain_order=True):
        part_dir = _root() / f"symbol={symbol}" / f"date={date}"
        seen = _existing_keys(part_dir)
        new = group.filter(~pl.col("t0").is_in(list(seen))) if seen else group
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


def _attach_weights(labels: pl.DataFrame, cont: pl.DataFrame, cfg: LabelConfig) -> pl.DataFrame:
    """Map each label's [t0, t1] to bar positions and attach uniqueness + weight."""
    if labels.is_empty():
        return labels.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("avg_uniqueness"),
            pl.lit(None, dtype=pl.Float64).alias("weight"),
        )
    bar_ts = cont.sort("ts_event")["ts_event"].to_numpy()
    starts = np.searchsorted(bar_ts, labels["t0"].to_numpy(), side="left")
    ends = np.searchsorted(bar_ts, labels["t1"].to_numpy(), side="left")
    out = sample_weights(
        starts,
        ends,
        returns=labels["ret"].to_numpy(),
        scheme=cfg.weights.scheme,
        time_decay=cfg.weights.time_decay,
    )
    return labels.with_columns(
        pl.Series("avg_uniqueness", out["avg_uniqueness"]),
        pl.Series("weight", out["weight"]),
    )


def build_labels(
    symbols: list[str],
    *,
    write_start: datetime | None = None,
    write_end: datetime | None = None,
    cfg: LabelConfig | None = None,
    store: DuckStore | None = None,
) -> dict[str, int]:
    """Compute triple-barrier labels over full history and write the lake tables.

    ``write_start/write_end`` bound only the rows persisted (computation always
    uses full history). Returns ``{symbol: rows_written}``.
    """
    cfg = cfg or LabelConfig.load()
    degraded = FeatureConfig.load().degraded_day_set()  # single source of truth
    own_store = store is None
    store = store or DuckStore()
    try:
        ts_ingest = datetime.now(UTC)
        result: dict[str, int] = {}
        for sym in symbols:
            cont = _continuous(store, sym)
            if cont.is_empty():
                result[sym] = 0
                continue
            rolls = store._read_rolls(sym)
            labels = generate_labels(cont, cfg, rolls=rolls, degraded_dates=frozenset(degraded))
            labels = _attach_weights(labels, cont, cfg)
            labels = labels.with_columns(
                pl.lit(sym).alias("symbol"),
                pl.lit(ts_ingest).cast(pl.Datetime("us", "UTC")).alias("ts_ingest"),
            )
            if write_start is not None:
                labels = labels.filter(pl.col("t0") >= write_start)
            if write_end is not None:
                labels = labels.filter(pl.col("t0") <= write_end)
            result[sym] = _write_symbol(labels.select(_COLUMNS), sym)
        return result
    finally:
        if own_store:
            store.close()


# --- retrieval (point-in-time correct) ------------------------------------- #


def read_labels(
    symbol: str, start: datetime, end: datetime, store: DuckStore | None = None
) -> pl.DataFrame:
    """Label rows for ``symbol`` with ``t0`` in ``[start, end]`` (UTC), latest-ingest wins."""
    own = store is None
    store = store or DuckStore()
    try:
        if not _glob(partition_glob(symbol)):
            return pl.DataFrame()
        glob_str = partition_glob(symbol)
        return store.con.execute(
            f"""
            SELECT * EXCLUDE (rn) FROM (
                SELECT *, row_number() OVER (PARTITION BY t0 ORDER BY ts_ingest DESC) AS rn
                FROM read_parquet('{glob_str}', hive_partitioning=false)
                WHERE t0 >= ? AND t0 <= ?
            ) WHERE rn = 1
            ORDER BY t0
            """,
            [start, end],
        ).pl()
    finally:
        if own:
            store.close()


def labels_with_features(
    symbol: str, start: datetime, end: datetime, *, store: DuckStore | None = None
) -> pl.DataFrame:
    """Labels in ``[start, end]`` with the as-of feature row attached at each ``t0``.

    Leak-free by construction: the attach is the store's ASOF JOIN with
    ``feature.ts_event <= t0``, so a label's features never come from its own
    future. Returns the aligned ``(features@t0, label, t1, weight, ...)`` matrix.
    """
    from ..features.build import read_features

    own = store is None
    store = store or DuckStore()
    try:
        labels = read_labels(symbol, start, end, store=store)
        if labels.is_empty():
            return labels
        hi = cast("datetime", labels["t0"].max())
        feats = read_features(symbol, _WIDE_START, hi, store=store)
        if feats.is_empty():
            return labels
        feats = feats.drop("symbol")  # avoid clashing with the label's symbol column
        # join key is t0 (renamed to ts_event for the asof primitive), kept as t0 after
        left = labels.with_columns(pl.col("t0").alias("ts_event"))
        joined = store.asof_join(left, feats, on="ts_event")
        return joined.drop("ts_event")
    finally:
        if own:
            store.close()


# --- meta-labeling hook (STRUCTURE ONLY — secondary model deferred) -------- #


def apply_meta_labeling(
    labels: pl.DataFrame,
    side: pl.Series | None = None,
    **kwargs: object,
) -> pl.DataFrame:
    """HOOK ONLY — not implemented. Reserved API for AFML meta-labeling (ch. 3.6).

    Meta-labeling adds a *secondary* model that decides whether to ACT on (and how
    to SIZE) a primary signal: the primary model/strategy proposes a ``side``
    (+1/−1), and the meta-label is 1 if taking that side would have been correct
    (hit the profit-take before the stop) and 0 otherwise. The label schema
    already carries the ``side`` and ``meta_label`` columns so this can be layered
    on later without a migration; this function is the seat where that logic will
    live. Building the secondary model is explicitly out of scope for Phase 3.
    """
    raise NotImplementedError(
        "meta-labeling is a deferred hook (Phase 3 builds the structure only); "
        "the schema carries `side`/`meta_label` for when it is implemented."
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="labeling.build", description=__doc__)
    p.add_argument("--symbols", nargs="+", default=None, help="default: settings.record_symbols")
    p.add_argument("--start", help="write-window start YYYY-MM-DD (UTC); compute is always full")
    p.add_argument("--end", help="write-window end YYYY-MM-DD (UTC)")
    args = p.parse_args(argv)

    symbols = args.symbols or Settings().record_symbols

    def _parse(d: str | None) -> datetime | None:
        return datetime.fromisoformat(d).replace(tzinfo=UTC) if d else None

    cfg = LabelConfig.load()
    print(f"label_version={cfg.label_version} symbols={symbols}")
    written = build_labels(
        symbols, write_start=_parse(args.start), write_end=_parse(args.end), cfg=cfg
    )
    for sym, n in written.items():
        print(f"  {sym}: +{n:,} label rows written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
