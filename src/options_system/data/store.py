"""DuckDB query layer over the Parquet lake — point-in-time correct by design.

This is the read side. It queries the Parquet files in place (no copy) and
exposes a small, deliberately narrow API:

* :meth:`DuckStore.get_bars` / :meth:`DuckStore.get_quotes_l1` — bounded reads.
* :meth:`DuckStore.asof_join` — attach the *latest-known* auxiliary value as of
  each row's ``ts_event`` using DuckDB's ``ASOF JOIN``.

**Point-in-time guarantee.** Every retrieval is bounded by ``ts_event`` and the
``asof_join`` only ever matches auxiliary rows with ``right.ts_event <=
left.ts_event``. There is no code path that lets a row see data stamped after
its own event time — no look-ahead, by construction. Later phases (features,
backtest) must attach external series only through ``asof_join``.
"""

from __future__ import annotations

from datetime import datetime
from glob import glob as _glob

import duckdb
import polars as pl

from config.settings import Settings

from .lake import DATASETS, Lake, empty_frame

_VALID_FREQS = {"1m": "bars_1m", "5s": "bars_5s"}


class DuckStore:
    """Thin, point-in-time-correct DuckDB wrapper over the lake."""

    def __init__(self, lake: Lake | None = None) -> None:
        self.lake = lake if lake is not None else Lake()
        self.con = duckdb.connect()
        # Pin the session to UTC so every TIMESTAMPTZ read comes back UTC-aware
        # (DuckDB otherwise renders in the local session zone, which would clash
        # with our UTC literals downstream). Keeps the whole layer UTC-only.
        self.con.execute("SET TimeZone='UTC'")

    def close(self) -> None:
        self.con.close()

    # -- internals ----------------------------------------------------------
    def _files(self, dataset: str, symbol: str | None = None) -> list[str]:
        return _glob(self.lake.partition_glob(dataset, symbol))

    def _read_window(
        self, dataset: str, symbol: str, start: datetime, end: datetime, key: tuple[str, ...]
    ) -> pl.DataFrame:
        if not self._files(dataset, symbol):
            return empty_frame(dataset)
        glob_str = self.lake.partition_glob(dataset, symbol)
        partition = ", ".join(f'"{c}"' for c in key)
        sql = f"""
            SELECT * EXCLUDE (rn) FROM (
                SELECT *, row_number() OVER (
                    PARTITION BY {partition} ORDER BY ts_ingest DESC
                ) AS rn
                FROM read_parquet('{glob_str}', hive_partitioning=false)
                WHERE ts_event >= ? AND ts_event <= ?
            ) WHERE rn = 1
            ORDER BY ts_event
        """
        return self.con.execute(sql, [start, end]).pl()

    # -- public reads -------------------------------------------------------
    def get_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        freq: str = "1m",
        continuous: bool = False,
    ) -> pl.DataFrame:
        """Bars for ``symbol`` in ``[start, end]`` (inclusive, both UTC).

        ``continuous=True`` returns the back-adjusted continuous series (built
        from the full raw history + roll events, then sliced to the window), so
        the adjustment is consistent regardless of the requested window.
        """
        if freq not in _VALID_FREQS:
            raise ValueError(f"freq={freq!r} invalid; use one of {sorted(_VALID_FREQS)}")
        dataset = _VALID_FREQS[freq]
        if not continuous:
            return self._read_window(dataset, symbol, start, end, DATASETS[dataset][1])

        # Continuous: build over ALL history (factors depend on every roll), then slice.
        from .continuous import build_continuous  # local import avoids a cycle

        if not self._files(dataset, symbol):
            return empty_frame(dataset).with_columns(pl.lit(1.0).alias("adj_factor"))
        glob_str = self.lake.partition_glob(dataset, symbol)
        raw = self.con.execute(
            f"""
            SELECT * EXCLUDE (rn) FROM (
                SELECT *, row_number() OVER (
                    PARTITION BY contract_id, ts_event ORDER BY ts_ingest DESC
                ) AS rn
                FROM read_parquet('{glob_str}', hive_partitioning=false)
            ) WHERE rn = 1
            """
        ).pl()
        rolls = self._read_rolls(symbol)
        cont = build_continuous(raw, rolls, adjustment=Settings().continuous_adjustment)
        return cont.filter((pl.col("ts_event") >= start) & (pl.col("ts_event") <= end)).sort(
            "ts_event"
        )

    def get_quotes_l1(self, symbol: str, start: datetime, end: datetime) -> pl.DataFrame:
        """Top-of-book quotes for ``symbol`` in ``[start, end]`` (UTC)."""
        return self._read_window("quotes_l1", symbol, start, end, DATASETS["quotes_l1"][1])

    def _read_rolls(self, symbol: str) -> pl.DataFrame:
        if not self._files("roll_events", symbol):
            return empty_frame("roll_events")
        glob_str = self.lake.partition_glob("roll_events", symbol)
        return self.con.execute(
            f"SELECT * FROM read_parquet('{glob_str}', hive_partitioning=false) ORDER BY ts_event"
        ).pl()

    # -- asof join (the leak-free attach primitive) -------------------------
    def asof_join(
        self,
        left: pl.DataFrame,
        right: pl.DataFrame,
        on: str = "ts_event",
        by: str | None = None,
        right_select: list[str] | None = None,
    ) -> pl.DataFrame:
        """Attach, to each ``left`` row, the most recent ``right`` row whose
        ``on`` value is ``<= left[on]`` (optionally partitioned by ``by``).

        Leak-free: a left row can never match a right row from its own future.
        ``right_select`` chooses which right columns to bring over (default: all
        except ``on``/``by``); ensure they don't clash with left column names.
        """
        if left.is_empty():
            return left
        excluded = {on, *({by} if by else set())}
        rcols = right_select or [c for c in right.columns if c not in excluded]
        rsel = ", ".join(f'r."{c}"' for c in rcols) if rcols else "NULL"
        by_cond = f'AND l."{by}" = r."{by}"' if by else ""

        self.con.register("_asof_left", left)
        self.con.register("_asof_right", right)
        try:
            sql = f"""
                SELECT l.*, {rsel}
                FROM _asof_left l
                ASOF LEFT JOIN _asof_right r
                  ON l."{on}" >= r."{on}" {by_cond}
            """
            return self.con.execute(sql).pl()
        finally:
            self.con.unregister("_asof_left")
            self.con.unregister("_asof_right")
