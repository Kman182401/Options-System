"""Databento historical backfill loader for MES/MNQ — SCAFFOLD (cost-guarded).

Downloads CME history (``GLBX.MDP3``; schemas ``ohlcv-1m`` and ``trades``) for a
date range and writes it into the SAME Parquet lake/schema as the live recorder,
so live + historical are unified and queried identically.

SAFETY / COST: a Databento range download consumes paid credits. This loader:

* **no-ops** (no network, exit 0) when ``OPTIONS_DATABENTO_API_KEY`` is unset;
* otherwise prints an **estimated cost** and stops, unless ``--confirm`` is
  passed. Only ``--confirm`` actually downloads.

The free Databento plan includes ~$125 of credit (see docs/SETUP.md). Run the
real backfill later, deliberately:

    uv run python -m options_system.data.databento_loader \\
        --start 2026-01-01 --end 2026-06-01 --schema ohlcv-1m --confirm
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime

import polars as pl

from config.settings import Settings

from .lake import SCHEMA_VERSION, Lake
from .recorder import session_for

DATASET = "GLBX.MDP3"
# Databento schema -> our lake dataset.
_SCHEMA_TO_DATASET = {"ohlcv-1m": "bars_1m", "trades": "trades"}


def _parse(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="databento_loader", description=__doc__)
    p.add_argument("--symbols", nargs="+", default=None, help="default: settings.record_symbols")
    p.add_argument("--start", help="YYYY-MM-DD (UTC)")
    p.add_argument("--end", help="YYYY-MM-DD (UTC)")
    p.add_argument("--schema", choices=sorted(_SCHEMA_TO_DATASET), default="ohlcv-1m")
    p.add_argument("--confirm", action="store_true", help="actually download (consumes credits)")
    return p.parse_args(argv)


def _estimate_cost(client, symbols, schema, start, end) -> float | None:
    try:
        return float(
            client.metadata.get_cost(
                dataset=DATASET,
                symbols=symbols,
                schema=schema,
                start=start,
                end=end,
                stype_in="parent",
                mode="historical-streaming",
            )
        )
    except Exception as exc:  # noqa: BLE001 - estimate is best-effort
        print(f"(could not estimate cost: {exc})", file=sys.stderr)
        return None


def _to_lake_rows(pdf, symbol: str, dataset: str) -> pl.DataFrame:
    """Map a Databento pandas frame into our canonical lake schema."""
    df = pl.from_pandas(pdf.reset_index())
    ts = pl.col("ts_event").cast(pl.Datetime("us", "UTC"))
    common = {
        "ts_event": ts,
        "ts_ingest": pl.lit(datetime.now(UTC)).cast(pl.Datetime("us", "UTC")),
        "symbol": pl.lit(symbol),
        "contract_id": pl.col("symbol").cast(pl.Utf8),  # Databento raw symbol = expiry
        "con_id": pl.col("instrument_id").cast(pl.Int64),
        "source": pl.lit("databento"),
        "schema_version": pl.lit(SCHEMA_VERSION, dtype=pl.Int32),
    }
    if dataset == "bars_1m":
        out = df.select(
            **common,
            open=pl.col("open"),
            high=pl.col("high"),
            low=pl.col("low"),
            close=pl.col("close"),
            volume=pl.col("volume").cast(pl.Float64),
            wap=pl.lit(None, dtype=pl.Float64),
            n_trades=pl.lit(None, dtype=pl.Int64),
        )
    else:  # trades
        out = df.select(
            **common,
            price=pl.col("price"),
            size=pl.col("size").cast(pl.Float64),
        )
    return out.with_columns(
        pl.col("ts_event").map_elements(session_for, return_dtype=pl.Utf8).alias("session")
    )


def _download_and_store(client, symbols: list[str], schema: str, start: str, end: str) -> int:
    lake = Lake()
    dataset = _SCHEMA_TO_DATASET[schema]
    written = 0
    for symbol in symbols:
        data = client.timeseries.get_range(
            dataset=DATASET,
            symbols=[f"{symbol}.FUT"],
            schema=schema,
            start=start,
            end=end,
            stype_in="parent",
        )
        pdf = data.to_df()  # prices as float dollars, symbol mapped
        if pdf.empty:
            print(f"{symbol}: no records for {start}..{end}")
            continue
        written += lake.write(dataset, _to_lake_rows(pdf, symbol, dataset))
    return written


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)
    settings = Settings()

    if settings.databento_api_key is None:
        print(
            "OPTIONS_DATABENTO_API_KEY not set — Databento backfill disabled (no-op).\n"
            "The free plan includes ~$125 of credit; see docs/SETUP.md. "
            "Set the key in .env to enable. No network call was made."
        )
        return 0

    if not args.start or not args.end:
        print("Provide --start and --end (YYYY-MM-DD) to estimate or download.", file=sys.stderr)
        return 2

    symbols = args.symbols or settings.record_symbols
    import databento as db  # local import: only when a key is present

    client = db.Historical(settings.databento_api_key.get_secret_value())
    parent_symbols = [f"{s}.FUT" for s in symbols]
    cost = _estimate_cost(client, parent_symbols, args.schema, args.start, args.end)
    cost_str = f"${cost:.2f}" if cost is not None else "unknown"
    print(
        f"Estimated Databento cost for {symbols} {args.schema} {args.start}..{args.end}: {cost_str}"
    )

    if not args.confirm:
        print("Dry run — nothing downloaded. Re-run with --confirm to download (consumes credits).")
        return 0

    n = _download_and_store(client, symbols, args.schema, args.start, args.end)
    print(f"Backfill complete: {n} rows written to the lake.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
