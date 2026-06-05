# data/

**The data layer: gets market and reference data in, and stores it locally.**
Two jobs. (1) *Live ingestion* — subscribe to real-time bars/ticks for the
traded instrument from IBKR (via `ib_async`) and hand them to the engine.
(2) *Historical backfill* — pull history (later from Databento for CME, plus
news/macro from Finnhub and free RSS/calendars) and persist everything to the
local store: **Parquet** files queried through **DuckDB**. This module defines
the on-disk schema and is the single source of truth for "what data do we
have". It does not compute features or make decisions — it only acquires,
validates, and stores raw data so the rest of the system can read it
deterministically.

## Lake layout (Phase 1.1)

Parquet, partitioned, under `data/` (gitignored):

```
data/<dataset>/symbol=<SYM>/date=<YYYY-MM-DD>/part-<uuid>.parquet
```

Datasets: `bars_1m`, `bars_5s`, `quotes_l1`, `trades`, `roll_events`. Every
market-data row carries **both** `ts_event` (exchange time) and `ts_ingest`
(receipt time), always UTC — the foundation for point-in-time-correct reads.

## Modules

- `lake.py` — canonical schemas + the append-only, idempotent writer
  (natural-key dedupe, zstd, never overwrites; `compact()` merges small files).
- `recorder.py` — live IBKR recorder: MES/MNQ 5s bars → `bars_5s` + aggregated
  `bars_1m`, and L1 top-of-book → `quotes_l1`. Reconnects, tags RTH/ETH, flushes
  on an interval. L2/depth is a marked extension point (deferred). Run:
  `uv run python -m options_system.data.recorder`.
- `store.py` — DuckDB query layer (`get_bars`, `get_quotes_l1`, `asof_join`).
  **Point-in-time correct**: `asof_join` only ever matches rows with
  `ts_event <= the row's ts_event`. No look-ahead, by construction.
- `continuous.py` — front-month selection + roll detection (volume crossover with
  calendar fallback) + back-adjusted continuous series (ratio by default). Raw
  per-contract data is never mutated.
- `validate.py` — reports dupes / gaps / bad-OHLC / future-ingest / non-monotonic.
  Never repairs.
- `databento_loader.py` — historical backfill into the same lake (scaffold;
  cost-guarded, no-ops without a key).

Retrieval is always through `store.py` so the point-in-time guarantee holds.
