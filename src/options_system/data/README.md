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
