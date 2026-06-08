# `macro/` — point-in-time economic-event ingestion (Phase 6)

Pulls a calendar of high-impact US economic releases from **FRED / ALFRED** and
stores it as the `macro_events` lake table. This is the *raw external input* for
the macro feature layer (`features/macro_features.py`); it builds no features
itself.

## What it ingests
CPI, Core CPI, PCE, Core PCE, nonfarm payrolls, unemployment rate, initial
jobless claims, real GDP, advance retail sales, PPI (final demand), and **FOMC**
scheduled meetings. Series ids + standard release clock times live in
`config/macro.yaml`.

ISM manufacturing/services PMIs are **not** ingested — ISM data was removed from
FRED on 2016-06-24 over licensing, so it is unavailable from this free source
(we do not fabricate it). See `docs/MACRO.md`.

## The point-in-time rule (the whole game)
* **Timing is public ahead** — the next release's date/time is knowable before it
  happens, so "minutes to next CPI/FOMC" is a legitimate feature.
* **Values are known only at release** — every `actual_pit` is the **first-print**
  value (FRED `output_type=4`, "Initial Release Only"); its `event_time` is the
  publication date (the vintage's `realtime_start`) combined with the release
  clock time (08:30 ET data, 14:00 ET FOMC), in UTC. No later revision is used.
* `surprise` (actual vs consensus) is **null** — FRED has no consensus series and
  we never fabricate one. The outcome feature is `actual_pit - prior` (the change
  vs the previous first-print), not a surprise.

## Run it
```bash
# free key: https://fred.stlouisfed.org/docs/api/api_key.html
OPTIONS_FRED_API_KEY="$(pass show fred/api_key)" \
  uv run python -m options_system.macro.ingest --show
```
Key-gated: with `OPTIONS_FRED_API_KEY` unset it no-ops with a clear message.
Network access uses only the Python stdlib (`urllib`) — no new dependency.

## Storage
`data/macro_events/date=<YYYY-MM-DD>/part-<uuid>.parquet`, partitioned by the
event-time date (not by symbol — macro context is instrument-independent),
append-only and idempotent on `(event_type, event_time)`. Read it back with
`read_macro_events(start, end)`. Method + leakage rules: `docs/MACRO.md`.
