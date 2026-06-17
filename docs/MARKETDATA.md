# MARKETDATA.md — Daily market-state layer (VIX/VXN + cross-asset)

## What it is
A free **daily** data layer of the market's own fear gauges and cross-asset state, pulled from
**FRED** (the St. Louis Fed's free API — the same key already used by the macro layer). It exists
to feed the program's one live lead — the **day-ahead realized-volatility forecast** — the inputs
most likely to help it: the VIX, the Nasdaq vol index (VXN), the yield curve, oil, the dollar, and a
credit-risk spread.

It is a separate axis from the macro layer: the macro layer ingests *first-print economic-release
events*; this layer ingests *continuous daily time series* (FRED `output_type=1`, the final EOD
value). For the series used here — VIX/VXN, Treasury constant-maturity yields, WTI, the broad dollar
index, the ICE BofA high-yield OAS — the EOD value is **not revised**, so the latest observation
equals the first print (the same rationale the macro layer uses for the fed-funds target).

> Stooq was the original plan for this data; as of 2026 it serves a JavaScript anti-bot
> proof-of-work wall instead of CSV, so it is unusable for automation. FRED is the robust path and
> needs no new key or dependency.

## Series (config/marketdata.yaml)
`VIXCLS` (VIX → MES), `VXNCLS` (Nasdaq-100 vol → MNQ), `DGS3MO`/`DGS2`/`DGS10` (the curve),
`DCOILWTICO` (WTI), `DTWEXBGS` (broad USD), `BAMLH0A0HYM2` (HY credit OAS). All verified live; the
list is config-only, so adding a series is a YAML edit. (`BAMLH0A0HYM2` is sparser at source — ~793
obs — and is forward-filled by the as-of join; verify if you want denser credit coverage.)

## Point-in-time rule (the leakage guarantee)
Each value's `observed_at` is stamped at the **END of its observation day in UTC**. A daily EOD
value is only conservatively "knowable" once the day is over, so a feature built for label time `t`
can only consume series values from strictly-earlier days (lag-by-one). This is deliberately the
leak-safe choice — it can never look ahead — and is unit-tested: a value observed at end-of-day D is
*not* visible to a label at noon on day D.

## Lake (data/market_daily/)
Long format (`series_id, obs_date, value, observed_at, ingested_at`), partitioned by
`series=<id>`, **idempotent on `(series_id, obs_date)`** — a daily re-run appends only new dates and
never rewrites an existing one (these series are not revised). Gitignored.

## Features (x1) — `marketdata_feature_version = x1`
Point-in-time as-of features at each target time (latest value with `observed_at <= t`):
`mkt_<label>_level`, `mkt_<label>_chg_{1,5,20}d` (change vs that many prior observations),
`mkt_<label>_z_60d` (trailing z-score), and term-structure spreads `mkt_curve_ust_10y_ust_2y` /
`mkt_curve_ust_10y_ust_3m`. Nulls (not zeros) before a series' first observation.

## How to run
```sh
# Ingest all configured series (key from `pass`, gated by --allow-network + OPTIONS_FRED_API_KEY):
OPTIONS_FRED_API_KEY=$(pass show fred/api_key) \
  uv run python -m options_system.marketdata.ingest --allow-network --show
```
First live ingest: 14,892 daily rows across 8 series, current to 2026-06-15.

## Status
**Data + features only — no model, no verdict, no spend.** The `x1` block is wired into the
volatility dataset as an **opt-in, default-off** feature block (`features.with_marketdata`); turning
it on for a real forecast is a separate, pre-registered experiment.
