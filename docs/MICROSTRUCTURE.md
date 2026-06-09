# Microstructure / order-flow (OFI) feature layer

A new, deliberately **separate** experiment from the price `feature_version=v1`
layer. The two prior nulls (price-only features, then a macro layer) tested a
*low-frequency* regime and found no edge. This layer tests the one regime they
did not: **short-horizon order flow** — does the imbalance between buy and sell
pressure in the order book predict the next few minutes of price?

It is **cost-staged**. Level-2/3 history is billed per byte and is expensive, so
this stage ingests the cheapest schema that can answer "is there *any* order-flow
signal here?" — top-of-book **MBP-1** (best bid/offer + trades). Deep-book
multi-level OFI (which needs the ~pricier MBP-10) is a **deliberate later
escalation, only if this stage shows a pulse.**

Nothing here touches the v1 price features, the labels, or the validation
framework. Every row is stamped `microstructure_feature_version = m1`.

## What it ingests

- **Instruments:** the full-size e-minis **ES** and **NQ** (front-month
  continuous, volume roll = Databento `ES.v.0` / `NQ.v.0`). Price discovery for
  the S&P 500 / Nasdaq-100 happens in these deep books; any edge found here
  transfers directly to the **MES / MNQ** micros (1/10 notional, same index) that
  the engine actually trades.
- **Schema:** `mbp-1` from `GLBX.MDP3` — every best-bid/offer update **and** every
  trade, in one stream.
- **Window:** small and recent (default `2026-03-01 → 2026-06-01`), configurable.
  The binding constraint is budget and local compute, not history.
- **Session:** RTH only by default (09:30–16:00 ET), where order-flow signal and
  liquidity are strongest and overnight book noise is excluded.

## The cost guard (read this before running a real pull)

Databento charges per byte. **MBP-1 on ES/NQ is not cheap** — these instruments
are so active at the touch that BBO updates rival deep-book volume. Measured
2026-06-08: ES ≈ $1.3/RTH-day, NQ ≈ $1.6/RTH-day; the full default 3-month window
is **~$130**, far over the cap.

So `config/microstructure.yaml` sets a hard **`databento_budget_usd_cap`**
(default `$40`). The ingest CLI:

1. estimates the cost of **every day chunk** with the free `metadata.get_cost`
   *before* downloading it, and tracks a running total;
2. **aborts** (downloads nothing more) the instant the running total + the next
   chunk would exceed the cap — it never splits work to sneak around the cap;
3. logs **actual billable bytes** (`metadata.get_billable_size`) per chunk and in
   total;
4. without `--confirm`, only estimates and prints — nothing is downloaded.

The API key is sourced securely (never written to `.env`, never committed): from
`Settings` if present, else from the `pass` store (`databento/api_key_2` (live),
then `databento/api_key_3` (reserved backup slot — `pass insert` it when the live
key is depleted)). The original `databento/api_key` ran out of credits and was
removed 2026-06-08.

## Dollar bars (the sampling clock)

Instead of fixed time bars, the event stream is aggregated into **dollar bars**: a
new bar closes once a fixed amount of **traded notional** (`price × size ×
multiplier`) accumulates (López de Prado, *Advances in Financial ML*). Dollar
bars sample faster when information arrives and are statistically much better
behaved (closer to IID) than time bars — the standard modern choice for
short-horizon ML. The per-instrument `dollar_threshold` is tuned to land at a few
hundred to ~1–2k bars per RTH session (realized median is reported by the QA
view). Each bar also records its **duration** (seconds spanned), which itself
carries regime information on an event clock.

Bars are built by a **streaming, O(1)-memory reducer** (`bars.build_dollar_bars`):
it holds only the current best bid/offer and the current bar's accumulators, so a
whole day streams through without ever materialising. Bars **never span a contract
roll or a session boundary** — a change in `instrument_id` or RTH date closes the
current bar and severs order-flow continuity.

## The features (`microstructure_feature_version = m1`)

All single-level (top of book) and **strictly causal** — every value in bar *t*
uses only events at or before bar *t*'s close.

| feature | what it measures |
|---|---|
| `ofi_top` | **the centerpiece** — single-level Order-Flow Imbalance (Cont-Kukanov-Stoikov 2014) summed over the bar; +ve = net buy pressure |
| `qimb_top_close` / `qimb_top_twa` | top-of-book queue imbalance (bid vs ask size), at close / time-weighted over the bar |
| `signed_vol` | signed aggressor volume (buy +, sell −) |
| `trade_imbalance` | `signed_vol` / traded volume, in [−1, 1] |
| `micro_minus_mid_close` / `_twa` | micro-price minus mid (drift toward the heavier side), at close / time-weighted |
| `spread_ticks_close` / `_twa` | bid-ask spread in ticks, at close / time-weighted |
| `depth_top_twa` | time-weighted top-of-book total size |
| `rv_intrabar` | realized intrabar volatility = √(Σ squared mid log-returns) |
| `duration_s` | bar wall-clock duration in seconds |
| `dmid` / `ret_bar` | mid change / log-return over the bar (Δmid is the OFI sanity target) |
| `ofi_top_lag1`, `ofi_top_roll3`, `signed_vol_roll3` | causal rolling history (previous bar / sum of last *k* bars, within a contract segment) |

**Out of scope here:** multi-level OFI (MLOFI) across the deeper book — that needs
MBP-10 and is the later escalation.

## Leakage guarantees

The non-negotiable rule: a feature at bar *t* depends only on events with
`ts_event ≤ bar t's close`. This is proven, not asserted:

- **Truncation-invariance teeth test** (`tests/test_microstructure_leakage.py`):
  rebuild from the event stream truncated at bar *k*'s close — bars 0..*k* must
  come out bit-for-bit identical. A planted forward-looking feature (one that
  reads the *next* bar) is shown to break this, proving the test can catch a leak.
- **Sanity (a diagnostic, not a leak):** the contemporaneous correlation between
  `ofi_top` and same-bar Δmid must be strongly positive (the well-established
  stylized fact). Measured on real data: **ES ≈ 0.88, NQ ≈ 0.73.**

## Storage & retrieval

Bars+features are written to the Parquet lake at
`data/micro_bars/symbol=<SYM>/date=<YYYY-MM-DD>/part-*.parquet`, zstd, idempotent
on `ts_event` per partition (re-running never duplicates), latest-ingest wins on
read. Query via `microstructure.ingest.read_micro_bars(symbol, start, end)`
(DuckDB under the hood). The price `feature_version=v1` artifacts are untouched.

## Parallel day reduction (the serial path is the source of truth)

Reducing a day of MBP-1 (millions of best-bid/offer + trade events) to dollar bars
is **CPU-heavy**; the download is I/O. Before the larger ~4-month pull, the reduction
is the runtime bottleneck, so it can be run across processes — **without changing a
single output value.** This is a behavior-preserving speedup: the only acceptable
difference between the serial and parallel paths is faster wall-clock.

**The serial reducer is the reference implementation.** `build_dollar_bars`,
`assemble_features`, `from_records`, and the per-day `reduce_work_unit` are unchanged
math. The parallel path just runs that same per-day reducer in worker processes.

- **The independent unit** is one `(symbol, session-day)` — a `DayWorkUnit`. A whole
  session-day stream goes to exactly **one** worker (so intrabar order-flow
  continuity is never split), and two days are never merged into one unit (the
  reducer already severs flow at every session / contract seam). This is *why*
  parallelisation is safe only across symbol-days: each unit is fully independent —
  no shared mutable reducer state, no flow crossing a seam.
- **The worker** is the module-level, picklable `reduce_work_unit(unit)` — it reads
  one local DBN file (or, in tests, in-memory events) and returns a fully reduced
  frame. It makes **no Databento API call** and never touches an Executor/Future.
- **Reassembly is deterministic.** `reduce_units(...)` places each result back by its
  input index, so worker completion order never affects output;
  `reduce_units_to_frame(...)` additionally sorts by `(symbol, ts_event, con_id,
  ts_open)` — a full ordering, independent of input order too. The lake write stays
  per-`(symbol, date)` partition, idempotent / latest-ingest-wins, exactly as before.
- **Budget gating + downloads stay strictly sequential** in `run_ingest` regardless of
  `--workers`: the `databento_budget_usd_cap` is a running total honoured in order, so
  a parallel run aborts at the exact same day a serial run would. Only the reduction
  fans out, in **waves** of `workers` days (≤ `workers` temp files in flight — memory
  and disk stay bounded; the full multi-month raw set is never materialised).

```bash
# Serial (default; bit-identical to every prior run):
uv run python -m options_system.microstructure.ingest --start 2026-05-18 --end 2026-05-23 --confirm

# Parallel reduction with 4 worker processes (same output, faster):
uv run python -m options_system.microstructure.ingest --start 2026-05-18 --end 2026-05-23 --confirm --workers 4

# Auto worker count = min(cpus, n_tasks, 8):
uv run python -m options_system.microstructure.ingest --start 2026-05-18 --end 2026-05-23 --confirm --workers auto
```

`--workers 1` (the default) uses the serial-equivalent path. `--workers >1` enables
parallel reduction. The dry-run (no `--confirm`) is unchanged — it still estimates
cost and downloads nothing.

### Proving equivalence offline (zero Databento credits)

`tests/test_microstructure_parallel.py` proves serial == parallel **exactly** on
synthetic MBP-1 fixtures — no network, no API key, no credits spent:

```bash
# Behavior-preserving equivalence + determinism + worker-count invariance:
uv run pytest tests/test_microstructure_parallel.py -q
```

It asserts bit-identical frames (`polars.testing.assert_frame_equal(check_exact=True,
check_dtypes=True)` — **no float tolerance**) across `workers=1/2/4`, repeated runs,
shuffled input order, and the contract-roll / session-boundary / trailing-partial /
quote-only / empty-day cases. **This step spends zero Databento credits and does not
run the large pull.**

## How to regenerate

```bash
# Dry run — estimate cost only, download nothing:
uv run python -m options_system.microstructure.ingest --start 2026-05-18 --end 2026-05-23

# Real pull (consumes credits; aborts if over the cap):
uv run python -m options_system.microstructure.ingest --start 2026-05-18 --end 2026-05-23 --confirm
# ...add --workers 4 (or auto) to parallelise the reduction; identical output, faster.

# QA / health report:
uv run python -m options_system.observability.micro_health --symbols ES NQ --start 2026-05-18 --end 2026-05-23
# or the dashboard:
uv run streamlit run src/options_system/observability/micro_health.py
```

A run logs dataset-level stats (bar counts, feature list, window, thresholds,
cost) to the local MLflow file store under `data/mlruns` (experiment
`microstructure-data`). No model is trained in this layer.

## What's next (Prompt 8)

Short-horizon **triple-barrier labels** (vertical barrier ~15–30 min, σ scaled to
intraday vol) built on top of these bars — then a model and an honest verdict.
The bars carry `mid_close` / `dmid` / `duration_s` so that labeling is
straightforward.
