# Macro / economic-event layer (`macro_version = v1`, `macro_feature_version = v1`)

Phase 5 found **no edge** from price-only intraday direction. The pre-registered
next step is to **add information the price series cannot contain** — structured
macro / economic-event context — and re-run the *identical* Phase-5 verdict.
This is a clean controlled experiment: **same labels, same validation framework,
same verdict gates; only the model inputs change.**

> **Bottom line (2026-06-08):** **still no significant edge** on MES or MNQ.
> Macro features *dominate* the model's SHAP importance (they are the features the
> tree leans on most) but that reliance **does not translate into out-of-sample
> skill** — accuracy stays at/below chance, excess-over-beta stays negative, and
> the deflated Sharpe stays ≈0. A second, informative null: structured macro
> context at this ~3-day horizon does not beat beta after deflation.

---

## What it is

Two pieces:

1. **`macro/` — point-in-time event ingestion.** Pulls a calendar of high-impact
   US economic releases from **FRED / ALFRED** into a `macro_events` lake table.
2. **`features/macro_features.py` — leak-safe event features.** Turns that table
   into per-`t0` inputs (timing + outcome), appended to the training matrix.

Nothing here trades, backtests economically, or promotes a model. It changes the
*inputs* to the existing Phase-5 pipeline and asks the existing question again.

---

## Events ingested (free, FRED/ALFRED)

| type | FRED series | release (ET) | notes |
|---|---|---|---|
| `cpi` | `CPIAUCSL` | 08:30 | CPI, all items (SA) |
| `core_cpi` | `CPILFESL` | 08:30 | Core CPI, ex food & energy (SA) |
| `pce` | `PCEPI` | 08:30 | PCE price index (SA) |
| `core_pce` | `PCEPILFE` | 08:30 | Core PCE (SA) |
| `nfp` | `PAYEMS` | 08:30 | Nonfarm payrolls (SA) |
| `unrate` | `UNRATE` | 08:30 | Unemployment rate (SA) |
| `claims` | `ICSA` | 08:30 | Initial jobless claims (SA) |
| `gdp` | `GDPC1` | 08:30 | Real GDP, advance estimate (SAAR) |
| `retail` | `RSAFS` | 08:30 | Advance retail sales (SA) |
| `ppi` | `PPIFIS` | 08:30 | PPI final demand (SA) |
| `fomc` | `DFEDTARU` | 14:00 | FOMC statement; outcome = fed-funds target upper limit |

The set, release clock times, and the FOMC schedule live in `config/macro.yaml`.

**ISM manufacturing & services PMIs are intentionally excluded.** The Institute
for Supply Management's data was **removed from FRED on 2016-06-24** over a
licensing dispute, so it is not available from this free source. We do not
fabricate it.

---

## Point-in-time correctness — the timing-vs-outcome rule (the whole game)

A scheduled event has two parts that become knowable at *different* times, and
they are handled by two different look-ups:

* **Timing is public in advance.** Release calendars are published well ahead, so
  "minutes to the next CPI / FOMC" is legitimately available **before** the event.
  Timing features look **forward** at the *schedule* (`event_time` only — never an
  outcome).
* **Outcomes are known only at release.** "Most-recent actual − prior" and rolling
  tendencies index **only into events with `event_time <= t0`** — a strictly
  backward look-up, so a bar at `t0` can never see a value released after it.

### First-print values (ALFRED vintage), never a revision
Every `actual_pit` is the value **as first released** (FRED `output_type=4`,
"Observations, Initial Release Only"). Each first-print observation's
`realtime_start` is its **publication date**; we combine that date with the
standard release clock time (08:30 ET data, 14:00 ET FOMC) and convert to UTC to
get `event_time`. Using a later revision would leak the future.

> `output_type=4` requires an explicit ALFRED real-time window
> (`realtime_start=2000-01-01`, `realtime_end=9999-12-31`); the default (today
> only) errors with "no vintage dates exist". The fed-funds target (`DFEDTARU`) is
> a daily series with too many vintages for that file type — but it is **never
> revised**, so its standard (latest) observation equals its first print; we use
> that and read it only as-of a meeting date.

### `surprise` is null — never fabricated
FRED carries no consensus/expectations series, so the `surprise` column (actual vs
consensus) is left **null**. The outcome feature is the change vs the previous
first-print release (`actual_pit − prior`), which is **not** a surprise.

### FOMC: scheduled calendar, emergencies excluded
FOMC events are the **8 regularly scheduled meetings per year** (decision dates
verified against the Fed's published FOMC calendars). The rate **outcome** is the
fed-funds target upper limit set at the meeting (announced in the 14:00-ET
statement → no look-ahead). **Intermeeting emergency actions are excluded** from
the scheduled calendar because they were not knowable in advance: the March-2020
COVID emergency cuts (2020-03-03/15/23/31) and the 2019-10-11 balance-sheet
announcement are omitted, and Jackson Hole symposiums are not FOMC meetings. The
calendared March-2020 slot is its scheduled 2020-03-18.

### Caveat — release-date provenance for data series
For the data releases we source the *next* release date from the ingested release
series (FRED only carries releases that have happened). Release dates for these
series are fixed in advance and effectively never move, so using the realized
date as the known-ahead schedule is point-in-time fair for all but the **final
release month** of the data (where the next release is not yet in the table, so
"minutes-to-next" is NaN). FOMC carries an explicit forward calendar, so its
timing is always populated.

---

## Feature list (`macro_feature_version = v1`, 27 features)

All are `Float64`; undefined values are **NaN** (LightGBM handles missing values
natively, so no row is ever dropped for a null macro column).

**Timing (schedule, known ahead)** — per type in `{fomc, cpi, nfp, pce}`:
`macro_mins_to_<type>`, `macro_mins_since_<type>` (8 columns).
Aggregate over high-impact events: `macro_mins_to_event`, `macro_mins_since_event`,
`macro_in_blackout` (within `blackout_lead_minutes` before any high-impact event),
`macro_events_next_24h` (count scheduled in the next `next_horizon_hours`),
`macro_is_fomc_day`, `macro_is_nfp_day` (6 columns).

**Outcome (release-time, strictly backward)** — `actual_pit − prior` of the most
recent release per type in `{cpi, core_cpi, pce, core_pce, nfp, unrate, claims,
gdp, retail, ppi, fomc}` (11 columns), plus a short rolling tendency
`macro_tendency_<type>` for `{cpi, nfp}` (mean of the last `tendency_window`
changes, 2 columns).

### Leakage tests (`tests/test_macro_features_leakage.py`)
The timing-vs-outcome split is proven, not asserted:

* **Outcomes are strictly backward** — hiding the outcomes of all events released
  after a cut leaves every outcome feature unchanged for `t0 <= cut`.
* **Timing uses the schedule and ignores outcomes** — minutes-to-next is the
  correct positive gap before a scheduled event, and is invariant to blanking
  every actual/prior.
* **Teeth** — a deliberately *forward* (leaky) outcome look-up *is* corrupted by
  the same hide-the-future manipulation, proving the backward-invariance check
  would catch a real leak.

---

## Integration into the training matrix

`models/dataset.py` appends the macro columns at each label `t0`
(`load_training_matrix(..., with_macro=True)`, the default). The macro features
are computed directly from the (tiny) event table — timing closed-form from the
schedule, outcomes a backward look-up — and `hstack`ed onto the price matrix. The
null/non-finite **row gate stays on the price features only**, so the row count is
**identical** to the price-only matrix (MES 10,827 / MNQ 11,688); macro columns
may be NaN where undefined. The matrix cache key includes the macro tag, and
assembly stays ~1 s. `with_macro=False` reproduces the Phase-5 price-only matrix
byte-for-byte.

---

## The re-verdict (identical gates) — price-only vs price+macro

Full history 2019→2026, same `label_version=v1`, same `validation_version`, same
verdict thresholds. The price-only column reproduces Phase 5 exactly.

| symbol | inputs | n | dir. acc | excess SR | long SR | excess **DSR** | **PBO** | CPCV excess-SR (mean) | VERDICT |
|---|---|---|---|---|---|---|---|---|---|
| **MES** | price-only (45) | 10,827 | 0.5164 | −0.0175 | 0.0299 | 0.006 | 0.83 | −0.025 | no significant edge |
| **MES** | price+macro (72) | 10,827 | **0.4976** | **−0.0371** | 0.0299 | **0.00001** | **0.88** | −0.011 | **no significant edge** |
| **MNQ** | price-only (45) | 11,688 | 0.5267 | −0.0127 | 0.0315 | 0.021 | 0.70 | −0.019 | no significant edge |
| **MNQ** | price+macro (72) | 11,688 | 0.5209 | −0.0142 | 0.0315 | 0.014 | **0.52** | −0.014 | **no significant edge** |

* **The verdict does not change** — both symbols remain *no significant edge*, and
  the four gates (accuracy > 0.52, PBO < 0.5, excess DSR > 0.5, positive mean
  excess) all still fail.
* **Macro did not help.** On MES it actively *hurt* — pooled OOS accuracy dipped
  below chance (0.498) and excess Sharpe got more negative. On MNQ the numbers
  barely moved (PBO improved to a borderline 0.52, CPCV excess-Sharpe slightly
  less negative) but excess-over-beta is still **negative** and the deflated
  Sharpe is still ≈0 — no real, beta-beating, deflated edge.

### What SHAP says (and why it is *not* a contradiction)
Macro features **dominate** the importance ranking — they account for ~66–69 % of
total mean |SHAP|, and the top drivers are `macro_chg_ppi`, `macro_chg_cpi`,
`macro_chg_core_cpi`, `macro_mins_since_nfp`, `macro_mins_since_pce`. No single
feature dominates (top share 7 % MES / 11 % MNQ; no leakage smell).

That the model **leans heavily on macro features yet has no out-of-sample edge** is
exactly what the validation framework exists to expose: in-sample reliance that
does not generalize. It also argues *against* leakage — a real outcome leak would
**inflate** OOS accuracy, whereas here accuracy is at/below chance. The macro
features are economically plausible and informative to the tree in-sample; they
simply do not predict next-move direction strongly enough to beat beta after the
cost of selection (PBO stays high).

---

## Implication for next steps

A **second null** is informative, not a failure: efficient instruments at this
~3-day horizon resist both price-only and structured-macro prediction. Per the
plan, the remaining levers (to test through this *same* verdict) are:

1. **Unstructured news / sentiment** (FinBERT / local LLM on the GPU) — the layer
   intentionally separated *after* macro so signal can be attributed.
2. **Microstructure / order-flow** (Databento trades / L2), likely paired with a
   **shorter-horizon label** where intraday information has a chance to matter.
3. **Options structure** (Phase 2).

The bar does not move; the data does. A strategy is built only when a configuration
clears all four gates here.

---

## Run it

```bash
# 1) ingest the macro events (free FRED key; key-gated no-op if unset)
OPTIONS_FRED_API_KEY="$(pass show fred/api_key)" \
  uv run python -m options_system.macro.ingest --show

# 2) re-run the verdict, price-only vs price+macro, side by side
uv run python -m options_system.models.run --symbols MES MNQ --compare
# (single run: add nothing for price+macro, or --no-macro for the price-only baseline)
```

Runs are saved to `data/models/runs/<symbol>.json` (price+macro, canonical) and
`<symbol>_price_only.json`, with `<symbol>_comparison.json` for the side-by-side,
and logged to the local MLflow store under `data/mlruns`. Rationale for each
choice: `docs/DECISIONS.md` (Phase 6).
