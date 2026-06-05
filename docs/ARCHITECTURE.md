# Architecture

Local-first, one machine, two separate "brains" that never mix. This document is
the plain-English data-flow narrative. The guiding rule (the prime directive) is
that a human can understand the whole system at any time, so every box below is a
module you can read and explain on its own.

---

## The two brains

1. **Live engine — deterministic, boring, fast.** It loads an already-approved
   model and runs the trading loop. There is **no LLM anywhere in this loop**, so
   it costs nothing per trade and behaves identically every run. Claude Code is
   never part of it.

2. **Offline learning loop — where improvement happens.** Runs on this GPU box,
   driven by the human + Claude Code: research, backtest, train, gate, deploy. It
   produces the artifact (a champion model) that the live engine consumes.

The only thing crossing from the offline brain to the live brain is a **vetted
model artifact** in the model registry. Nothing else leaks across.

---

## Live engine — the per-decision path

```
            ┌─────────────┐     ┌──────────┐     ┌─────────────────────┐
 IBKR  ───▶ │  data/      │ ──▶ │ features/│ ──▶ │ models/ (inference) │
 (live)     │  ingestion  │     │ (no      │     │  LightGBM signal     │
 news/macro │             │     │  leakage)│     │  + sentiment feature │
            └─────────────┘     └──────────┘     └──────────┬──────────┘
                                                            │ signal
                                                            ▼
                                                   ┌─────────────────┐
                                                   │ strategy/       │
                                                   │ (nautilus       │
                                                   │  Strategy)      │
                                                   └────────┬────────┘
                                                            │ proposed order
                                                            ▼
                                               ┌────────────────────────────┐
                                               │ risk/  (SACROSANCT)         │
                                               │ size · cap · daily kill ·   │
                                               │ veto/flatten · broker stop  │
                                               └────────────┬───────────────┘
                                                            │ approved order
                                                            ▼
                                               ┌────────────────────────────┐
                                               │ execution/                  │
                                               │ nautilus live node →        │
                                               │ ib_async → IBKR (PAPER)     │
                                               └────────────┬───────────────┘
                                                            │ fills / state
                                                            ▼
                                   ┌────────────────────────────────────────────┐
                                   │ observability/   logs (DuckDB/Parquet) +    │
                                   │ Streamlit dashboard + Telegram alerts        │
                                   └────────────────────────────────────────────┘
```

Reading it as a sentence: **data** comes in from IBKR (and later news/macro) and
is stored; **features** turns stored data into point-in-time-correct inputs;
**models** runs LightGBM (plus a sentiment feature) to produce a signal;
**strategy** (a `nautilus_trader` Strategy) turns the signal into a proposed
order; **risk** sizes/caps/vetoes it and guarantees a broker-side stop;
**execution** sends the approved order through `ib_async` to the IBKR **paper**
account; and everything is **logged** and surfaced via the dashboard + Telegram.

Two properties make this trustworthy:
- **Risk is in the path of every order.** Nothing reaches the broker unchecked,
  and every open position rests a hard stop **at IBKR**, so an outage can't leave
  a position naked.
- **Strategy code is the same in backtest and live**, because both run inside
  `nautilus_trader`. That kills the classic "worked in backtest, broke live" gap.

---

## Offline learning loop — how a model earns its place

```
 Databento history ─┐
                    ├─▶ backtest/  ──▶ models/ (train) ──▶ champion–challenger gate ──▶ registry ──▶ live
 logged live data ──┘   nautilus +     LightGBM, FinBERT/      (out-of-sample,           (models/)     engine
                        walk-forward    8B-LLM (GPU)            realistic costs)
```

1. **Backtest** replays history through the *same* strategy code with realistic
   costs and slippage, using **walk-forward** windows (train on a window, test on
   the next, roll forward) — never a single in-sample fit.
2. **Train** the LightGBM signal model (and the sentiment scorer) on the GPU box.
3. **Gate (champion vs challenger):** a new model is promoted only if it beats the
   current champion on out-of-sample results. **Overfitting is the enemy** — good
   in-sample numbers count for nothing.
4. The promoted model lands in the **registry** under `models/`, and the live
   engine picks it up. That hand-off is the only bridge between the two brains.

---

## Data layer in detail (Phase 1.1, implemented)

```
 IBKR paper (ib_async, read-only)
   ├─ reqRealTimeBars(5s) ─┐
   └─ reqMktData (L1)  ────┤
                           ▼
                  data/recorder.py ──(append-only, dedup)──▶ Parquet lake (data/)
                           │                                  bars_5s · bars_1m ·
                  5s→1m aggregation,                          quotes_l1 · trades ·
                  RTH/ETH tagging,                            roll_events
                  ts_event + ts_ingest                                │
                                                                      ▼
   Databento history ──(scaffold, cost-guarded)──▶ same lake     data/store.py (DuckDB)
                                                                  get_bars / get_quotes_l1
                                                                  asof_join  ← leak-free
                                                       ┌──────────────┼──────────────┐
                                                       ▼              ▼              ▼
                                              continuous.py     validate.py   observability/
                                              roll + back-adj   dupes/gaps/   data_health.py
                                              (raw kept)        ohlc/ingest   (Streamlit)
```

The contract with the rest of the system: **everything reads through
`store.py`**, whose `asof_join` makes look-ahead impossible. Raw per-contract
data is immutable; the continuous (back-adjusted) series is derived on demand.
Live (IBKR, forward from today) and historical (Databento) land in the *same*
schema, so they're queried identically.

## Feature layer in detail (Phase 2, implemented)

```
 store.get_bars(continuous=True)              config/features.yaml
   (outright front-month, back-adjusted)   →  features/config.py (typed, versioned)
              │                                         │
              ▼                                         ▼
   features/compute.py  ── causal, degree-0 ──▶  features/build.py
   ~45 price/vol/volume/time/cross-asset       (versioned tables → data/features/,
   features; trailing windows only             idempotent) + read_features / features_asof
              │                                         │
              ▼                                         ▼
   tests/test_features_leakage.py            observability/features_health.py
   truncation-invariance proof               (coverage, null rates, degraded)
```

The defining property: a feature at `t` uses only bars with `ts_event <= t`.
Price features are **degree-0 in the price scale** (returns/ratios/normalized) so
they survive ratio back-adjustment — a price *level* on the continuous series
would secretly encode the future roll. The leakage test rebuilds the series
point-in-time and proves `feature(full)[t] == feature(truncated)[t]`. Feature
tables are versioned (`feature_version`), carry `session` and a `degraded` flag,
and are retrieved leak-free through the store's `asof_join`. Catalog:
`docs/FEATURES.md`; rationale: `docs/DECISIONS.md` Phase 2.

## Where each module lives

| Concern | Module | One-liner |
|---|---|---|
| shared plumbing | `common/` | logging, config, shared types |
| data in + storage | `data/` | IBKR live + historical backfill → DuckDB/Parquet |
| model inputs | `features/` | leakage-safe, point-in-time features |
| text sentiment | `sentiment/` | FinBERT (GPU), optional 8B via Ollama |
| signal model | `models/` | LightGBM train + registry + champion–challenger |
| decisions | `strategy/` | nautilus Strategy (Claude-researched later) |
| safety | `risk/` | sizing, caps, kill-switch, broker-side stops |
| orders | `execution/` | nautilus live node → ib_async → IBKR paper |
| evaluation | `backtest/` | nautilus backtest + walk-forward |
| visibility | `observability/` | Streamlit dashboard + Telegram alerts |

Typed configuration for all of it is in `config/` (`settings.py` + `config.yaml`),
loaded once and shared. See `CLAUDE.md` for the project anchor and `docs/` for the
glossary, decisions, and setup.
