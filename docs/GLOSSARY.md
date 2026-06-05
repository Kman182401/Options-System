# Glossary

Plain-English definitions of the terms used across this project. Kept short on
purpose — the goal is shared vocabulary, not exhaustive theory.

## Instruments & markets
- **Futures contract** — an agreement to buy/sell an asset at a set price on a
  future date, traded on an exchange. We trade them intraday, not to expiry.
- **CME** — the Chicago Mercantile Exchange (the venue for our futures).
- **MES** — Micro E-mini S&P 500 future. A small-notional future tracking the
  S&P 500. Our primary Phase 1 instrument (small size = small risk while building).
- **MNQ** — Micro E-mini Nasdaq-100 future. The Nasdaq-100 counterpart to MES.
- **Micro future** — a fractional-size version of a standard future (e.g. MES is
  1/10 the size of the E-mini ES), so each tick is worth less — ideal for testing.
- **Front month** — the nearest-to-expiry contract, which is the most actively
  traded. Futures have several listed expiries; we trade the front month and
  "roll" to the next as expiry approaches.
- **Tick** — the smallest price increment a contract can move.
- **Vertical spread (Phase 2)** — an options strategy: buy one option and sell
  another of the same type and expiry but a different strike. Risk and reward are
  both **defined** (capped) up front — which is why it's the planned options step.

## Trading mechanics
- **Paper trading** — trading against a simulated/broker-provided practice account
  with no real money. This system is **paper-only** until explicitly changed.
- **Intraday** — positions opened and closed within the day (we allow up to ~3-day
  holds, but the style is short-term).
- **Stop-loss (broker-side)** — a resting order **held at the broker (IBKR)** that
  auto-closes a position at a set adverse price. Because it lives at the broker, it
  still fires if our machine or internet dies.
- **Slippage** — the difference between the price you expected and the price you
  actually got. Backtests must model it or they lie.
- **Fill** — an executed order (fully or partially).

## Modeling & validation
- **Signal model** — the model that outputs a trading signal from features. Here:
  **LightGBM** (gradient-boosted decision trees — fast, interpretable).
- **Feature** — a numeric input to the model derived from data (a return, a
  volatility measure, a sentiment score, time-of-day, …).
- **Leakage / look-ahead bias** — accidentally using information from the future
  (relative to the decision time) when computing a feature or label. It inflates
  backtests and destroys live performance. `features/` is built to prevent it.
- **In-sample vs out-of-sample** — data the model was trained on vs data it has
  never seen. Only out-of-sample results are trusted.
- **Overfitting** — a model that memorizes noise in the training data and fails on
  new data. The project's stated enemy.
- **Walk-forward analysis** — repeatedly train on a past window and test on the
  next unseen window, rolling forward through history. A realistic stand-in for
  how the model would have performed live.
- **Champion–challenger** — keep a live "champion" model; a new "challenger" only
  replaces it if it beats it on honest out-of-sample evaluation.
- **Model registry** — versioned storage of trained model artifacts + their
  metrics, under `models/`. The live engine loads the approved champion from here.

## Sentiment & ML tooling
- **FinBERT** — a BERT language model fine-tuned on financial text; our baseline
  sentiment scorer (positive / negative / neutral), run on the GPU.
- **Ollama** — a local LLM runner; the optional path to an ~8B model for richer
  sentiment. Local and offline; never in the live trade loop.

## System & infra
- **nautilus_trader** — the trading engine. Its key value here: the **same
  strategy code runs in backtest and live**, giving backtest = live parity.
- **ib_async** — the Python library that talks to Interactive Brokers (the
  maintained successor to `ib_insync`). Needs IB Gateway/TWS running.
- **IB Gateway / TWS** — Interactive Brokers' connection apps. Gateway is the
  lightweight, headless-friendly one we use (paper account).
- **DuckDB / Parquet** — our local data store: Parquet files (columnar on-disk
  format) queried with DuckDB (an in-process analytical SQL engine). No server.
- **Risk Manager** — the sacrosanct safety layer between every decision and every
  order; can veto, resize, or flatten, and always rests a broker-side stop.
- **Two brains** — the deterministic **live engine** vs the **offline learning
  loop**; see `docs/ARCHITECTURE.md`.
