# features/

**Turns raw stored data into model-ready features — without leaking the
future.** This module computes the indicators and transforms the signal model
consumes (returns, volatility, microstructure stats, time-of-day, sentiment
joins, etc.). Its defining constraint is **leakage safety**: every feature for
time `t` must be computable using only information available at or before `t`,
and the *exact same* code path must run in backtest and live so a feature can
never "see" data it wouldn't have had in production. Look-ahead bias here
silently destroys a strategy, so this module is written defensively and is
heavily unit-tested. No model training and no trading decisions live here —
just deterministic, point-in-time-correct feature construction.
