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

## What's built (Phase 2, `feature_version = v1`)

- **`config.py` + `config/features.yaml`** — declarative, validated feature
  config (families, windows in bars, params) with a versioned `feature_version`.
- **`compute.py`** — the causal Polars engine: ~45 interpretable price /
  volatility / volume / time / cross-asset features. Every window is trailing and
  every price feature is **degree-0 in price scale** (returns / ratios /
  normalized), so it is invariant to ratio back-adjustment — see the catalog and
  `docs/DECISIONS.md` Phase 2.
- **`build.py`** — versioned, idempotent feature-table writer into
  `data/features/symbol=…/date=…/` (computed over full history, so values never
  depend on the build window) plus leak-free retrieval (`read_features`,
  `features_asof` via the store's `ASOF JOIN`).
- **`docs/FEATURES.md`** — the catalog (one row per feature, with roll-safety),
  kept in sync with the engine by a test.
- **Leakage proof** — `tests/test_features_leakage.py` rebuilds the continuous
  series point-in-time and asserts `feature(full)[t] == feature(truncated)[t]`
  for every feature, catching both forward windows and the back-adjustment trap.

Run it:

```fish
uv run python -m options_system.features.build --symbols MES MNQ
uv run streamlit run src/options_system/observability/features_health.py
```

No labels/targets, models, strategy, or news features here — those are later
phases. The `news` config seat is a disabled placeholder (no data ingested).
