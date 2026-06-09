# ta/

**Additive technical-analysis feature layer (`feature_version = v2`).** A curated
set of classic oscillators computed *locally* over the same continuous 1-minute
bars the v1 price layer uses — no external TA service, no new dependency. It sits
alongside the v1 price layer and the macro layer and **never duplicates** v1's
RSI / MACD / ADX / Bollinger / OBV / z-scores.

Like v1, its defining constraint is **leakage safety**: every feature for time `t`
is computable using only bars at or before `t`, and the *exact same* code path
runs in backtest and live. Every emitted feature is also **degree-0 in the price
scale** (a ratio of price differences, or a difference of log-prices), so it is
invariant to ratio back-adjustment.

## What's built (`feature_version = v2`, starter subset)

- **`config.py` + `config/ta.yaml`** — declarative, validated config (families,
  windows in bars, params) with a versioned `ta_feature_version`.
- **`compute.py`** — the causal Polars engine. Five indicator families, all
  `ta_`-namespaced:
  - `ta_stoch_k_14`, `ta_stoch_d_3` — Stochastic %K (close's position in the
    trailing 14-bar high/low range) and its 3-bar SMA. Bounded 0–100.
  - `ta_cci_20` — Commodity Channel Index: typical-price deviation from its SMA
    over the mean absolute deviation.
  - `ta_mfi_14` — Money Flow Index: volume-weighted RSI on typical price. 0–100.
  - `ta_vi_plus_14`, `ta_vi_minus_14` — Vortex indicator (directional movement
    over the summed true range).
  - `ta_trix_15` — 1-bar change of the triple-EWM of log(close) (smoothed ROC).
- **`build.py`** — versioned, idempotent table writer into
  `data/ta_features/symbol=…/date=…/` (computed over full history, so values never
  depend on the build window) plus leak-free retrieval (`read_ta`, `ta_asof` via
  the store's `ASOF JOIN`).
- **`docs/TA_FEATURES.md`** — the catalog (one row per feature, with roll-safety),
  kept in sync with the engine by `tests/test_ta_catalog.py`.
- **Leakage proof** — `tests/test_ta_leakage.py` rebuilds the continuous series
  point-in-time and asserts `feature(full)[t] == feature(truncated)[t]` for every
  feature, catching both forward windows and the back-adjustment trap.

Run it:

```fish
uv run python -m options_system.ta.build --symbols MES MNQ
```

## Isolation & model opt-in

This layer is **purely additive**. It writes only to `data/ta_features/`, never
touches the v1 `data/features/` or the microstructure `data/micro_bars/` lakes.

As of Phase 10 it can be **opted into** model training via `with_ta` — but it is
**off by default**, so the canonical price+macro model is unchanged unless TA is
explicitly requested:

```fish
# Build the lake, then run the honest opt-in comparison through the same gates
uv run python -m options_system.ta.build --symbols MES MNQ
uv run python -m options_system.models.run --symbols MES MNQ --compare-ta
```

`load_training_matrix(..., with_ta=True)` appends the TA columns after price+macro
(backward as-of at each label `t0`); `--with-ta` runs a single TA-enabled model and
`--compare-ta` runs price+macro vs price+macro+TA side-by-side. See `docs/MODEL.md`
("Phase 10 — opt-in TA v2 controlled experiment"). No labels/targets, strategy, or
risk logic live here — just deterministic, point-in-time-correct feature construction.
