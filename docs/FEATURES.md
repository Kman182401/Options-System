# Feature Catalog (`feature_version = v1`)

Every feature the engine (`features/compute.py`) emits, one row each. This is a
primary "understand the whole system" artifact — it is kept exactly in sync with
the code by `tests/test_features_catalog.py` (every emitted feature appears here,
and nothing here is an orphan).

**Leakage contract.** All features are **causal** (trailing windows only) and all
price-derived features are **degree-0 in the price scale** (returns / ratios /
normalized), which makes them invariant to ratio back-adjustment — see
`docs/DECISIONS.md` Phase 2 and the truncation-invariance test
(`tests/test_features_leakage.py`). Windows are in **bars** (= minutes).

- **Basis** = which series the feature reads: `continuous` (back-adjusted
  front-month), `raw-front` (= `continuous.close / adj_factor`, the true recorded
  price, used only inside ratios), or `time`/`volume` (price-scale independent).
- **Roll-safe?** = passes truncation-invariance. All are `yes`; the column notes
  *why* (what makes it degree-0 / price-independent).

## Returns

| name | family | definition | formula | window | basis | roll-safe? |
|---|---|---|---|---|---|---|
| `ret_1` | returns | 1-bar log return | `ln(close_t) − ln(close_{t−1})` | 1 | continuous | yes — log-diff cancels the global factor |
| `ret_5` | returns | 5-bar log return | `ln(close_t) − ln(close_{t−5})` | 5 | continuous | yes — log-diff |
| `ret_15` | returns | 15-bar log return | `ln(close_t) − ln(close_{t−15})` | 15 | continuous | yes — log-diff |
| `ret_30` | returns | 30-bar log return | `ln(close_t) − ln(close_{t−30})` | 30 | continuous | yes — log-diff |
| `ret_60` | returns | 60-bar log return | `ln(close_t) − ln(close_{t−60})` | 60 | continuous | yes — log-diff |

## Trend / momentum

| name | family | definition | formula | window | basis | roll-safe? |
|---|---|---|---|---|---|---|
| `ema_slope_12` | momentum | slope of the 12-bar log-price EMA | `(EMA_12 − EMA_12[−5]) / 5` on `ln(close)` | 12 | continuous | yes — diff of log-EMA |
| `ema_slope_26` | momentum | slope of the 26-bar log-price EMA | as above, span 26 | 26 | continuous | yes — diff of log-EMA |
| `ema_slope_60` | momentum | slope of the 60-bar log-price EMA | as above, span 60 | 60 | continuous | yes — diff of log-EMA |
| `ema_slope_120` | momentum | slope of the 120-bar log-price EMA | as above, span 120 | 120 | continuous | yes — diff of log-EMA |
| `ema_dist_z_12` | momentum | distance of price from its EMA, z-scored | `(ln(close) − EMA_12) / rolling_std(·, 12)` | 12 | continuous | yes — ratio of log-distances |
| `ema_dist_z_26` | momentum | price-vs-EMA distance, z-scored | span/window 26 | 26 | continuous | yes — ratio |
| `ema_dist_z_60` | momentum | price-vs-EMA distance, z-scored | span/window 60 | 60 | continuous | yes — ratio |
| `ema_dist_z_120` | momentum | price-vs-EMA distance, z-scored | span/window 120 | 120 | continuous | yes — ratio |
| `macd` | momentum | MACD line on **log** price | `EMA_12(ln close) − EMA_26(ln close)` | 12/26 | continuous | yes — log-EMA difference |
| `macd_signal` | momentum | MACD signal line | `EMA_9(macd)` | 9 | continuous | yes — difference of log-EMAs |
| `macd_hist` | momentum | MACD histogram | `macd − macd_signal` | 12/26/9 | continuous | yes — difference |
| `roc_15` | momentum | 15-bar rate of change | `close_t / close_{t−15} − 1` | 15 | continuous | yes — price ratio |
| `roc_60` | momentum | 60-bar rate of change | `close_t / close_{t−60} − 1` | 60 | continuous | yes — price ratio |
| `adx_14` | momentum | Wilder ADX (trend strength, 0–100) | Wilder smoothing of `DX = 100·|+DI−−DI|/(+DI+−DI)` | 14 | continuous | yes — DI are ratios (DM/ATR) |

## Mean reversion

| name | family | definition | formula | window | basis | roll-safe? |
|---|---|---|---|---|---|---|
| `rsi_14` | mean_reversion | Wilder RSI (0–100) | `100 − 100/(1+RS)`, `RS = avgGain/avgLoss` | 14 | continuous | yes — gain/loss ratio |
| `bb_pctb_20` | mean_reversion | Bollinger %B (position in band) | `(close − lower)/(upper − lower)`, bands `SMA±2σ` | 20 | continuous | yes — ratio of price distances |
| `zscore_30` | mean_reversion | rolling z-score of price | `(close − SMA_30)/std_30` | 30 | continuous | yes — z-score |
| `zscore_120` | mean_reversion | rolling z-score of price | `(close − SMA_120)/std_120` | 120 | continuous | yes — z-score |

## Volatility

| name | family | definition | formula | window | basis | roll-safe? |
|---|---|---|---|---|---|---|
| `rv_15` | volatility | realized vol (std of 1-bar returns) | `std(ret_1, 15)` | 15 | continuous | yes — function of returns |
| `rv_60` | volatility | realized vol | `std(ret_1, 60)` | 60 | continuous | yes — returns |
| `rv_390` | volatility | realized vol (one RTH session) | `std(ret_1, 390)` | 390 | continuous | yes — returns |
| `atr_pct_14` | volatility | ATR as a fraction of price | `Wilder_ATR_14 / close` | 14 | continuous | yes — ATR/price ratio |
| `parkinson_30` | volatility | Parkinson high–low range vol | `sqrt( mean(ln(high/low)², 30) / (4 ln2) )` | 30 | continuous | yes — log high/low ratio |
| `gk_30` | volatility | Garman–Klass vol | `sqrt(mean(0.5·ln(h/l)² − (2ln2−1)·ln(c/o)², 30))` | 30 | continuous | yes — log price ratios |
| `vol_regime_60` | volatility | current vol vs its rolling median | `rv_60 / rolling_median(rv_60, 1380)` | 60/1380 | continuous | yes — ratio of vols |

## Volume

| name | family | definition | formula | window | basis | roll-safe? |
|---|---|---|---|---|---|---|
| `rvol_30` | volume | relative volume vs 30-bar mean | `volume / rolling_mean(volume, 30)` | 30 | volume | yes — volume is back-adj independent |
| `rvol_390` | volume | relative volume vs session mean | `volume / rolling_mean(volume, 390)` | 390 | volume | yes — volume independent |
| `rvol_tod` | volume | volume vs time-of-day baseline | `volume / mean(volume at same minute-of-day, prior 20 days)` | 20 days | volume | yes — volume independent; baseline excludes current day |
| `vol_z_390` | volume | rolling z-score of volume | `(volume − mean_390)/std_390` | 390 | volume | yes — volume independent |
| `obv_norm_60` | volume | normalized signed-volume flow (−1..1) | `Σ sign(ret)·volume / Σ volume` over 60 | 60 | volume | yes — sign × volume, bounded ratio |
| `vwap_dist` | volume | distance of price from session VWAP | `close / VWAP_session − 1` (VWAP resets each CME trade date) | session | continuous | yes — close/VWAP ratio |

## Time / session (cyclical, price-independent)

| name | family | definition | formula | window | basis | roll-safe? |
|---|---|---|---|---|---|---|
| `mod_sin` | time | sine of minute-of-day | `sin(2π·minute_of_day/1440)` (ET) | — | time | yes — depends only on `ts_event` |
| `mod_cos` | time | cosine of minute-of-day | `cos(2π·minute_of_day/1440)` | — | time | yes — time only |
| `dow_sin` | time | sine of day-of-week | `sin(2π·dow/7)` | — | time | yes — time only |
| `dow_cos` | time | cosine of day-of-week | `cos(2π·dow/7)` | — | time | yes — time only |
| `mins_since_rth_open` | time | minutes since the RTH open (signed) | `minute_of_day − 570` (09:30 ET) | — | time | yes — time only |
| `mins_to_rth_close` | time | minutes to the RTH close (signed) | `960 − minute_of_day` (16:00 ET) | — | time | yes — time only |

## Cross-asset (MES ↔ MNQ)

A feature for one symbol uses the other's **contemporaneous-or-earlier** data only
(attached by a backward as-of join). The ratio/spread are oriented MNQ-over-MES
consistently regardless of which symbol's row they sit on.

| name | family | definition | formula | window | basis | roll-safe? |
|---|---|---|---|---|---|---|
| `xa_ret_spread` | cross_asset | 1-bar return spread MNQ − MES | `ret_1(MNQ) − ret_1(MES)` | 1 | continuous | yes — difference of returns |
| `xa_ratio_z_390` | cross_asset | z-score of the raw MNQ/MES price ratio | `z( raw_MNQ / raw_MES , 390 )`, raw = `close/adj_factor` | 390 | raw-front | yes — uses true raw prices (no back-adjustment) |
| `xa_corr_60` | cross_asset | rolling correlation of 1-bar returns | `corr(ret_1(this), ret_1(other), 60)` | 60 | continuous | yes — function of returns |

## Carried metadata (not features)

`ts_event`, `ts_ingest`, `symbol`, `session` (RTH/ETH), `degraded` (warmup rows
or a Databento-flagged degraded day), `feature_version`.

## Hooks (built nothing yet)

`news` / macro features — no data ingested. The config has a disabled `news`
seat (`config/features.yaml`); the engine emits nothing for it.
