# TA Feature Catalog (`feature_version = v2`)

Every feature the TA engine (`ta/compute.py`) emits, one row each. This is an
additive, isolated layer alongside the v1 price catalog (`docs/FEATURES.md`) and
it deliberately does **not** duplicate v1's RSI / MACD / ADX / Bollinger / OBV /
z-scores. The table is kept exactly in sync with the code by
`tests/test_ta_catalog.py` (every emitted feature appears here, and nothing here
is an orphan).

**Leakage contract.** All features are **causal** (trailing windows only) and
every emitted feature is **degree-0 in the price scale** — a ratio of price
differences, or a difference of log-prices — which makes it invariant to ratio
back-adjustment (see the truncation-invariance test `tests/test_ta_leakage.py`).
Windows are in **bars** (= minutes). Every name is `ta_`-namespaced so it never
collides with the v1 or macro layers.

- **Basis** = which series the feature reads: `continuous` (back-adjusted
  front-month price) or `volume`. All are price-scale independent.
- **Roll-safe?** = passes truncation-invariance. All are `yes`; the column notes
  *why* (what makes it degree-0).

## Momentum / oscillators

| name | family | definition | formula | window | basis | roll-safe? |
|---|---|---|---|---|---|---|
| `ta_stoch_k_14` | stochastic | %K: close's position in the trailing high/low range | `100·(close − min(low,14)) / (max(high,14) − min(low,14))` | 14 | continuous | yes — ratio of price differences cancels the global factor |
| `ta_stoch_d_3` | stochastic | %D: 3-bar SMA of %K | `SMA(ta_stoch_k_14, 3)` | 3 | continuous | yes — SMA of a roll-safe ratio |
| `ta_cci_20` | cci | typical-price deviation scaled by mean abs deviation | `(tp − SMA(tp,20)) / (0.015·MAD(tp,20))`, `tp=(h+l+c)/3` | 20 | continuous | yes — numerator and MAD both scale by the factor |
| `ta_mfi_14` | mfi | volume-weighted RSI on typical price | `100·posMF / (posMF + negMF)` over `tp·volume`, 14 bars | 14 | continuous + volume | yes — money-flow ratio cancels the factor; diff sign unchanged |
| `ta_vi_plus_14` | vortex | upward vortex movement over true range | `Σ\|high − low₋₁\|(14) / Σ TR(14)` | 14 | continuous | yes — ratio of price-difference sums |
| `ta_vi_minus_14` | vortex | downward vortex movement over true range | `Σ\|low − high₋₁\|(14) / Σ TR(14)` | 14 | continuous | yes — ratio of price-difference sums |
| `ta_trix_15` | trix | 1-bar change of the triple-EWM of log price | `EWM³(ln close, 15)_t − EWM³(ln close, 15)_{t−1}` | 15 | continuous | yes — constant `ln(f_k)` cancels in the difference |
