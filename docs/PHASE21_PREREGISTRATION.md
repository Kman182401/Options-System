# Phase 21 Pre-Registration — Volatility-Forecast Skill Verdict

**Committed before any Phase 21 model is trained or any data reaches a model.** The commit date of
this file is the pre-registration timestamp.

Nothing here authorizes strategy, economic backtest, risk, execution, or live trading. Phase 21
produces a **forecast-skill verdict only**: does a machine-learning model forecast realized volatility
more accurately than the standard econometric benchmark? The system stays hard-locked to paper.

## Why this pivot, and the honest prior

Six levers targeting intraday *direction* returned honest nulls through the fixed framework. Direction
is the efficiently-priced, hard-to-forecast quantity. **Realized volatility is different**: it clusters
and mean-reverts, and is one of the most forecastable quantities in finance. Unlike the directional
levers, there is published precedent that gradient-boosted trees (the project’s existing LightGBM
stack) can beat the standard benchmark on the standard loss with statistical significance. So a
*positive* result is genuinely plausible here — but the benchmark (HAR-RV) is strong and parsimonious,
and an honest null (ML does not beat it on our data) remains entirely possible and fully acceptable.
**Forecast skill is not tradeable money** — beating the benchmark authorizes only a later economic-value
study, never trading.

## The single question Phase 21 answers

Does a LightGBM model, using the standard HAR predictors **plus** the project’s existing feature blocks,
forecast h-day-ahead realized volatility **more accurately than the HAR-RV benchmark**, by the standard
QLIKE loss, with a statistically significant Diebold-Mariano improvement that is robust across
volatility regimes — **per symbol**?

## The realized-volatility target (FIXED a priori)

- **Source:** the existing `bars_1m` 1-minute OHLCV lake (MES, MNQ, 2019→2026), regular trading hours.
- **Daily RV estimator:** daily realized variance = the **average of the five 5-minute sub-sampled
  grids** of summed squared 5-minute log returns within each RTH session (the noise-reduced 5-minute
  realized-variance estimator; 5-minute sampling is the literature standard for balancing accuracy
  against microstructure noise, and the five offset grids reduce estimator noise). 5-minute returns are
  resampled from the 1-minute bars.
- **Forecast target:** `y_t = log( mean of daily RV over the next h trading days )` — the standard
  multi-horizon HAR target in logs. **Primary horizon h = 5** (one trading week — multi-day,
  options-relevant, less noisy than 1-day). **Secondary diagnostic horizons h = 1 and h = 22** are
  reported but **not gated**.

## The benchmark — arm B (FIXED a priori)

**HAR-RV (Corsi 2009)** on log-RV: an OLS regression of `y_t` on three causal predictors — the daily
lag `log RV_t`, the weekly average `log( mean RV over the last 5 days )`, and the monthly average
`log( mean RV over the last 22 days )`. This is *the* standard realized-volatility benchmark; beating it
is the bar (the analog of “beat buy-and-hold” in the directional phases — skill is measured **over and
above HAR-RV**, never as a raw R²).

## The treatment — arm T (FIXED a priori)

A **single, fixed, heavily-regularized LightGBM regressor** (no hyperparameter search) on `y_t`, with
features = the three HAR predictors **plus** the project’s existing leak-safe feature blocks
(price/vol/volume/time/cross-asset from `features/`, macro-event features, and — where available on the
covered dates — the OFI and `s2` sentiment aggregates). The objective targets the QLIKE loss on the
variance scale (a QLIKE-consistent custom objective); if a custom objective proves infeasible, the
disclosed fallback is squared-error on log-RV — this fallback must be stated, never silently chosen.
The config reuses the `mm1` regularization philosophy (shallow trees, strong L1/L2, sub-1
bagging/feature fractions, `seed 7`), fixed a priori.

**Why a single fixed config and no search:** the Phase-19 and Phase-20 near-misses both died on
selection overfit (PBO blow-out from searching configs). Fixing one regularized config a priori removes
the selection-overfit failure mode entirely, so the verdict is a clean Diebold-Mariano accuracy test
with nothing to deflate. No grid, no tuning.

## Loss and metrics (FIXED a priori)

- **Primary loss: QLIKE** on the variance scale (the literature’s preferred loss — it has the highest
  power in the Diebold-Mariano test and is robust to the noise in the RV proxy). Lower is better.
- **Secondary (reported, not gated):** RMSE and MAE on the forecasts; a Mincer-Zarnowitz calibration
  regression (regress realized on forecast; report intercept and slope) as an unbiasedness diagnostic.

## The comparison test (FIXED a priori)

**Diebold-Mariano**, one-sided, on the per-period QLIKE loss differential (treatment vs HAR-RV), at the
**5% level**, computed on the pooled out-of-sample forecasts, **per symbol**. (Because the treatment is
a single fixed config, there is no multiple-comparison / selection problem to correct.)

## Validation (FIXED a priori)

**Anchored expanding-window walk-forward** out-of-sample evaluation (the time-series standard): an
initial training window of 2019-01 → 2021-12, then expanding-window refits forward, scoring
2022-01 → 2026-06 out-of-sample. **Purge/embargo for the forward target:** the h-day forward RV makes
adjacent targets overlap, so the `h − 1` days straddling each train/test boundary are dropped — the
same overlap discipline as the triple-barrier `t1`. **Regime split (for G2):** each OOS day is labeled
**turbulent** if its trailing-22-day average RV is above the *causal expanding-window median*, else
**calm** — a fixed, leak-safe a-priori rule.

## Sample (FIXED)

Full `bars_1m` history 2019→2026, **per symbol (MES, NQ/MNQ separately), never pooled**. ~1,700 trading
days per symbol — a large power improvement over the ~1,000 intraday events of the directional phases.

## Verdict gates (FIXED a priori)

A **volatility-forecast skill candidate** (per symbol) must clear **both**:

|Gate                      |Threshold                                                                                                                                  |
|--------------------------|-------------------------------------------------------------------------------------------------------------------------------------------|
|**G1 — accuracy**         |Treatment OOS QLIKE strictly lower than HAR-RV’s at h = 5, with a one-sided Diebold-Mariano test significant at the 5% level               |
|**G2 — regime robustness**|The QLIKE improvement is sign-consistent (treatment ≤ HAR-RV) in **both** the calm and turbulent OOS sub-samples (not driven by one regime)|

Calibration (Mincer-Zarnowitz), the h = 1 and h = 22 horizons, and per-feature SHAP are **reported,
not gated**.

## Decision rule (FIXED a priori)

- **Both symbols clear G1 and G2** → “volatility-forecast skill candidate” → authorizes **only** a
  future Phase 22 **economic-value study** (does the more accurate forecast improve position-sizing /
  risk overlay / option-pricing decisions, evaluated with realistic costs?). **Never** authorizes live
  trading directly. A single-symbol pass while the other fails is flagged **fragile**.
- **Either symbol fails** → no skill candidate. Given the published precedent that trees can beat
  HAR-RV, a null here is informative in its own right (the benchmark is hard to beat on these specific
  instruments / features) and is recorded honestly in `docs/RESEARCH_VERDICTS.md`. The volatility lever
  is then re-scoped only by deliberate operator decision, not auto-re-litigated.

## Anti-snooping commitments (FIXED)

- The RV estimator, the horizon set, the HAR benchmark form, the treatment feature set, the single fixed
  LightGBM config, the QLIKE loss, the DM test and its 5% level, the walk-forward scheme, the regime
  split rule, and the gates are all fixed before any modeling and are not changed after seeing results.
- **No hyperparameter search**, no horizon-fishing, no estimator-swapping to find a pass. Each arm runs
  once per symbol.
- If the QLIKE custom objective is infeasible and the squared-error-on-log-RV fallback is used, that is
  **disclosed** in the results, not silently substituted.
- If the pipeline must change for a legitimate engineering reason, the model version is bumped and the
  change is documented in `docs/DECISIONS.md`.

## What this document is NOT

Not an implementation and not a model run. It commits zero data and zero models. The Phase 21
implementation prompt will reference this frozen contract and must not deviate from it.
