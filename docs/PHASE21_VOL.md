# PHASE21_VOL.md — Volatility-forecast skill verdict (Phase 21)

## What it is
The deliberate pivot after six honest nulls on intraday *direction*: redirect the proven leak-safe
framework at a genuinely forecastable target — **realized volatility** — and ask one pre-registered
question: does a LightGBM model forecast h-day-ahead RV **more accurately than the HAR-RV
benchmark**, by QLIKE, with a significant Diebold-Mariano improvement that holds across volatility
regimes — per symbol? The frozen contract is
[`docs/PHASE21_PREREGISTRATION.md`](PHASE21_PREREGISTRATION.md) (committed `9c56435`, before any
modeling). This run does not deviate from it.

It is a **forecast-skill verdict only** — **forecast skill is not tradeable money**. Beating HAR-RV
would authorize *only* a later Phase-22 economic-value study, never live trading. Nothing trades.
Reads only the local lakes — no Databento, no IBKR, no network, no spend.

## The realized-volatility target
- **Source:** the `bars_1m` 1-minute lake (MES, MNQ, 2019→2026), regular trading hours.
- **Daily RV:** the noise-reduced **5-minute sub-sampled** estimator — the average of the five
  offset-grid (origins 0–4) sums of squared 5-minute log returns within each RTH session.
- **Target:** `y_t = log( mean daily RV over the next h trading days )`. **Primary h = 5** (gated);
  **h = 1 and h = 22** are reported diagnostics, not gated.

## The two arms
- **Arm B — HAR-RV (Corsi 2009):** OLS of `y_t` on the three causal HAR predictors (daily/weekly/
  monthly trailing log-RV). The bar to beat (skill is measured *over and above* HAR-RV).
- **Arm T — fixed LightGBM:** a single heavily-regularized regressor (mm1 philosophy, `seed 7`,
  **no hyperparameter search** — which removes the selection-overfit failure mode that killed the
  Phase-19/20 near-misses), on the HAR predictors **plus** the existing leak-safe feature blocks
  (price/vol/volume/time/cross-asset, macro-event, and `s2` sentiment where covered). It minimizes a
  **QLIKE-consistent custom objective** on the variance scale (the target is centered on the train
  mean so the custom objective starts at the right level — LightGBM does not boost-from-average for a
  custom objective). The objective **actually used was `qlike`** (the disclosed L2-on-log-RV fallback
  was not needed).

## Validation
**Anchored expanding-window walk-forward**: train anchors from the start of history; OOS scoring from
`2022-01-01` to end of data (~2026-06), split into 18 expanding refits. The `h`-day forward-target
overlap is purged + embargoed (`h − 1` days) via the shared leak-safe purge primitive. The **regime
split** labels each OOS day *turbulent* if its trailing-22-day mean RV exceeds the **causal
expanding-window median**, else *calm*.

## Verdict gates
- **G1 — accuracy:** treatment OOS QLIKE strictly below HAR-RV at h = 5 with a one-sided
  Diebold-Mariano test (HAC/Newey-West variance at lag `h − 1` + Harvey-Leybourne-Newbold small-sample
  correction) significant at 5%.
- **G2 — regime robustness:** the QLIKE improvement is sign-consistent (treatment ≤ HAR) in **both**
  the calm and turbulent OOS sub-samples.

## Run — measured 2026-06-13
Daily base: **1,826 RTH sessions** per symbol (2 incomplete dropped), 155 treatment features;
sentiment coverage just **4.4%** of days (GDELT history is ~3 months of the 7-year window — `s2`
features are null elsewhere, kept not imputed). Objective used: **qlike**.

### Primary horizon h = 5 — the gated verdict (✓ pass / ✗ fail)

| Symbol | n OOS | QLIKE HAR | QLIKE treat | DM stat (HLN) | one-sided p | G1 | G2 | verdict |
|--------|-------|-----------|-------------|---------------|-------------|----|----|---------|
| **MES** | 1,135 | 0.20988 | 0.21904 | −0.869 | 0.807 | ✗ | ✗ | **no skill** |
| **MNQ** | 1,135 | 0.15666 | 0.16102 | −0.704 | 0.759 | ✗ | ✗ | **no skill** |

At h = 5 the treatment is **marginally worse** than HAR-RV (higher QLIKE) and the DM test is nowhere
near significant — HAR-RV wins.

### Regime detail at h = 5 (why G2 also fails)

| Symbol | calm QLIKE (HAR / treat) | turbulent QLIKE (HAR / treat) |
|--------|--------------------------|-------------------------------|
| **MES** | 0.1552 / **0.1420** ✓ | 0.2721 / **0.3067** ✗ |
| **MNQ** | 0.1146 / **0.1072** ✓ | 0.2063 / **0.2245** ✗ |

The treatment is **better in calm regimes but worse in turbulent ones** — it degrades exactly when
volatility matters most. G2 is the gate that catches this, and it correctly fails.

### Diagnostic horizons (reported, NOT gated)

| Symbol | h | QLIKE HAR | QLIKE treat | one-sided DM p | beats HAR? |
|--------|---|-----------|-------------|----------------|------------|
| **MES** | 1 | 0.30815 | **0.24639** | ~0.0 | **yes (sig.)** |
| **MES** | 22 | 0.21031 | 0.39011 | ~1.0 | no |
| **MNQ** | 1 | 0.26722 | **0.22404** | 1e-6 | **yes (sig.)** |
| **MNQ** | 22 | 0.15015 | 0.24197 | ~1.0 | no |

At the **1-day horizon the treatment beats HAR-RV significantly on both symbols** (G1 *and* G2 would
pass) — the first time across all phases that the ML model beats its benchmark on a significance
test. At h = 22 it loses badly (HAR's monthly component dominates the long horizon). The gated
horizon (h = 5) sits in between and is a null.

### Arm-T SHAP (which features drive the forecast)
- **MES:** `har_log_rv_w5`, `rv_390`, `har_log_rv_m22`, `gk_30`, `macro_tendency_nfp`,
  `macro_mins_since_fomc`. Block shares: HAR 0.23 / other-price-vol 0.77 / sentiment 0.00.
- **MNQ:** `har_log_rv_w5`, `har_log_rv_m22`, `rv_390`, `macro_tendency_nfp`,
  `macro_mins_since_fomc`, `gk_30`. Block shares: HAR 0.26 / other-price-vol 0.74 / sentiment 0.00.

The model leans ~75% on the existing price/vol features (intraday realized vol `rv_390`,
Garman-Klass `gk_30`) and ~25% on the HAR lags; sentiment contributes ~0% (4.4% coverage). The extra
blocks are used, but at h = 5 they do not improve on HAR out-of-sample.

### Calibration (Mincer-Zarnowitz, reported)
Treatment intercept/slope ≈ MES (1.12, 1.12), MNQ (1.25, 1.14) — slightly biased (slope > 1), i.e.
the treatment's h = 5 forecasts are modestly mis-calibrated, consistent with the null.

## Decision (per the frozen rule)
- **Per symbol, both G1 and G2 must clear at h = 5.** MES fails both; MNQ fails both.
- **DECISION: `no_significant_skill` — both symbols.** No skill candidate; no symbol authorized for a
  Phase-22 economic-value study.

## Honest interpretation
- **The verdict is an honest null at the pre-registered horizon (h = 5):** ML does not beat HAR-RV on
  these instruments at the one-week horizon, and it fails specifically in the turbulent regime.
- **But the lever is not dead — it is horizon-specific.** At **h = 1** the treatment beats HAR-RV
  with strong significance on **both** symbols (and would clear both gates). This is the first
  benchmark-beating signal in the whole research program, and it is consistent with the literature
  prior that trees can beat HAR at short horizons. The pre-registered gate was h = 5 (chosen as
  options-relevant and less noisy), so the verdict is null — but the h = 1 result is a concrete,
  pre-registered-diagnostic pointer for a future re-scope (a deliberate operator decision, not an
  auto-re-litigation): a 1-day-horizon volatility forecast is where ML demonstrably adds accuracy.
- **No model is promoted; nothing trades.** Forecast skill — even the real h = 1 skill — is not
  tradeable money; it would authorize only a Phase-22 economic-value study, never live trading.

> **This is a forecast-skill verdict, not a strategy and not an economic backtest.** No strategy,
> backtest, risk, execution, or live trading is authorized.

## Run it
```sh
# Volatility-forecast skill verdict, both symbols, full history:
uv run python -m options_system.volatility.run --symbols MES MNQ
#   flags: --no-mlflow  --no-interpret  --rebuild-cache

# Read-only summary (QLIKE table + DM + regimes + calibration + decision):
uv run python -m options_system.observability.volatility_health --symbols MES MNQ
```
Per-symbol summaries → `data/volatility/runs/<symbol>.json`; combined verdict →
`data/volatility/runs/verdict.json` (gitignored). MLflow experiment `volatility-forecast`.
