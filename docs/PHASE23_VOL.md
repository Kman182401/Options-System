# PHASE23_VOL.md — 1-day RV-forecast CONFIRMATION verdict (Phase 23)

## What it is
Phase 21 found, as a *non-gated diagnostic*, that the fixed LightGBM treatment beats HAR-RV at the
**1-day** horizon on both symbols — the program's first benchmark-beating signal — while its
pre-registered week-ahead horizon (h = 5) was a null. Because h = 1 was chosen *after* seeing three
horizons, Phase 23 is a **confirmation study**, not a re-run: it re-asks the h = 1 question under a
**materially harder, pre-registered bar** (frozen contract: `docs/PHASE23_PREREGISTRATION.md`,
`config/phase23_vol_h1.yaml`, `volatility_version=vf2`). Forecast-skill verdict only — forecast
skill is not tradeable money. Reads only local lakes; no Databento/IBKR/network/spend.

## Identical to Phase 21 (a true confirmation, not a new model search)
Same 5-minute sub-sampled RV target, same single fixed regularized LightGBM (seed 7, **no search**),
**same existing leak-safe feature set** (the Phase-22 free-data blocks stay OFF — "confirm first"),
same anchored expanding walk-forward (OOS 2022→2026, 18 folds), same regime split, same symbol set
`{MES, MNQ}` per symbol, same QLIKE + HAC/HLN Diebold-Mariano machinery. Verified: the treatment and
HAR QLIKE at h = 1 reproduce the Phase-21 diagnostic **byte-for-byte** (MES treat 0.246401 / HAR
0.308148; MNQ 0.224083 / 0.267216).

## What is harder (the new bar)
- **Gated horizon is now h = 1** (the overlap is 0, so the DM test is at its cleanest).
- **Four-benchmark battery** — the treatment must beat **all** of: HAR-RV, **random walk** (today's
  RV), **EWMA/RiskMetrics** (λ = 0.94), and **GARCH(1,1)** (Gaussian QMLE, dependency-free `scipy`,
  refit per walk-forward fold; innovations are **RTH-internal** returns — `log(last RTH close) −
  log(first RTH close)` — so they exclude the overnight gap and sit on the same window as the RV
  target).
- **Four gates, all required per symbol:** G1 accuracy vs HAR (QLIKE↓ + significant one-sided DM);
  G2 regime robustness (treat ≤ HAR in both calm and turbulent); **G3** beat each challenger with a
  significant DM; **G4** beat HAR and RW in ≥ 13/18 folds (the one-sided 5% binomial sign-test
  threshold).

## Result — measured 2026-06-23 (OOS n = 1,139 per symbol)

### Gated horizon h = 1 — all four gates PASS, both symbols
| Symbol | QLIKE treat | HAR | RW | EWMA | GARCH | DM p (vs each) | G1 | G2 | G3 | G4 | verdict |
|--------|------------:|----:|---:|-----:|------:|----------------|----|----|----|----|---------|
| **MES** | 0.2464 | 0.3081 | 0.7602 | 0.3602 | 0.3311 | 0.0 / 1e-5 / 0.0 / 0.0 | ✓ | ✓ | ✓ | ✓ | **confirmed** |
| **MNQ** | 0.2241 | 0.2672 | 0.6947 | 0.2951 | 0.2845 | 1e-6 / 2e-6 / 3e-6 / 5e-6 | ✓ | ✓ | ✓ | ✓ | **confirmed** |

The treatment has the lowest QLIKE of every forecaster and beats each with an overwhelmingly
significant one-sided DM on both symbols.

### Regime robustness (G2) — the contrast with the h = 5 null
At h = 5 the treatment *degraded in turbulence* (the gate that killed Phase 21). At h = 1 it beats
HAR in **both** regimes:

| Symbol | calm QLIKE (HAR / treat) | turbulent QLIKE (HAR / treat) |
|--------|--------------------------|-------------------------------|
| **MES** | 0.3174 / **0.2390** ✓ | 0.2975 / **0.2549** ✓ |
| **MNQ** | 0.2677 / **0.2115** ✓ | 0.2667 / **0.2390** ✓ |

### Temporal stability (G4) and GARCH health
Treatment beats HAR in **16/18** folds and RW in **18/18** on both symbols (threshold 13/18). The
GARCH(1,1) benchmark **converged on 17/18 folds (MES) and 18/18 (MNQ)** — the one MES fallback was
the disclosed carry-forward of the previous fold's parameters, never a silent RiskMetrics swap.

### Where the skill comes from (SHAP, reported)
Block shares ≈ HAR 0.24 / other price-vol 0.76 / sentiment 0.00 (marketdata & GKG are off). The edge
is intraday realized-vol structure (e.g. `rv_390`, `gk_30`) predicting next-day RV more accurately
than HAR's daily/weekly/monthly lags — economically sensible, and consistent with the literature
that trees beat HAR at short horizons.

## Honest interpretation
- **The strawman worry was disproved, not confirmed.** We hardened the bar fearing HAR was easy to
  beat at one day. In fact HAR turned out to be the **toughest** classical benchmark (lowest QLIKE of
  the four); RW, EWMA and the (correctly RTH-scaled) GARCH were all easier. The treatment beats the
  hardest of them anyway, in every regime, in 16/18 folds.
- **This is forecast skill, not money.** Per the frozen decision rule, the confirmed result
  authorizes **only** the Phase-24 free-data incremental study (does `x1`/`s3` improve it?) and,
  later, an economic-value study. **No strategy, backtest, risk, execution, or live trading is
  authorized.** The system stays hard-locked to paper.

> Forecast-skill verdict, not a strategy and not an economic backtest.

## Run it
```sh
uv run python -m options_system.volatility.run_h1 --symbols MES MNQ
```
Per-symbol summaries → `data/volatility/runs_h1/<symbol>.json`; combined verdict →
`data/volatility/runs_h1/verdict.json` (gitignored). MLflow experiment `volatility-forecast-h1`.
Tests: `tests/test_phase23_vol_h1.py`.
