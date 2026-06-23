# Phase 23 Pre-Registration — 1-Day Realized-Volatility Forecast (Confirmation Study)

**Committed before any Phase 23 model is trained or any data reaches a model.** The commit date of
this file is the pre-registration timestamp.

Nothing here authorizes strategy, economic backtest, risk, execution, or live trading. Phase 23
produces a **forecast-skill verdict only**: does the project's fixed LightGBM model forecast
*1-day-ahead* realized volatility more accurately than a **battery of hard econometric benchmarks**?
The system stays hard-locked to paper. Reads only the local lakes — no Databento, no IBKR, no
network, no spend (`OPTIONS_DATABENTO_SPEND_OK` stays unset).

## Why this study, and the honest prior

Phase 21 ran the frozen volatility contract at its pre-registered horizon **h = 5** and returned an
honest null on both symbols. But its *reported, non-gated* diagnostic at **h = 1** showed the
LightGBM treatment beating HAR-RV with strong significance on both symbols (MES one-sided DM
p ≈ 0; MNQ p = 1e-6) — the **first benchmark-beating signal in the entire research program**, and
consistent with the literature prior that trees beat HAR at short horizons.

**The integrity problem this document exists to solve:** h = 1 was selected *after* looking at three
horizons {1, 5, 22}. Re-running the same code at h = 1 and declaring victory would be circular —
celebrating the horizon we cherry-picked. The defense is **not** a re-run; it is to subject the
h = 1 result to a **materially harder, pre-registered bar** — new hurdles the diagnostic never
faced. A result that survives challenges it was never tuned against is real; one that crumbles saved
us from trading noise.

Two threats dominate, and the new gates target them directly:
1. **Strawman benchmark.** HAR-RV is easy to beat at the 1-day horizon (its daily lag is only one of
   three OLS terms). Beating *only* HAR at h = 1 is the least impressive possible claim. → **G3**.
2. **Single-period artifact.** An average-over-OOS win can be driven by one regime (e.g. the 2022
   volatility spike). → **G4**.

The residual horizon-selection concern is minor by comparison: the h = 1 p-values survive a
Bonferroni correction over the three horizons by many orders of magnitude. The work here is to
defeat threats 1 and 2.

## The single question Phase 23 answers

Does the project's **single fixed, no-search LightGBM regressor**, on its **existing leak-safe
feature set** (unchanged from Phase 21), forecast **1-day-ahead** realized volatility **more
accurately — by QLIKE, with a significant one-sided Diebold-Mariano improvement — than EACH of four
benchmarks {HAR-RV, random walk, EWMA/RiskMetrics, GARCH(1,1)}, robustly across both volatility
regimes and across time — per symbol?**

## What is IDENTICAL to Phase 21 (so this is a true confirmation, not a new model search)

Held byte-for-identical to the frozen Phase-21 contract (`docs/PHASE21_PREREGISTRATION.md`):

- **RV target estimator:** 5-minute, five sub-sampled offset grids averaged, RTH only, from the
  `bars_1m` lake (MES, MNQ, 2019→2026). `min_5min_returns_per_session = 20`.
- **Treatment model — arm T:** the *same* single fixed, heavily-regularized LightGBM regressor
  (`seed 7`, **no hyperparameter search**), on the *same* existing leak-safe blocks (HAR predictors +
  price/vol/volume/time/cross-asset + macro-event + `s2` sentiment where covered). QLIKE-consistent
  custom objective on the variance scale with train-mean centering; disclosed fallback `l2_log_rv`.
  **No new features** — the Phase-22 free-data blocks (`with_marketdata` / `with_gkg`) stay **OFF**
  (that is Phase 24's separate, pre-registered question — "confirm first").
- **Validation:** anchored expanding-window walk-forward, OOS `2022-01-01` → end of data, 18
  expanding refits. Purge/embargo = `h − 1` days; **at h = 1 this is 0 days — non-overlapping,
  the cleanest possible setting for the DM test.**
- **Regime split (for G2):** OOS day is *turbulent* if its trailing-22-day mean RV exceeds the causal
  expanding-window median, else *calm*.
- **Symbol set:** EXACTLY `{MES, MNQ}`, **per symbol, never pooled.** The canonical verdict is defined
  over exactly these two; a subset/superset run is refused for the saved verdict.
- **DM machinery:** one-sided, 5% level, HAC/Newey-West variance at lag `h − 1` + Harvey-Leybourne-
  Newbold small-sample correction. At h = 1 the lag is 0 (no overlap), so the HAC term reduces to the
  plain DM variance — a feature, not a deviation.

## What is NEW in Phase 23 (the harder bar)

### Primary horizon (FIXED a priori)
**h = 1 trading day** is now the *gated* horizon (it was a Phase-21 diagnostic). h = 5 and h = 22 are
re-reported as diagnostics, **not gated**.

### Benchmark battery — the treatment must beat ALL FOUR (FIXED a priori)
All four forecast the same next-day RV and are scored on the same QLIKE loss over the same OOS days.
Requiring the treatment to beat **every** benchmark is a conservative *intersection* test (harder,
not a fishing expedition — there is no multiplicity to inflate when all must pass):

1. **HAR-RV (Corsi 2009)** — the Phase-21 benchmark (continuity; already beaten at h = 1).
2. **Random walk (RW)** — forecast = today's RV (a martingale in variance). Notoriously hard to beat
   one day ahead; the single most important hardness check.
3. **EWMA / RiskMetrics** — exponentially-weighted moving average of daily RV with **fixed
   λ = 0.94** (the RiskMetrics daily standard). No tuning of λ.
4. **GARCH(1,1)** — Gaussian QMLE GARCH(1,1) with a constant mean on the daily RTH log-return series;
   1-step-ahead conditional-variance forecast. *The* canonical volatility benchmark ("does anything
   beat a GARCH(1,1)?", Hansen & Lunde 2005). **Implementation (FIXED, dependency-free):** estimated
   by Gaussian quasi-MLE via `scipy.optimize.minimize` (L-BFGS-B) under bounds ω > 0, α ≥ 0, β ≥ 0,
   α + β < 1, variance-targeting initialization; **re-fit once per expanding walk-forward fold** on
   that fold's training data, then the conditional-variance recursion is rolled forward through the
   fold's OOS days using *observed* returns with the fold-train parameters **frozen** (pseudo-out-of-
   sample, no look-ahead on parameters). *Disclosed fallbacks (never silent):* if the optimizer fails
   to converge / hits the stationarity bound on a fold, carry forward the previous fold's converged
   parameters; if the first fold fails, fall back to RiskMetrics(λ = 0.94) for that fold. **Disclosed
   caveat:** GARCH models the variance of daily *returns* while HAR/RW/EWMA operate directly on the RV
   series; the small return-vs-RV scale difference is an inherent, well-documented property of the
   GARCH-vs-RV comparison, disclosed here and not corrected for.

### Verdict gates — per symbol, ALL FOUR must clear at h = 1 (FIXED a priori)

| Gate | Threshold |
|------|-----------|
| **G1 — accuracy vs HAR** | Treatment OOS QLIKE strictly below HAR-RV's at h = 1, one-sided DM significant at 5%. (continuity with Phase 21) |
| **G2 — regime robustness** | Treatment QLIKE ≤ HAR-RV in **both** the calm and turbulent OOS sub-samples (sign-consistent), computed at h = 1. |
| **G3 — benchmark hardness** | Treatment OOS QLIKE strictly below **each** of {RW, EWMA(0.94), GARCH(1,1)}, with a one-sided DM significant at 5% **against each**. Defends against the strawman threat. |
| **G4 — temporal stability** | Treatment QLIKE below HAR's in **≥ 13 of the 18** walk-forward folds **AND** below RW's in **≥ 13 of 18** folds. (13/18 is the one-sided 5% binomial sign-test threshold: under a 50/50 null, P(≥13/18) ≈ 0.048.) Defends against the single-period-artifact threat. |

Calibration (Mincer-Zarnowitz), the h = 5 and h = 22 horizons, per-feature SHAP, and the
per-benchmark QLIKE table are **reported, not gated**.

## Decision rule (FIXED a priori)

- **Per symbol: PASS iff G1 ∧ G2 ∧ G3 ∧ G4 all clear at h = 1.**
- **Both symbols PASS** → **"confirmed 1-day RV forecast skill."** This authorizes **only** (a)
  Phase 24 — the pre-registered incremental-value study of the Phase-22 free-data blocks
  (`with_marketdata` / `with_gkg`) on this same h = 1 target — and eventually an economic-value
  study. **It never authorizes live trading directly.** Forecast skill is not tradeable money.
- **Exactly one symbol passes** → flagged **fragile**; recorded, no skill candidate promoted.
- **Either/both fail** → honest null at h = 1 too; the volatility lever is then re-scoped only by a
  deliberate operator decision, not auto-re-litigated. Recorded in `docs/RESEARCH_VERDICTS.md`.

## Anti-snooping commitments (FIXED)

- The primary horizon (h = 1), the four-benchmark battery and every one of its fixed parameters
  (λ = 0.94; the GARCH spec, estimator, bounds, refit scheme, and disclosed fallbacks), the four
  gates and their thresholds (including 13/18), the QLIKE loss, the DM test and its 5% level, the
  walk-forward scheme, the regime rule, the feature set (existing only), and the single fixed LightGBM
  config are all fixed **before** any Phase-23 modeling and are **not** changed after seeing results.
- **No hyperparameter search, no horizon-fishing, no benchmark-swapping, no estimator-swapping** to
  manufacture a pass. Each arm runs once per symbol.
- Disclosed-not-silent fallbacks only: the QLIKE→`l2_log_rv` objective fallback and the GARCH
  non-convergence fallbacks above.
- The RV estimator is held fixed (not multiplied across variants) to avoid inflating the test count;
  estimator-sensitivity is a possible *future* robustness check, never a gate here.
- If the pipeline must change for a legitimate engineering reason, the `volatility_version` is bumped
  and the change is documented in `docs/DECISIONS.md`.

## Versioning & artifacts

- `volatility_version: "vf2"` (Phase 23; Phase 21 was `vf1`). Frozen knobs:
  `config/phase23_vol_h1.yaml` (this commit).
- Per-symbol summaries → `data/volatility/runs_h1/<symbol>.json`; combined verdict →
  `data/volatility/runs_h1/verdict.json` (gitignored). MLflow experiment `volatility-forecast-h1`.

## What this document is NOT

Not an implementation and not a model run. It commits zero data and zero models. The Phase 23
implementation will reference this frozen contract and must not deviate from it.
