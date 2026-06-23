# Phase 24 Pre-Registration — Free-Data Incremental-Value Study (1-day RV forecast)

**Committed before any Phase 24 model is trained or any data reaches a model.** The commit date of
this file is the pre-registration timestamp.

Nothing here authorizes strategy, economic backtest, risk, execution, or live trading. Phase 24
produces a **forecast-skill verdict only**: does adding a *free* data block to the **confirmed
Phase-23 h = 1 volatility forecast** make it **more accurate** — significantly, regime-robustly, and
across walk-forward folds — per symbol? The system stays hard-locked to paper. Reads only local
lakes — no Databento, no IBKR, no network, no spend (`OPTIONS_DATABENTO_SPEND_OK` stays unset).

## Why this study, and the honest prior

Phase 23 **confirmed** a 1-day realized-volatility forecast skill on both symbols (the fixed LightGBM
beats HAR-RV, random-walk, EWMA and GARCH(1,1), regime- and fold-robust) using only the *existing*
leak-safe features — the Phase-22 free-data blocks were deliberately held OFF ("confirm first"). This
phase asks the deferred question: **do the free Phase-22 blocks add incremental accuracy on top of
that confirmed model?** VIX is itself a forward-looking volatility measure, so a *positive* result is
genuinely plausible for the market-data block; an honest null (the confirmed baseline is already
hard to improve with these features) is equally acceptable and is the program's modal outcome.

## The single question Phase 24 answers

For each free-data block **B** ∈ {`x1` market-data, `s3` GKG news-tone}, per symbol: does the fixed
Phase-23 LightGBM, trained on the **Phase-23 baseline features + B**, forecast 1-day-ahead realized
volatility with **lower OOS QLIKE than the Phase-23 baseline (features without B)** — by a
significant one-sided Diebold-Mariano test, robust across volatility regimes and across walk-forward
folds?

## What is IDENTICAL to Phase 23 (so this is a clean incremental test)

The entire Phase-23 frozen core is **reused verbatim** (loaded from `config/phase23_vol_h1.yaml`):
the 5-minute sub-sampled RV target, the single fixed no-search LightGBM (seed 7), the gated horizon
**h = 1**, the anchored expanding walk-forward (OOS 2022-01-01 → end, 18 folds), the regime split,
the symbol set `{MES, MNQ}` (per symbol, never pooled), and the QLIKE + HAC/HLN Diebold-Mariano
machinery. **The ONLY difference between the two arms of each test is the presence of block B.** The
baseline arm is *exactly* the confirmed Phase-23 model (its feature set, byte-for-byte). Because
adding a treatment block does not change the row gate (treatment-feature nulls are kept,
LightGBM-native), the baseline and augmented arms are evaluated on the **same OOS rows**.

## The two arms of each test (FIXED a priori)

- **Baseline B0** — the confirmed Phase-23 feature set (HAR + price/vol/volume/time/cross-asset +
  macro + `s2` where covered). `with_marketdata=False`, `with_gkg=False`.
- **Augmented A_B** — B0 **plus** block B (`with_marketdata=True` for `x1`, or `with_gkg=True` for
  `s3`). Everything else identical.

Each block is tested as its **own independent additive arm** (x1 vs B0, s3 vs B0). No combined arm
is gated in this phase (a combined x1+s3 arm may be reported as a non-gated diagnostic once both
blocks individually clear coverage).

## Coverage precondition (FIXED a priori) — the Phase-18 discipline

An arm is **run and gated only if its block covers ≥ 80 % of the OOS decision days** (the fraction of
OOS rows with at least one non-null block feature). This mirrors the Phase-18 coverage gates: a block
that is mostly null over the test window cannot be honestly tested for edge (it would measure the
null-imputation, not the block). An arm below the threshold is **reported as coverage-insufficient
and DEFERRED**, never run as a null.

- **`x1` (FRED VIX/cross-asset):** the lake spans 2018→present → full OOS coverage → **runs now**.
- **`s3` (GKG news tone):** the bulk backfill currently reaches ~2022-08 (chronological fill from
  2019) → ≈14 % of the 2022→2026 OOS window → **deferred** until the backfill lifts coverage ≥ 80 %,
  then re-run unchanged. (Coverage is re-measured at run time, not assumed.)

## Loss, comparison, validation (FIXED a priori — inherited from Phase 23)

- **Loss:** QLIKE on the variance scale (lower better).
- **Comparison:** one-sided Diebold-Mariano on the per-day QLIKE differential **B0 − A_B** (positive
  ⇒ augmented is more accurate), at the 5 % level, HAC lag = h − 1 = 0 (no overlap at h = 1) + the
  HLN small-sample correction, per symbol.
- **Validation:** the Phase-23 anchored expanding walk-forward (OOS 2022→2026, 18 folds); both arms
  share fold boundaries and OOS rows.
- **Regime split:** the Phase-23 causal trailing-22-day-vs-expanding-median rule.

## Verdict gates — per arm, per symbol, ALL must clear at h = 1 (FIXED a priori)

| Gate | Threshold |
|------|-----------|
| **G1 — incremental accuracy** | Augmented OOS QLIKE strictly below the baseline's, one-sided DM (B0 vs A_B) significant at 5 %. |
| **G2 — regime robustness** | Augmented QLIKE ≤ baseline in **both** the calm and turbulent OOS sub-samples (sign-consistent). |
| **G3 — temporal stability** | Augmented QLIKE below the baseline's in **≥ 13 of the 18** walk-forward folds (the one-sided 5 % binomial sign-test threshold). |

Calibration, the h = 5 / h = 22 incremental diagnostics, per-feature SHAP and the block's SHAP share
are **reported, not gated**.

## Decision rule (FIXED a priori)

- **Per block, per symbol: the block ADDS VALUE iff G1 ∧ G2 ∧ G3 clear at h = 1.**
- **Both symbols add value for a block** → that block is an **accepted incremental improvement** →
  authorizes folding it into the volatility model (a `volatility_version` bump, e.g. `vf3`) and into
  the eventual economic-value study. **Never authorizes live trading.**
- **One symbol** → flagged **fragile**; not folded in.
- **Neither / coverage-insufficient** → the block does not add value (or is deferred); the confirmed
  Phase-23 baseline stands unchanged. Recorded honestly in `docs/RESEARCH_VERDICTS.md`.

## Anti-snooping commitments (FIXED)

- The baseline (the exact Phase-23 feature set), the block definitions, the additive-arm design, the
  80 % coverage threshold, the three gates and their thresholds (including 13/18), the QLIKE loss,
  the DM test and its 5 % level, and the reused Phase-23 core are all fixed **before** any Phase-24
  modeling and are **not** changed after seeing results.
- **No hyperparameter search, no feature-subset fishing within a block, no threshold-tuning** to
  manufacture a pass. Each arm runs once per symbol.
- Disclosed-not-silent fallbacks only (the inherited QLIKE→`l2_log_rv` objective fallback).
- A deferred arm is re-run **unchanged** once its coverage clears; the contract is not re-opened.

## Versioning & artifacts

- `freedata_version: "fd1"`. Frozen knobs: `config/phase24_freedata.yaml` (this commit); the core is
  inherited from `config/phase23_vol_h1.yaml` (`vf2`).
- Per-symbol summaries → `data/volatility/runs_fd/<symbol>.json`; combined verdict →
  `data/volatility/runs_fd/verdict.json` (gitignored). MLflow experiment `volatility-freedata-fd1`.

## What this document is NOT

Not an implementation and not a model run. It commits zero data and zero models. The Phase 24
implementation references this frozen contract and must not deviate from it.
