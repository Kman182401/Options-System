# Phase 19 Pre-Registration — Sentiment Micro-Model A/B Edge Verdict

**Committed before any Phase 19 model is trained or any data reaches a model.** The commit
date of this file is the pre-registration timestamp. This document fixes the Phase 19 protocol,
gates, and decision rule in advance so the repository itself proves the goalposts predate the
data — the corrective to the Phase 18 provenance gap (Phase 18’s gates were pre-registered in an
upstream prompt but were not transcribed into a committed doc until the verdict-date commit).

Nothing here authorizes strategy, backtest, risk, execution, or live trading. Phase 19 produces a
**signal edge verdict only**, exactly like Phase 14. The system remains hard-locked to paper.

## The single question Phase 19 answers

Given the Phase 14 microstructure micro-model (`mm1`: 17 `m1` order-flow features, `ml1` ~30-min
triple-barrier labels, 3-class target), **does adding the `s2` sentiment feature block produce a
real, deflated edge — over and above the order-flow-only baseline — on the rows that actually have
sentiment coverage?** Coverage was established in Phase 18; coverage is not edge.

## Two arms (the only delta is the sentiment block)

| Arm | Feature set | Model spec |
|-----|-------------|------------|
| **Baseline (B)** | The unchanged `mm1` features (17 `m1` OFI features) | `mm1` spec, on the Phase 19 row set |
| **Treatment (T)** | `mm1` features **+** the full `s2` sentiment block (the `sent_*` columns produced by `sentiment.join.attach_to_micro_labels` on `t0`) | `mm2` spec (= `mm1` + `s2`), on the same rows |

Every other element — bars, labels, target, folds, class weighting, search grid, selection metric,
seed, trial count, and the verdict gates — is **identical** between the two arms and identical to
Phase 14. The sentiment feature block is the sole experimental variable.

Run artifacts for each arm stamp the model version (`mm1` / `mm2`), the row region, and the run date,
so saved runs stay self-describing.

## Row set (frozen)

- **Supported region only:** micro labels (`data/micro_labels/`) with `t0 >= 2026-03-10` (the Phase 18
  GDELT archive cutoff), so every Treatment row has a real, point-in-time-knowable sentiment context.
- **Per symbol, never pooled:** ES and NQ are evaluated separately (the micro-model is per-symbol, as
  in Phase 14). The Phase 18 pooled G3 count does not apply here.
- **Both arms run on the identical row set** for each symbol.
- **Expected sample (from the Phase 18 coverage report):** supported-region labels ES 1,132 / NQ 1,036;
  of these, sentiment-covered ES 1,111 / NQ 1,020. This is the **same ~1,000-effective-labels-per-symbol
  regime as Phase 14**. Statistical power to detect a small edge after trial deflation is limited by
  design; **an honest null is the expected and fully acceptable outcome.**

## Inherited fixed parameters (frozen — copied from `config/micro_model.yaml`, `mm1`)

- **Target:** `multiclass_3`, classes `[-1, 0, 1]`; predicted class is the signal (`-1` short / `0`
  flat / `+1` long). Timeout (`0`, ~78–80%) is modelled, never dropped or sign-collapsed.
- **Class weighting:** `balanced_fold_local`, `use_sample_weight_in_balance: true` — computed inside
  each training fold from `y_train` only, multiplied into the persisted uniqueness/sample weights;
  the early-stopping inner-val block uses the same fold-local mapping.
- **LightGBM base:** `objective multiclass`, `num_class 3`, `n_estimators 600` (upper bound, early-
  stopped), `learning_rate 0.02`, `max_depth 3`, `num_leaves 7`, `min_child_samples 100`,
  `subsample 0.7`, `subsample_freq 1`, `colsample_bytree 0.7`, `reg_alpha 1.0`, `reg_lambda 10.0`,
  `max_bin 127`, `n_jobs 1`, `seed 7`.
- **Early stopping:** `rounds 50`, `inner_val_fraction 0.2`, on a **purged** chronological tail of the
  training fold (training rows whose `[t0, t1]` overlap the tail are purged).
- **Search:** the same 8-config grid — `max_depth [2, 3]` × `min_child_samples [50, 150]` ×
  `reg_lambda [5.0, 20.0]` = 8 trials; `selection_metric = gross_signal_sharpe`, selected on pooled
  OOS gross signal Sharpe inside the purged K-fold (no test-fold peeking).
- **Validation:** purged + embargoed K-fold (pooled OOS, each sample scored once) + Combinatorial
  Purged CV (5 paths); purge/embargo use each label’s `t1`. **No random split anywhere.**
- **Deflation:** `n_trials = 8` per arm; the Deflated Sharpe Ratio is deflated by those 8 within-arm
  trials (identical to Phase 14).
- **Determinism:** `seed 7`.

**No new hyperparameters, no new search dimensions, no tuning of any of the above.**

## Sentiment feature handling (frozen)

- The full `s2` block is added **exactly as joined** — no new feature engineering, no new feature-
  selection mechanism. Adding such a mechanism would itself be a new pre-registered knob and would
  break “the only delta is the sentiment block.”
- **Known dead features, included deliberately as-is:** in the supported region the `1d` and `240m`
  `has_any` presence flags are ~98% constant. A gradient-boosted tree cannot make a useful split on a
  (near-)constant column, so these are harmless to include and are **not** filtered. Any realized
  signal is expected to come from the **score aggregates** (`mean`/`sum`/`max_abs`/per-class mean
  sentiment) and the **sparser `15m` window** (~86% coverage, more variance), not the presence flags.
- SHAP is reported for the Treatment arm to show which sentiment features (if any) it actually uses.

## Verdict gates (UNCHANGED — copied verbatim from `mm1`; the bar never moves, only the inputs do)

An **edge candidate** (per symbol, per arm) must clear **ALL** of:

| Gate | Threshold |
|------|-----------|
| PBO (selection-overfit probability) | < 0.5 |
| Gross signal DSR (deflated by 8 trials) | > 0.5 |
| Mean gross signal return | > 0 |
| Action rate (`pred != 0`) | >= 0.03 |
| 3-class macro F1 | >= 0.20 |
| CPCV path-distribution median gross Sharpe | > 0 |

Gross return = `pred_class * realized return to the barrier`, **no commissions, no slippage** — a
gross signal-return proxy, not an executable backtest.

## A/B attribution logic (frozen)

- **Primary verdict, per symbol = does the Treatment arm clear all six gates?**
- The Baseline arm is run identically as a **reference for attribution**. Phase 14 OFI-only already
  failed on both symbols (ES failed 4/6; NQ passed 5/6 but failed the deciding CPCV gate). Therefore:
  - **T passes, B fails** → the edge is attributable to the sentiment block.
  - **T passes, B passes** → the pass is **not cleanly attributable to sentiment** (the restricted row
    set alone moved the baseline); flag for scrutiny rather than crediting sentiment.
  - **T fails** → null for that symbol, regardless of B.
- Verdicts are **per symbol, never pooled.**

## Decision rule (frozen)

- **T clears all six gates on a symbol** → “sentiment edge candidate” for that symbol → authorizes
  **only** a future Phase 20 economic backtest with realistic costs/slippage for that symbol. **Never
  live trading.** A single-symbol candidate while the other symbol fails is flagged **fragile** (the
  Phase 14 NQ near-miss is the cautionary precedent).
- **T fails on both symbols** → **sentiment is the fifth honest null.** Recorded in
  `docs/RESEARCH_VERDICTS.md`. The remaining forks become the next operator decision: (a) MBP-10
  multi-level order-flow depth — **paid, billing-gated, blocked** until billing is deliberately
  unfrozen; or (b) a deliberate horizon/regime redesign. **No further sentiment re-tuning** — the lever
  is not re-litigated.

## Anti-snooping commitments (frozen)

- No post-hoc change to any gate threshold or parameter above.
- No adding, removing, or re-engineering features after seeing results.
- No re-running with different seeds to fish for a pass; each arm is run once per symbol under these
  exact settings; all 8 search configs per arm are honestly counted as trials.
- If the pipeline must change for a legitimate engineering reason, the model version is bumped and the
  change is documented in `docs/DECISIONS.md` — never silently absorbed.

## What this document is NOT

Not an implementation and not a model run. It commits zero data and zero models. The Phase 19
implementation prompt will reference this frozen contract and must not deviate from it.
