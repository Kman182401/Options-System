# Validation framework — the truth-detector

This is the machinery every model is judged by **before** it is allowed near
money. Its entire job is to stop the system fooling itself: to make leakage,
overlap and selection bias impossible to hide. If a number here looks good, it
has to be good for an honest reason.

Read this top-to-bottom and you can explain, in plain English, why a backtest
number is trustworthy or not — that is the prime directive.

---

## The problem it solves

Two failure modes turn an impressive backtest into a lie:

1. **Leakage from overlapping labels.** Our labels (Phase 3, triple-barrier) span
   up to three days, so neighbouring labels share outcome windows. Plain
   K-fold puts overlapping samples on both sides of the train/test line, and the
   model "learns" the test answer. The reported skill is fake.
2. **Selection bias.** Try enough models/parameters and one will look great on any
   finite sample by luck. A single Sharpe ratio cannot tell skill from a lucky
   draw out of many tries.

The framework neutralises both: **purging + embargo** (using each label's
resolution time `t1`) for the first, **PBO / Deflated Sharpe** for the second.

---

## Building blocks

### Purge + embargo (`_purge.py`)
Each sample carries a label interval `[t0, t1]`. For a given test fold:

* **Purge** — drop any training sample whose `[t0, t1]` overlaps the test
  interval. Two intervals overlap iff `t0 ≤ test_end and t1 ≥ test_start`. This
  removes train labels that were still resolving while the test window was live.
* **Embargo** — drop a further `embargo_pct` of bars *immediately after* each test
  block, to kill residual serial-correlation leakage. Embargo is forward-only;
  the purge already handles the overlap/before side.

This single primitive is the one place leakage logic lives; every splitter calls
it, and it is checked against a hand-computed reference in
`tests/test_validation_purge.py`.

### Purged K-fold (`purged_kfold.py`)
Contiguous test folds over the `t0`-sorted samples, each purged + embargoed.
sklearn-compatible (`split`, `get_n_splits`). Every sample is tested out-of-sample
exactly once. `split_report()` returns per-fold train/test/purged/embargoed counts.

### Combinatorial Purged CV — CPCV (`cpcv.py`)
The honest workhorse. Split into `N` contiguous groups, test on **every**
combination of `k` of them. That gives `C(N, k)` purged splits and, by recombining
the held-out groups, `C(N-1, k-1)` distinct out-of-sample **paths** — a *whole
distribution* of OOS performance instead of one fragile number. With our default
`N=6, k=2` that is 15 splits and 5 paths.

Why it matters here: average label uniqueness is ≈ 0.23, so any single
walk-forward path is high-variance and easy to fool. The spread across CPCV paths
is what tells you whether an edge is real or a coin that landed your way.

### Walk-forward (`walk_forward.py`)
Anchored (train grows from the start) or rolling (fixed lookback) sequential
train→test, purged at each boundary. The chronological path you would actually
have lived through — the realistic complement to CPCV.

---

## The overfitting statistics (`stats.py`) — the verdict

These are the *gate*, not the accuracy score. A high CV accuracy with PBO near 1
or DSR ≤ 0 means we found noise.

| Statistic | Question it answers | Read it as |
|---|---|---|
| **PSR** (Probabilistic Sharpe Ratio) | Is the Sharpe real given sample length, skew and fat tails? | Probability the true Sharpe > benchmark. Higher is better. |
| **DSR** (Deflated Sharpe Ratio) | Does the Sharpe survive after accounting for how many configs were tried? | PSR against a noise-derived threshold. **DSR ≤ 0.5 → unconvincing; ≤ 0 territory → noise.** |
| **PBO** (Probability of Backtest Overfitting) | Does the in-sample-best config underperform out-of-sample? | Fraction of CSCV splits where the IS winner is below the OOS median. **PBO → 1 = overfit selection; → 0 = robust.** |
| **minTRL** | How long must the track record be to trust the Sharpe? | Number of observations needed for PSR to reach the confidence; compare to your actual `n`. |

Conventions baked in (so nobody re-derives them wrong): all of `SR`, benchmark,
skew and kurtosis are **per-bar** (never mix an annualised Sharpe with per-bar
moments); kurtosis is **raw** (Normal = 3); Φ/Φ⁻¹ come from `statistics.NormalDist`.
Every formula is checked against a closed-form reference in
`tests/test_validation_stats.py` (e.g. PSR at the benchmark is exactly 0.5; the
Normal denominator is `√(1 + ½·SR²)`).

---

## Effective sample size — read it everywhere

~11k labels do **not** carry 11k independent observations. With average
uniqueness ≈ 0.23 they carry ≈ 2.7k, and per-fold it is smaller still. The harness
reports **effective N = Σ average-uniqueness** for every fold and path. This number
should temper how much you trust any single result and how complex the later model
is allowed to be. The raw row count overstates the information you actually have.

---

## The evaluation harness (`evaluate.py`)

Given one or more sklearn-compatible estimators and the leak-free
`(features@t0, y, t1, weight)` matrix (assembled via the labeling layer's
already-leak-tested `labels_with_features`), the harness:

1. runs **purged K-fold** → every sample scored OOS once → pooled metrics per
   estimator + a `(periods × estimators)` matrix for **PBO**;
2. runs **CPCV** → a distribution of OOS-path Sharpes per estimator;
3. honours the **sample weights** in every fit and weighted metric, and reports
   **effective N** per fold/path;
4. computes **PSR** and **DSR** (deflating for the number of estimators tried).

It is model-agnostic — it judges whatever estimator you hand it. The shipped
baselines (a most-frequent dummy and a standardised logistic regression) are
deliberately unskilled. **On real data they must look like chance.** If a dummy
ever looks profitable through this harness, something leaks — stop and investigate
before trusting anything downstream.

### The teeth test (the most important test)
`tests/test_validation_teeth.py` proves the framework actually catches leakage,
two ways:

* **Mechanism** — a training sample whose window overlaps the test fold is purged;
  a sample that resolves before the test is retained. The leak path fails the
  check, the safe path passes (direct analogue of the feature-leakage teeth test).
* **Skill collapse** — labels are the *forward returns of a random walk* (zero
  global predictability), with fold size ≤ label horizon so overlap reaches every
  test point. A KNN shows inflated OOS skill (~0.78) **without** purging — purely
  from overlapping train labels — and that skill collapses to chance (~0.50)
  **with** purging + embargo. A validation framework that can't demonstrate this
  is theater.

---

## How to run

```bash
# Evaluate the baselines on each symbol and save a run (data/validation/<symbol>.json)
uv run python -m options_system.validation.evaluate --symbols MES MNQ

# Open the read-only validation-health view over the saved runs
uv run streamlit run src/options_system/observability/validation_health.py
```

The configuration (`config/validation.yaml`, `validation_version`) fixes the fold
counts, embargo, CPCV groups, metric set and seed; the same seed always yields the
same result.

---

## What this is **not**

It is not a model, not an economic/`nautilus` backtest, not a strategy, and not a
hyperparameter search. It is the infrastructure all of those will be judged
through. When the real signal model arrives (Phase 5), it plugs into this harness
and is evaluated **only** here — and any hyperparameter search must happen *inside*
the CV and be counted as trials for the Deflated Sharpe Ratio. See
`docs/DECISIONS.md` (Phase 4) for the rationale behind each choice.
