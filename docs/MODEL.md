# Signal model & honest edge verdict (`model_version = v1`)

This is the answer to one question, asked honestly: **does a model predict the
triple-barrier label better than chance — after deflation, and over and above
market beta?** Nothing here trades, backtests economically, or promotes a model.
It produces a single, auditable **VERDICT** per symbol, and the current verdict is
the input that decides what we build next.

> **Bottom line (2026-06-08):** **No significant edge** on either MES or MNQ. The
> price-only directional model does not beat a coin flip after deflation, and it
> *loses to buy-and-hold*. This is a valid, expected result — not a failure — and
> it says: iterate on **data and features** (the deferred macro/sentiment layer,
> richer microstructure) before building any strategy. **(Update — Phase 6: the
> macro/economic-event layer was added and re-tested through this identical
> verdict; still no significant edge. See "Phase 6 re-verdict" below and
> `docs/MACRO.md`.)**

---

## What the model is

* **Target — direction.** The Phase-3 triple-barrier label is `{-1, 0, +1}`. The
  tiny vertical-timeout class (`0`, ~3-4%) is folded into a **directional** target
  `y_dir ∈ {-1, +1}` by the sign of its realised return at `t1` (`timeout_handling
  = sign_return`; no rows discarded). The model predicts up vs down.
* **Estimator — a deliberately *under-powered* LightGBM.** Effective N is only
  ~2.5-2.7k against 45 features, so the model is constrained so it *cannot* overfit:
  shallow trees (`max_depth ≤ 3`, `num_leaves 7`), large leaves
  (`min_child_samples ≥ 100`), L1+L2 (`reg_alpha 1`, `reg_lambda 5-20`), sub-1
  bagging/feature fractions (`0.7`/`0.6`), and early stopping **inside** each CV
  fold against a *purged* tail. Sample weights (Phase-3 uniqueness) are honoured in
  every fit and every metric. Fixed seeds + single-threaded → byte-reproducible.
* **Tuning — small, and paid for.** A 2×2×2 grid over `max_depth`,
  `min_child_samples`, `reg_lambda` (8 configs) runs **inside** the purged K-fold;
  no config ever sees its own test fold. `n_trials = 8` is fed to the Deflated
  Sharpe Ratio, so every extra config raises the bar a result must clear.

## How it is judged (only through the Phase-4 framework)

Every number comes from `validation/`: purged K-fold for the pooled out-of-sample
pass, **CPCV** for the path distribution, and **PBO / PSR / DSR** for the
overfitting verdict. Two things this phase adds on top of the baseline harness:

* **Skill is reported over and above beta.** The return metric is the *excess* of
  the model's position-weighted return over a **perma-long benchmark**
  (`excess = (position − 1) · ret`, information-ratio style). A model that is
  effectively always-long has zero excess and shows no edge, however much beta it
  rode. The Deflated Sharpe is computed on this **excess** series.
* **Selection bias is priced in.** PBO compares the 8 configs across in-/out-of-
  sample halves; the DSR deflates by `n_trials = 8`.

**An edge requires ALL of:** directional accuracy `> 0.52`, PBO `< 0.5`, excess
**DSR** `> 0.5`, and positive mean excess-over-long. Miss any one → *no
significant edge*. (Thresholds in `config/models.yaml`.)

---

## The verdict (full history 2019→2026, `feature_version = v1`, `label_version = v1`)

| symbol | n | eff. N | dir. acc (vs 0.50) | strategy SR | long-only SR | excess-over-long SR | excess **DSR** (deflated, 8 trials) | **PBO** | VERDICT |
|---|---|---|---|---|---|---|---|---|---|
| **MES** | 10,827 | 2,548 | 0.5164 | 0.0162 | 0.0299 | **−0.0175** | **0.006** | **0.83** | **no significant edge** |
| **MNQ** | 11,688 | 2,732 | 0.5267 | 0.0176 | 0.0315 | **−0.0127** | **0.021** | **0.70** | **no significant edge** |

Read it plainly:

* **It loses to buy-and-hold.** On both symbols the strategy Sharpe (0.016 / 0.018)
  is *below* the perma-long benchmark (0.030 / 0.031). The model's only "profit" is
  beta — and it subtracts from it. Excess-over-long Sharpe is **negative**.
* **No skill after deflation.** Excess DSR is ≈0 (0.006 / 0.021) — far under the
  0.5 bar. The CPCV path distribution agrees: the out-of-sample excess Sharpe is
  **negative on every one of the 5 paths** (MES mean −0.025, range [−0.033, −0.014];
  MNQ mean −0.019, range [−0.029, −0.010]).
* **Selection is overfit.** PBO 0.83 / 0.70 (>0.5) — the in-sample-best config
  tends to underperform out-of-sample, exactly the selection-bias signature the
  framework exists to catch.
* **MNQ's directional accuracy (0.527) nudges past chance**, but that alone is not
  an edge: it still loses to beta and the selection is overfit, so the verdict
  holds. Raw accuracy is the *weakest* of the four gates by design.

The baselines remain the sanity anchor: a perma-long dummy looks "profitable" only
through beta (high return-PSR), which is precisely why the skill metric is netted
of beta here.

---

## What SHAP says

Global importances (mean |SHAP|, refit on full history for interpretation only —
**not** a performance number) are **spread thin across many features**, with **no
single dominant driver** (top-feature share ≈ 7% on both symbols; dominance flag
off). That is consistent with — and corroborates — the null verdict: nothing in the
feature set carries a strong, stable directional signal.

* **MES top drivers:** `obv_norm_60`, `xa_ret_spread`, `xa_ratio_z_390`, `rv_390`,
  `ema_dist_z_12`, `ret_1` — signed-volume flow, the MNQ–MES return spread and price
  ratio, session realised vol, and short-horizon momentum.
* **MNQ top drivers:** `rv_390`, `vwap_dist`, `ema_dist_z_12`, `ret_15`, `rv_15`,
  `rvol_30` — realised vol, distance from session VWAP, momentum and relative volume.

These are **economically plausible** (volatility, momentum, flow, cross-asset) and
none smells like leakage — the right outcome for a leak-tested feature set. The
problem is not a suspicious feature; it is that the *plausible* features simply
don't predict next-move direction strongly enough to beat beta after costs of
selection.

---

## Why this is the right (and expected) outcome

Price-only intraday direction is genuinely hard, and a marginal-or-null result was
the prior, not a surprise. The unacceptable outcome would have been a *fake* edge —
so the rigor was pointed at *not* manufacturing one: in-CV tuning counted as trials,
skill netted of beta, deflation by trial count, PBO on the selection, and a CPCV
distribution instead of one lucky number. The model is **explainable** (SHAP shows
why it predicts what it predicts) and **honest** (every number is framework-sourced).

## Implication for next steps

Because there is **no edge**, we do **not** build the nautilus economic backtest or
a strategy on top of this signal yet. The productive direction is to **improve the
inputs**, then re-run this exact verdict:

1. **Add information the price series cannot contain** — the deferred macro /
   news / sentiment layer (`features/` has a disabled `news` seat; `sentiment/` is
   scaffolded). Regime and event context is the most likely source of real,
   beta-independent skill.
2. **Revisit features/labels** — microstructure (order-flow imbalance, depth),
   longer horizons, or meta-labeling on a primary signal (the schema already
   carries the `side`/`meta_label` seat).
3. **Re-evaluate through this same harness.** The bar does not move; the data does.

When (and only when) a configuration clears all four gates here will a strategy be
worth building.

---

## Phase 6 re-verdict — adding structured macro context

Per the plan, Phase 6 adds a **macro / economic-event** feature layer (FRED/ALFRED
first-print releases + the FOMC calendar) and re-runs **this exact verdict** — same
labels, same CV, same gates — changing *inputs only*. Full method + leakage rules:
`docs/MACRO.md`.

| symbol | inputs | dir. acc | excess SR | excess **DSR** | **PBO** | VERDICT |
|---|---|---|---|---|---|---|
| MES | price-only (45) | 0.5164 | −0.0175 | 0.006 | 0.83 | no significant edge |
| MES | **price+macro (72)** | 0.4976 | −0.0371 | 0.00001 | 0.88 | **no significant edge** |
| MNQ | price-only (45) | 0.5267 | −0.0127 | 0.021 | 0.70 | no significant edge |
| MNQ | **price+macro (72)** | 0.5209 | −0.0142 | 0.014 | 0.52 | **no significant edge** |

The verdict is **unchanged** on both symbols. Macro did not help — on MES it
slightly *hurt* (pooled OOS accuracy dipped below chance, excess Sharpe more
negative); on MNQ it barely moved the numbers (PBO fell to a borderline 0.52) but
excess-over-beta stays negative and the deflated Sharpe ≈0. **Notably, macro
features dominate the model's SHAP importance (~66–69 % of total) yet carry no
out-of-sample edge** — the validation framework's job is precisely to expose that
gap, and the high-importance-but-null result argues *against* leakage (a real leak
would inflate OOS accuracy, not depress it). The productive levers remain news /
sentiment, microstructure (likely with a shorter-horizon label), or options
structure — each tested through this same harness.

## Phase 10 — opt-in TA v2 controlled experiment

Phase 10 wires the isolated **v2 technical-analysis layer** (`docs/TA_FEATURES.md`,
`data/ta_features/`) into this exact verdict as an **opt-in, non-default** experiment.
It changes *inputs only* — same labels, same CV, same LightGBM config, same PBO / DSR /
CPCV / excess-over-beta gates, same thresholds.

**TA is additive and opt-in.** The default model is unchanged: `load_training_matrix`
defaults to `with_ta=False`, the default `models.run` invocation is still the canonical
price+macro baseline, and the canonical `<symbol>.json` verdict file is never overwritten
by the TA experiment (the candidate is saved as `<symbol>_macro_ta.json`, the comparison as
`<symbol>_ta_comparison.json`). TA columns are appended *after* price and macro, NaN-during-
warmup is tolerated (no rows dropped), and any degenerate `±inf` is sanitised to null before
training.

**TA is not new information.** Every TA feature is a deterministic *transformation of the
same price stream* the v1/price layer already uses (it is explicitly de-duplicated against
v1's RSI/MACD/ADX/Bollinger/OBV/z-scores). The honest prior is therefore *low* added alpha:
a price-derived transform cannot inject information the price series does not already contain.
The point of the experiment is to **measure**, not to hope.

**The honest test:** price+macro (baseline) vs price+macro+TA (candidate) through the four
identical gates — directional accuracy `> 0.52`, PBO `< 0.5`, excess **DSR** `> 0.5`, and
positive mean excess-over-long. TA "wins" only if the candidate clears **all four**.

| symbol | inputs | dir. acc | excess SR | excess **DSR** | **PBO** | VERDICT |
|---|---|---|---|---|---|---|
| MES | price+macro (72) | 0.4976 | −0.0371 | 0.00001 | 0.88 | no significant edge |
| MES | **price+macro+TA (79)** | 0.4988 | −0.0360 | 0.000 | 0.47 | **no significant edge** |
| MNQ | price+macro (72) | 0.5209 | −0.0142 | 0.014 | 0.52 | no significant edge |
| MNQ | **price+macro+TA (79)** | 0.5218 | −0.0153 | 0.022 | 0.89 | **no significant edge** |

**Result (run 2026-06-09, `ta_feature_version = v2`, 7 TA columns added): TA cleared
no gates on either symbol; the verdict is unchanged — _no significant edge_.** The
numbers barely move and excess-over-beta stays **negative** everywhere (the model
still loses to buy-and-hold both with and without TA):

* **MES — "TA worsened" (by the headline metric).** Directional accuracy +0.0012 and
  excess Sharpe +0.001 (still negative); excess DSR stays ≈0 (1e-5 → 0). PBO *fell*
  sharply (0.88 → 0.47) but a low PBO on a model with negative excess and zero DSR is
  not an edge — the other three gates still fail.
* **MNQ — "TA improved but did not clear gates".** Excess DSR ticked up (0.014 →
  0.022) — still **≪ 0.5** — while PBO *worsened* (0.52 → 0.89) and excess Sharpe got
  slightly more negative. No gate is cleared.
* **The PBO swinging in opposite directions (MES down, MNQ up) is itself a tell:** the
  7 TA columns are near-collinear transforms of the price the model already has, so
  they destabilise config selection without adding information. This is exactly the
  low-alpha, redundant-feature outcome the honest prior predicted.

This is a **valid, expected result**, not a failure. TA is a transformation of the
same price stream; it cannot inject information the price series does not already
contain, and the gates correctly refuse to reward it. The productive levers remain
genuinely new information (news / sentiment, deeper microstructure / order flow),
each tested through this same unchanged harness. Per-symbol detail:
`data/models/runs/<symbol>_ta_comparison.json`.

Run the experiment:

```bash
# Build the TA lake first (local, no Databento), then the side-by-side verdict
uv run python -m options_system.ta.build --symbols MES MNQ
uv run python -m options_system.models.run --symbols MES MNQ --compare-ta
```

## How to run / view it

```bash
# Full pipeline per symbol: matrix → in-CV search → CPCV/PBO/DSR verdict → SHAP → MLflow
uv run python -m options_system.models.run --symbols MES MNQ

# Opt-in TA experiment: price+macro vs price+macro+TA through the identical gates
uv run python -m options_system.models.run --symbols MES MNQ --compare-ta

# Read-only model-health view (verdict, gate metrics, CPCV distribution, SHAP)
uv run streamlit run src/options_system/observability/model_health.py

# Browse the MLflow runs (local file store, no cloud)
uv run mlflow ui --backend-store-uri data/mlruns
```

Runs are saved to `data/models/runs/<symbol>.json` (+ a SHAP PNG) and logged to a
local MLflow file store under `data/mlruns`. Config — the regularisation, the grid,
the verdict thresholds — lives in `config/models.yaml`, versioned by `model_version`.
Rationale for each choice: `docs/DECISIONS.md` (Phase 5).
