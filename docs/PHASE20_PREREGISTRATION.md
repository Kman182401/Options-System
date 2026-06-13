# Phase 20 Pre-Registration — Meta-Labeling Edge Verdict

**Committed before any Phase 20 model is trained or any data reaches a model.** The commit date of
this file is the pre-registration timestamp. It fixes the Phase 20 protocol, gates, and decision rule
in advance, so the repository itself proves the goalposts predate the data.

Nothing here authorizes strategy, economic backtest, risk, execution, or live trading. Phase 20
produces a **signal edge verdict only**, like Phases 14 and 19. The system stays hard-locked to paper.

## Why meta-labeling, and why now

Five levers — price (5), macro (6), TA (10), microstructure/order-flow (14), and sentiment (19) — have
returned honest nulls through the same fixed framework. All five attacked the problem the same way:
add features to a model that picks a direction. The recurring failure has a recurring shape: the
~78–80% timeout regime makes the directional calls rare and low-precision (Phase 14: high action rate,
no skill; Phase 19 ES treatment: a 5/6 near-miss that died on PBO — a selection overfit). Meta-labeling
(López de Prado, *Advances in Financial ML*, Ch. 3) attacks a **different axis**: keep a simple primary
that picks the side, and train a **secondary (“meta”) model that decides whether to act** on that side —
a precision filter purpose-built for exactly this imbalanced, low-precision regime. It changes the
question from “which way?” to “is this particular call worth taking?”, and it is the rule-respecting way
to give the Phase 19 sentiment signal a second look: *does sentiment help a gate model know when to
trust the order-flow call?* — a genuinely new experiment, not a re-run of the forbidden Phase 19 test.

## The single question Phase 20 answers

Given a fixed, simple, causal **primary side rule**, does a **meta-model** that decides *whether to act*
turn the primary’s low-precision directional calls into a real, deflated edge — and does the `s2`
sentiment block contribute to that gate?

## The primary side rule (FIXED a priori, deterministic, causal — not fitted)

For every micro-label event, the **primary side** is `sign(ofi_top)` evaluated as-of `t0` (the `m1`
top-of-book order-flow-imbalance feature): `+1` (long) when `ofi_top > 0`, `-1` (short) when
`ofi_top < 0`. This is the canonical “follow the order flow” rule (positive imbalance ⇒ buying
pressure). It is a **deterministic causal function of past data** — not a fitted model — so it requires
no cross-validation and introduces **no primary-model leakage and no nested out-of-fold prediction**.
The meta-model is therefore the *only* fitted component. Events with `ofi_top == 0` (rare, no imbalance)
have no primary side and are excluded from the meta-set. If `ofi_top` is not the exact column name in
the `m1` feature set, the implementation must STOP and report rather than substitute a different
feature.

## The meta-label (binary, FIXED a priori)

For each event with a primary side `s ∈ {-1, +1}`, using the realized micro-label `label`
(`+1` upper / `-1` lower / `0` timeout-or-close) from `data/micro_labels/`:

- `meta_label = 1` if `label == s` (the primary called the correct barrier side — a true positive),
- `meta_label = 0` otherwise (wrong direction, **or** a `0` timeout — the directional bet did not pay).

This binary target is the primary’s correctness. It never reads `t1` outcome beyond the already-resolved
`label`, and purge/embargo on `t1` apply exactly as in Phase 14.

## The meta-model and its features

A binary LightGBM classifier predicting `P(meta_label = 1 | features)`. Features = the `m1` order-flow
block **plus** the `s2` sentiment block (attached on `t0` via `sentiment.join.attach_to_micro_labels`),
plus the primary’s own `ofi_top` magnitude. **Fold-local balanced class weighting** (computed inside
each training fold from `y_train` only, identical mechanism to `mm1`) handles the binary imbalance.

**Decision threshold is FIXED a priori at τ = 0.5** — act on the primary side when
`P(meta_label = 1) > 0.5`, otherwise stay flat. τ is **never tuned or searched**. If the resulting
action rate is near zero, that is an honest result (it fails the action gate → null), not something to
fix by moving τ.

No new feature engineering beyond the existing joins; sentiment nulls (no prior events) are **kept**
(LightGBM-native NaN), never imputed.

## Arms (the comparison)

- **B0 — always-act reference:** take the primary side `s` on every event (no gate). Gross proxy
  `s · ret_t1` on every event. This is the primary’s unconditional performance.
- **M — meta-gated (PRIMARY VERDICT arm):** take side `s` only when the meta-model says act
  (`P > 0.5`), else flat. Gross proxy `s · ret_t1` when acting, `0` when flat.

SHAP is reported for arm **M** to show whether the `s2` sentiment features are used by the gate. The
implementation MAY add an OFI-only meta reference (meta-model without the sentiment block) purely to
attribute sentiment’s contribution; if added it is a diagnostic, **not** a gate, and its configs count
as trials.

## Row set and window (FROZEN)

- **Full Phase-14 window** `t0 ∈ [2026-01-26, 2026-06-06]`, per symbol (ES, NQ separately) — to maximize
  effective N for the core meta-labeling question (~1,000/symbol, vs the 735/696 of the Phase-19
  supported-region subset).
- The `s2` sentiment features are **null/NaN before 2026-03-10** (the Phase-18 archive cutoff) and
  populated thereafter; nulls are kept, not imputed. The meta-model uses sentiment where it has coverage.
- **Secondary sentiment sub-analysis** on the supported region (`t0 >= 2026-03-10`) reports sentiment’s
  SHAP contribution where coverage is complete. This sub-analysis is diagnostic, not a gate.

## Inherited fixed parameters (FROZEN — read from `config/micro_model.yaml`, `mm1`; do not restate or alter)

LightGBM base params, the early-stopping config, the 8-config search grid
(`max_depth [2,3]` × `min_child_samples [50,150]` × `reg_lambda [5,20]`, selection on
`gross_signal_sharpe` of the **meta-gated** signal), the purged+embargoed K-fold (pooled OOS, scored
once, purge/embargo on `t1`), Combinatorial Purged CV (5 paths), PBO across the configs, PSR/DSR
deflated by the trial count, and `seed 7`. The meta-model is binary (`objective binary`, `num_class`
not applicable) but every other knob is inherited unchanged. **No new hyperparameters, no new search
dimensions, no tuning.** Trials = the meta-model’s config grid (plus any OFI-only-meta reference
configs if that diagnostic is run); the DSR is deflated by the honest total.

## Verdict gates (FROZEN)

**Five gates inherited verbatim** from `config/micro_model.yaml` (`mm1`), applied to arm M’s
**meta-gated** gross signal — the bar does not move:

|Gate                                          |Threshold|
|----------------------------------------------|---------|
|PBO (selection-overfit probability)           |< 0.5    |
|Gross signal DSR (deflated by the trial count)|> 0.5    |
|Mean gross signal return                      |> 0      |
|Action rate (`P > 0.5` fraction)              |>= 0.03  |
|CPCV path-distribution median gross Sharpe    |> 0      |

**One gate transparently adapted** (disclosed, not silently moved): the `mm1` sixth gate was 3-class
macro F1 ≥ 0.20, which is undefined for a binary meta-model. It is replaced by a **meta-skill gate**,
fixed a priori here: arm M is credited only if **both** (a) the meta-model’s out-of-fold balanced
accuracy ≥ 0.52 (modestly above chance), **and** (b) the hit-rate on acted-on events strictly exceeds
the always-act (B0) hit-rate (the gate adds precision over acting on everything). This directly tests
the one thing meta-labeling claims to do — filter false positives — and is the deliberate, disclosed
substitute for the 3-class-only F1 gate.

Gross return = `primary_side · ret_t1`, **no commissions, no slippage** — a gross signal-return proxy,
not an executable backtest.

## Leakage and CV (FROZEN)

Because the primary side is a deterministic causal function of as-of-`t0` data (not a fitted model),
there is no primary-model leakage and **no nested out-of-fold construction is needed**. The meta-model
is the sole fitted component and runs entirely inside the existing purged+embargoed K-fold / CPCV /
PBO / DSR machinery (purge and embargo on each label’s `t1`). No random split anywhere.

## Attribution and decision rule (FROZEN)

- **Per symbol, primary verdict = does arm M clear all five inherited gates AND the meta-skill gate?**
- B0 is the reference (the primary without the gate); it is expected to fail (consistent with the
  Phase-14 OFI null). Meta-labeling’s job is to filter B0 into something that passes.
- **M clears every gate on a symbol** → “meta-labeling edge candidate” for that symbol → authorizes
  **only** a future Phase 21 economic backtest with realistic costs/slippage for that symbol. **Never
  live trading.** A single-symbol candidate while the other symbol fails is flagged **fragile**.
- **M fails on both symbols** → **meta-labeling is the next honest null.** Because meta-labeling is the
  canonical remedy for exactly this low-precision/imbalanced regime, its failure is strong evidence that
  the binding constraint is **sample size / edge existence at this horizon**, not model framing — which
  escalates the strategic fork (acquire more data/depth, redesign the horizon/market, or accept the
  result). Recorded in `docs/RESEARCH_VERDICTS.md`. The meta-labeling lever is then **not re-litigated**
  (no τ tuning, no primary-rule swapping to fish for a pass).

## Anti-snooping commitments (FROZEN)

- The primary side rule (`sign(ofi_top)` at `t0`), the meta-label definition, τ = 0.5, the feature set,
  the gates, and all inherited parameters are fixed before any modeling and are not changed after seeing
  results.
- Each arm is run once per symbol; all meta-model configs (and any OFI-only-meta diagnostic configs) are
  honestly counted as trials for the DSR.
- No re-running with different seeds, thresholds, or primary rules to fish for a pass.
- If the pipeline must change for a legitimate engineering reason, the model version is bumped and the
  change is documented in `docs/DECISIONS.md` — never silently absorbed.

## What this document is NOT

Not an implementation and not a model run. It commits zero data and zero models. The Phase 20
implementation prompt will reference this frozen contract and must not deviate from it.
