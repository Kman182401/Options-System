# PHASE20_META.md — Meta-labeling edge verdict (Phase 20)

## What it is
A pre-registered **meta-labeling** test (López de Prado, *Advances in Financial ML*,
Ch. 3) that answers one frozen question: given a fixed, deterministic **primary** that
picks the side of every bet, does a fitted binary **meta-model** that decides *whether to
act* turn the primary's low-precision directional calls into a real, deflated edge — and
does the `s2` sentiment block help that gate? The full frozen contract is
[`docs/PHASE20_PREREGISTRATION.md`](PHASE20_PREREGISTRATION.md) (committed `257113f`,
before any modeling). This run must not — and does not — deviate from it.

It is a **signal edge verdict only**: every return number is a gross signal-return proxy
(`primary_side · ret_t1`, **no commissions, no slippage**). It is **not a strategy and not
an economic backtest**, nothing trades, and it authorizes **no live trading**. Reads only
the local lakes — no Databento, no IBKR, no network, no spend.

## Why meta-labeling (a different axis from Phases 5/6/10/14/19)
The five prior levers all attacked the problem the same way — add features to a model that
**picks a direction** — and all returned honest nulls in the ~78–80% timeout regime, where
directional calls are rare and low-precision. Meta-labeling changes the question from
"which way?" to "**is this particular call worth taking?**": keep a simple primary for the
side, train a secondary precision filter for the *act/don't-act* decision. It is the
rule-respecting way to give the Phase-19 sentiment signal a second look — *does sentiment
help a gate know when to trust the order-flow call?* — a genuinely new experiment, not a
re-run of the forbidden Phase-19 test.

## The primary side rule (fixed a priori, causal, NOT fitted)
For every micro-label event the **primary side** is `sign(ofi_top)` evaluated as-of `t0`
(the `m1` top-of-book order-flow-imbalance feature): `+1` long when `ofi_top > 0`, `-1`
short when `ofi_top < 0`. It is a **deterministic causal function of past data** — not a
fitted model — so it needs no cross-validation and introduces **no primary-model leakage
and no nested out-of-fold construction**. Events with `ofi_top == 0` (no imbalance) have no
side and are excluded (ES 1, NQ 2 rows). The meta-model is the **sole fitted component**.

## The meta-label (binary, fixed a priori)
For each event with primary side `s`, using the realized micro-label `label` (`+1` upper /
`-1` lower / `0` timeout-or-close): `meta_label = 1` iff `label == s` (the primary called
the correct barrier side), else `0` (wrong direction **or** a `0` timeout — the bet did not
pay). It reads only the already-resolved `label`; purge/embargo on `t1` apply exactly as in
Phase 14. The base rate `mean(meta_label)` — the primary's unconditional precision — is
**ES 0.122 / NQ 0.103**.

## The two arms
- **B0 — always-act reference:** take side `s` on every event (no gate). The primary's
  unconditional performance; the reference the meta-gate must beat. Expected to fail.
- **M — meta-gated (PRIMARY VERDICT arm):** a binary LightGBM meta-model over the `m1` OFI
  block **+** the `s2` `sent_*` block **+** the primary's `|ofi_top|` magnitude predicts
  `P(meta_label = 1)`; **act when `P > τ` (τ = 0.5, fixed, never tuned)**, else flat. Gross
  proxy `s · ret_t1` when acting, `0` when flat.

Bars, labels, the **fold-local balanced class weighting** (now over the binary classes),
the inherited LightGBM params, the **8-config search grid** (`max_depth [2,3]` ×
`min_child_samples [50,150]` × `reg_lambda [5,20]`, selection on the **meta-gated**
`gross_signal_sharpe`), the purged+embargoed K-fold, CPCV (5 paths), PBO, the trial-
deflated DSR, and `seed 7` are read unchanged from `config/micro_model.yaml` (`mm1`); τ,
the meta-skill floor, the window and the primary feature come from `config/phase20.yaml`.
Sentiment nulls (pre-`2026-03-10`) are **kept** (LightGBM-native NaN), never imputed.

## Verdict gates
The **five inherited mm1 gross gates** are applied **unchanged** to arm M's meta-gated
signal: PBO < 0.5, gross DSR > 0.5, mean gross return > 0, action rate ≥ 0.03, CPCV median
gross Sharpe > 0. The mm1 sixth gate (3-class macro F1 ≥ 0.20) is **undefined for a binary
meta-model** and is replaced — transparently, fixed a priori — by a **meta-skill gate**:
arm M is credited only if **both** (a) the meta-model's out-of-fold balanced accuracy ≥
0.52, **and** (b) the acted-on hit-rate strictly exceeds the always-act (B0) hit-rate (the
gate adds precision over acting on everything). This directly tests the one thing
meta-labeling claims to do — filter false positives.

## Run — measured 2026-06-13
Window `t0 ∈ [2026-01-26, 2026-06-06]` (the **full** Phase-14 window, per symbol), to
maximize effective N. Meta-set: **ES 1,688 rows (effective N 1,089.3), NQ 1,519 (1,023.1)**.

### Per-symbol, per-arm gate table (✓ pass / ✗ fail)

| Symbol | Arm | selected config | PBO `<0.5` | gross DSR `>0.5` | mean gross `>0` | action `≥0.03` | CPCV med `>0` | meta-skill | verdict |
|--------|-----|-----------------|-----------|------------------|-----------------|----------------|---------------|-----------|---------|
| **ES** | B0 (ref) | — | n/a ✗ | 0.290 ✗ | +2.4e-5 ✓ | 1.000 ✓ | −0.0135 ✗ | ✗ | **no edge** |
| **ES** | M (meta) | d2 / mcs50 / λ20 | 0.318 ✓ | 0.060 ✗ | +1.2e-5 ✓ | 0.394 ✓ | −0.0101 ✗ | **✓** | **no edge** (2 fail) |
| **NQ** | B0 (ref) | — | n/a ✗ | 0.425 ✗ | +4.7e-6 ✓ | 1.000 ✓ | −0.0048 ✗ | ✗ | **no edge** |
| **NQ** | M (meta) | d3 / mcs150 / λ20 | **0.722 ✗** | 0.247 ✗ | −1.2e-5 ✗ | 0.507 ✓ | −0.0272 ✗ | **✓** | **no edge** (4 fail) |

(B0 has no search ⇒ PBO undefined, and acts on everything ⇒ meta-skill fails by
construction; its CPCV is degenerate — a fixed predictor's paths are identical.)

### Meta-skill detail (the gate DID add precision — just not enough)
| Symbol | balanced acc (≥0.52) | always-act hit (B0) | acted-on hit (M) | meta-skill |
|--------|----------------------|---------------------|------------------|------------|
| **ES** | 0.522 ✓ | 0.122 | **0.134** ✓ | **pass** |
| **NQ** | 0.575 ✓ | 0.103 | **0.130** ✓ | **pass** |

### Arm-M SHAP — does the gate use sentiment?
- **ES (full window):** `ofi_top`, `sent_60m_sum_score`, `rv_intrabar`,
  `sent_240m_topic_inflation_mean_score`, `sent_60m_topic_inflation_mean_score`,
  `duration_s`. **Sentiment share of total |SHAP| = 0.74** (supported region: 0.75).
- **NQ (full window):** `signed_vol_roll3`, `ofi_top`, `sent_15m_topic_fed_mean_score`,
  `spread_ticks_close`, `sent_60m_mean_neg`, `sent_1d_topic_fed_mean_score`. **Sentiment
  share = 0.69** (supported region: 0.71).

The meta-gate leaned **heavily** on the `s2` block (≈70–75% of SHAP mass) — this is **not**
a "the gate ignored sentiment" null.

## Attribution and decision (per the frozen rule)
- **Per symbol, primary verdict = does arm M clear all five inherited gates AND the
  meta-skill gate?** ES fails 2/6 (gross DSR, CPCV median); NQ fails 4/6 (PBO, gross DSR,
  mean gross, CPCV median). **Both fail.**
- **DECISION: `no_significant_edge` — meta-labeling is the next honest null.** No edge
  candidate, nothing fragile, no symbol authorized for a Phase 21 backtest.

## Honest interpretation
- **The meta-skill gate passed on both symbols** — the binary gate genuinely *did* what
  meta-labeling promises: it raised the acted-on hit-rate above the always-act base rate
  (ES 0.122 → 0.134, NQ 0.103 → 0.130) at balanced accuracy 0.52 / 0.57. So the gate adds
  real precision. **But the lift is too small** to make the gross signal profitable after
  deflation: gross DSR collapses (ES 0.06, NQ 0.25, both ≪ 0.5) and every CPCV path median
  is negative on both symbols. A marginally-more-precise filter on a signal whose per-trade
  edge is essentially zero still has no deflated edge.
- **NQ also blew out PBO to 0.722** — the 8-config search found an in-sample-stronger gate
  whose selection is overfit (the same failure mode that killed Phase-19 ES treatment).
- **Sentiment was used, not ignored** (≈70–75% of the gate's SHAP mass), and still did not
  generalise — consistent with the Phase-19 finding that the `s2` block is in-sample
  prominent but out-of-sample empty.
- **Because meta-labeling is the canonical remedy for exactly this low-precision/imbalanced
  regime, its failure is strong evidence that the binding constraint is sample size / edge
  existence at this horizon, not model framing.** This escalates the strategic fork (acquire
  more data/depth, redesign the horizon/market, or accept the result). The meta-labeling
  lever is **not re-litigated** (no τ tuning, no primary-rule swapping).
- **No model is promoted; nothing trades.**

> **This is not a strategy and not an economic backtest.** All return figures are a gross
> signal-return proxy with no commissions or slippage. No strategy, backtest, risk,
> execution, or live trading is authorized.

## Run it
```sh
# Meta-labeling edge verdict, both arms, both symbols, full window:
uv run python -m options_system.microstructure_model.phase20_meta --symbols ES NQ
#   flags: --no-mlflow  --no-interpret  --rebuild-cache

# Read-only summary (gate table + meta-skill + attribution + decision):
uv run python -m options_system.observability.phase20_meta_health --symbols ES NQ
```
Per-arm summaries → `data/phase20_meta/runs/<symbol>_<arm>.json`; combined verdict →
`data/phase20_meta/runs/verdict.json` (gitignored). MLflow experiment
`micro-signal-model-phase20-meta` (local file store).
