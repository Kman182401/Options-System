# PHASE19_AB.md — Sentiment micro-model A/B edge verdict (Phase 19)

## What it is
A controlled A/B that answers one pre-registered question: **does adding the `s2`
sentiment feature block to the Phase-14 microstructure micro-model produce a real,
deflated edge — over and above the order-flow-only baseline — on the rows that actually
have sentiment coverage?** The full frozen contract is
[`docs/PHASE19_PREREGISTRATION.md`](PHASE19_PREREGISTRATION.md) (committed `380ed03`,
before any modeling). This run must not — and does not — deviate from it.

It is a **signal edge verdict only**: every return number is a gross signal-return proxy
(`pred_class · ret_t1`, **no commissions, no slippage**). It is **not a strategy and not
an economic backtest**, nothing trades, and it authorizes **no live trading**. Reads only
the local lakes — no Databento, no IBKR, no network, no spend.

## The two arms (the only delta is the sentiment block)
- **Baseline (B)** — `with_sentiment=False` → the unchanged 17 `m1` OFI features. Stamp `mm1`.
- **Treatment (T)** — `with_sentiment=True` → the OFI features **plus** the `s2` `sent_*`
  aggregate block, attached on `t0` via `sentiment.join.attach_to_micro_labels`. Stamp `mm2`.

Bars, labels, target, folds, fold-local class weighting, the 8-config search
(`max_depth [2,3]` × `min_child_samples [50,150]` × `reg_lambda [5,20]`, selection on
`gross_signal_sharpe`), CPCV (5 paths), PBO, the trial-deflated DSR, `seed 7`, and the
**six verdict gates** are read unchanged from `config/micro_model.yaml` (`mm1`). The
sentiment block is the sole experimental variable. The sentiment nulls (no prior events)
are **kept** (LightGBM-native NaN), never imputed; the ~98%-constant `1d`/`240m` `has_any`
flags are included as-is (a tree cannot split a constant — filtering them would be a new
pre-registered knob). Both arms run on the **identical** supported-region row set per
symbol (asserted; a mismatch aborts the A/B).

## Verdict gates (unchanged from `mm1`)
An edge candidate (per symbol, per arm) must clear **ALL** of: PBO < 0.5, gross DSR > 0.5,
mean gross return > 0, action rate ≥ 0.03, macro F1 ≥ 0.20, **CPCV median gross Sharpe > 0**.

## Run — measured 2026-06-13
Window `t0 ∈ [2026-03-10, 2026-06-06]` (the Phase-18 supported archive region), per symbol.
Row counts match the pre-registration (ES 1,132 / NQ 1,036); **effective N (Σ uniqueness)
is 735.5 / 696.3** — somewhat below the full-window Phase-14 ~1,000 because the supported
region is a ~3-month subset. Power to detect a small edge after deflation is correspondingly
limited — exactly as the contract anticipated ("an honest null is the expected and fully
acceptable outcome").

### Per-symbol, per-arm gate table (✓ pass / ✗ fail)

| Symbol | Arm | selected config | PBO `<0.5` | gross DSR `>0.5` | mean gross `>0` | action `≥0.03` | macro F1 `≥0.20` | CPCV med `>0` | verdict |
|--------|-----|-----------------|-----------|------------------|-----------------|----------------|------------------|---------------|---------|
| **ES** | B `mm1` | d2 / mcs150 / λ5 | 0.476 ✓ | 0.156 ✗ | −6.1e-6 ✗ | 0.650 ✓ | 0.298 ✓ | −0.0167 ✗ | **no edge** (3 fail) |
| **ES** | T `mm2` | d2 / mcs150 / λ20 | **0.734 ✗** | 0.840 ✓ | +6.5e-5 ✓ | 0.766 ✓ | 0.258 ✓ | +0.0100 ✓ | **no edge** (1 fail: PBO) |
| **NQ** | B `mm1` | d3 / mcs50 / λ5 | 0.278 ✓ | 0.077 ✗ | −2.0e-5 ✗ | 0.670 ✓ | 0.266 ✓ | −0.0397 ✗ | **no edge** (3 fail) |
| **NQ** | T `mm2` | d2 / mcs150 / λ20 | 0.496 ✓ | 0.081 ✗ | +1.6e-5 ✓ | 0.891 ✓ | **0.177 ✗** | −0.0341 ✗ | **no edge** (3 fail) |

### Treatment-arm SHAP (which sentiment features it actually used)
- **ES:** `sent_1d_topic_rates_mean_score`, `sent_1d_max_abs_score`, `rv_intrabar`,
  `sent_15m_max_abs_score`, `sent_240m_topic_rates_count`, `sent_60m_mean_neg`.
- **NQ:** `sent_1d_topic_rates_mean_score`, `sent_15m_max_abs_score`, `sent_1d_mean_neg`,
  `sent_240m_topic_fed_count`, `sent_60m_topic_inflation_mean_score`, `signed_vol_roll3`.

The treatment arm leaned **heavily** on sentiment — `sent_1d_topic_rates_mean_score` is the
#1 feature on **both** symbols, and it comes from the score aggregates and shorter windows,
exactly where the contract said signal would have to live (not the near-constant presence
flags). So this is **not** a "the model ignored sentiment" null: the block was prominent
in-sample and still did not generalise out-of-sample.

## Attribution and decision (per the frozen rule)
- **Per symbol, primary verdict = does Treatment clear all six gates?** Both fail → **null on
  both symbols**, so attribution is `null` for each (a T failure is a null regardless of B,
  and B also fails on both — the Phase-14 OFI-only result reproduced on the restricted rows).
- **DECISION: `no_significant_edge` — sentiment is the fifth honest null.** No edge candidate,
  nothing fragile, no symbol authorized for a Phase-20 backtest.

## Honest interpretation
- **The fifth dead lever** after price (Phase 5), macro (Phase 6), TA (Phase 10) and
  microstructure/order-flow (Phase 14). The bar never moved; only the inputs did.
- **ES treatment was the near-miss** — it passed 5/6, including a positive CPCV median
  (+0.010) and a gross DSR of 0.84, and adding sentiment moved ES's DSR and CPCV median in
  the "better" direction versus baseline. But **PBO blew out from 0.476 (baseline) to 0.734
  (treatment)**: the sentiment block let the 8-config search find an in-sample-stronger
  config whose selection is **overfit**, which is precisely what the PBO gate exists to
  catch. The marginal-looking DSR did not survive the selection-overfit test. (This mirrors
  the Phase-14 NQ near-miss, where a marginally positive pooled Sharpe failed the CPCV gate.)
- **NQ treatment** failed 3/6 (gross DSR, macro F1, CPCV median) — an unambiguous null; the
  high action rate (0.89) with low macro F1 (0.18) shows it traded a lot without class skill.
- **No model is promoted; nothing trades.** Per the frozen decision rule the remaining forks
  are MBP-10 multi-level order-flow depth (**paid, billing-gated, blocked**) or a deliberate
  horizon/regime redesign — **not** sentiment re-tuning. The lever is not re-litigated.

> **This is not a strategy and not an economic backtest.** All return figures are a gross
> signal-return proxy with no commissions or slippage. No strategy, backtest, risk,
> execution, or live trading is authorized.

## Run it
```sh
# A/B edge verdict, both arms, both symbols, supported region (t0 >= 2026-03-10):
uv run python -m options_system.microstructure_model.phase19_ab --symbols ES NQ
#   flags: --no-mlflow  --no-interpret  --rebuild-cache

# Read-only summary (gate table + attribution + decision):
uv run python -m options_system.observability.phase19_ab_health --symbols ES NQ
```
Per-arm summaries → `data/phase19_ab/runs/<symbol>_<arm>.json`; combined verdict →
`data/phase19_ab/runs/verdict.json` (gitignored). MLflow experiment
`micro-signal-model-phase19-ab` (local file store).
