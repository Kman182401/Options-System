# MICRO_MODEL.md — Phase-14 microstructure signal model + honest verdict

## What it is
The first model trained on the MBP-1 **microstructure** dataset: a 3-class LightGBM
classifier over the m1 order-flow features (`microstructure_feature_version=m1`,
17 features on dollar bars) with the ml1 short-horizon triple-barrier labels
(`micro_label_version=ml1`, ~30-min horizon). `micro_model_version=mm1`.

It is a **signal-verdict** exercise only — it answers *“is there a real, deflated
microstructure edge candidate?”* It is **not a strategy and not an economic
backtest**, nothing here trades, and the return numbers are a **gross signal-return
proxy with no commissions or slippage**.

## Why it is a separate model from the daily one
The daily price/macro/TA model (`models/`, `config/models.yaml`) is a *binary
directional* model on 1-minute price bars with daily-horizon triple-barrier labels.
This one is deliberately separate: different bars (event-clock dollar bars), different
labels (intraday ~30-min horizon), a **3-class** target, and **fold-local class
weighting** for the ~78–80 % timeout regime. Mixing them would muddy both. It lives in
its own package (`microstructure_model/`) with its own config
(`config/micro_model.yaml`) and its own MLflow experiment (`micro-signal-model`).

## Target: −1 / 0 / +1 (never collapsed)
The micro label is the barrier outcome: `−1` = lower barrier, `0` =
vertical-timeout-or-session-close, `+1` = upper barrier. The **predicted class is the
signal** — `−1` short, `0` flat/no-trade, `+1` long. The dominant timeout class is
**modelled, not dropped and not folded into a directional binary by the sign of the
return.** Predicting "flat" is a legitimate, first-class output (no trade), which is
why a 3-class target is the right shape for a short-horizon order-flow signal.

## Why class weighting is fold-local
Label 0 is ~78–80 % of the data, so an unweighted model just predicts "0" everywhere
(high accuracy, zero trades, zero signal). Class weights fix that — but they must be
computed **inside each training fold from `y_train` only**, never from the whole
dataset, or the weighting leaks the global class balance across the CV boundary. For
each fold we compute balanced weights from that fold's `y_train`
(`use_sample_weight_in_balance` makes the balance equalise each class's effective
*mass*, not its raw count) and **multiply** them into the persisted uniqueness/sample
weights: `eff_w_i = sample_weight_i · class_weight[y_i]`. The early-stopping inner
validation block uses the **same** fold-local mapping. Proven in
`tests/test_microstructure_model_weights.py`.

## How validation works (reused Phase-4 machinery, no random CV)
All numbers come from the leak-safe framework:
- **Purged + embargoed K-fold** (`PurgedKFold`) for the pooled out-of-sample pass —
  every sample scored OOS exactly once.
- **Combinatorial Purged CV** (`CombinatorialPurgedCV`) for a *distribution* of OOS
  paths (here 5 paths) — not one fragile number.
- **PBO** across the 8 search configs (selection-overfit probability).
- **PSR / DSR** on the gross signal returns, deflated by the 8 trials.

Purge and embargo use each label's `t1`, so overlapping ~30-min labels never leak
across a fold boundary. There is **no random train/test split** anywhere.

## Metrics reported
- **Classification:** weighted accuracy, balanced accuracy, macro F1, confusion
  matrix, true + predicted class distributions, action rate (`pred ≠ 0`).
- **Gross signal-return proxy:** `gross = pred_class · ret_t1`, excess-over-long
  `(pred−1)·ret_t1`, gross/excess Sharpe, gross PSR/DSR, and the CPCV path
  distribution. **No costs — not an executable backtest.**

## Verdict gates (fixed a priori in `config/micro_model.yaml`, never tuned)
An **edge candidate** requires ALL of: PBO < 0.5, gross DSR > 0.5, mean gross return
> 0, action rate ≥ 0.03, macro F1 ≥ 0.20, and **CPCV median gross Sharpe > 0**.
Otherwise **no significant edge**. An "edge candidate" would authorise only the next
phase (an economic backtest with realistic costs/slippage) — never live trading.

## First real run — measured 2026-06-09
Window `2026-01-26 → 2026-06-06` (the full combined dataset), ES + NQ, 8 trials,
selection metric = gross signal Sharpe. Class balance (true) is the expected
~78–80 % timeout regime: ES `−1/0/+1 = 0.124/0.778/0.098`, NQ `0.114/0.800/0.086`.

| | n | eff N | selected config | action | wAcc | balAcc | macro F1 | gross SR | gross DSR | mean gross | PBO | CPCV gross SR (med [min..max]) | **verdict** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ES | 1,689 | 1,090 | d3 / mcs150 / λ5 | 0.666 | 0.353 | 0.355 | 0.279 | −0.0232 | 0.030 | −1.1e-05 | **0.897** | −0.0296 [−0.053..−0.010] | **no significant edge** |
| NQ | 1,521 | 1,024 | d3 / mcs50 / λ5 | 0.629 | 0.401 | 0.394 | 0.298 | +0.0199 | 0.505 | +3.6e-05 | 0.333 | **−0.0230** [−0.061..−0.005] | **no significant edge** |

Predicted-class balance: ES `−1/0/+1 = 0.416/0.335/0.250`, NQ `0.546/0.372/0.083`.

Gate detail:
- **ES fails 4 of 6 gates** — PBO 0.897 (severe selection overfit), gross DSR 0.030,
  negative mean gross return, and a negative CPCV median. An unambiguous null.
- **NQ passes 5 of 6** — PBO 0.333 ✓, gross DSR 0.505 ✓ (barely), positive mean gross
  ✓, action ✓, macro F1 ✓ — but **fails the deciding CPCV gate**: every one of the 5
  out-of-sample paths is negative (median −0.023, *max* −0.005). The marginally
  positive pooled gross Sharpe was a near-zero artefact that the honest path
  distribution does not survive. This is exactly why the CPCV median is a gate.

## Honest interpretation
- **Verdict: no significant edge — both symbols.** Microstructure is the fourth dead
  lever after price (Phase 5), macro (Phase 6) and TA (Phase 10).
- **The prediction did NOT collapse to the timeout class.** Fold-local class weighting
  worked: action rate is 0.63–0.67 and the model trades all three classes. The cost of
  that is low weighted accuracy (0.35 / 0.40, well below the 0.78–0.80 "predict-0"
  base rate) — expected and intended; accuracy is not the objective here.
- **OFI features behave as designed but don't generalise.** For NQ the SHAP top is
  order-flow-led (`signed_vol_roll3`, `ofi_top`, `ofi_top_lag1`, `micro_minus_mid_twa`);
  for ES it is `depth_top_twa`, `rv_intrabar`, `signed_vol_roll3`, `ret_bar`,
  `duration_s`, `ofi_top`. No single feature dominates (no leakage smell). The model
  leans on the right signals — they simply don't carry an out-of-sample edge at this
  horizon and feature depth.
- **No model is promoted; nothing trades.** A later escalation (if pursued) would be
  MBP-10 multi-level OFI (deeper book) rather than re-tuning this model — and that is a
  separate, cost-gated decision.

> **This is not a strategy and not an economic backtest.** All return figures are a
> gross signal-return proxy with no commissions or slippage.

## Run it
```bash
uv run python -m options_system.microstructure_model.run \
    --symbols ES NQ --start 2026-01-26 --end 2026-06-06
# flags: --no-mlflow  --no-interpret  --rebuild-cache
uv run python -m options_system.observability.micro_model_health --symbols ES NQ
```
Summaries → `data/micro_models/runs/<symbol>.json` (gitignored). No Databento, no
IBKR, no network — reads only the local lakes.
