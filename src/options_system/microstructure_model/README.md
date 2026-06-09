# microstructure_model — Phase-14 micro signal model (3-class, honest verdict)

The first model trained on the MBP-1 microstructure dataset (m1 order-flow features
on dollar bars, ml1 ~30-min triple-barrier labels). It is **separate** from the
daily price/macro/TA model (`models/`): different bars, different labels, a 3-class
target, and fold-local class weighting for the ~78-80 % timeout regime.

This is a **signal verdict only** — not a strategy, not an economic backtest. Nothing
here trades, and even an "edge candidate" authorises only the *next* phase (an
economic backtest with realistic costs/slippage), never live trading.

## Target (3-class, never collapsed)
`label ∈ {-1, 0, +1}` = lower barrier / vertical-timeout-or-close / upper barrier.
The predicted class **is** the signal: `-1` short, `0` flat/no-trade, `+1` long. The
dominant timeout class is modelled, not dropped and not folded into a binary by sign.

## Fold-local class weighting
Class weights are computed INSIDE each training fold from `y_train` only (never
globally — see `tests/test_microstructure_model_weights.py`), then multiplied into the
persisted uniqueness/sample weights: `eff_w_i = sample_weight_i · class_weight[y_i]`.
With `use_sample_weight_in_balance` the balance equalises each class's *effective
mass*. The early-stopping inner-val weights use the same fold-local mapping.

## Validation (reused Phase-4 machinery, no random CV)
Purged + embargoed K-fold (pooled OOS), Combinatorial Purged CV (path distribution),
PBO across the 8 search configs, PSR/DSR on the gross signal returns. Purge/embargo
use each label's `t1`, so overlapping labels never leak across the boundary.

## Metrics
- **Classification:** weighted + balanced accuracy, macro F1, confusion matrix, true
  & predicted class distributions, action rate (`pred != 0`).
- **Gross signal-return proxy:** `gross = pred_class · ret_t1` (NO costs/slippage),
  excess-over-long `(pred-1)·ret_t1`, gross/excess Sharpe, gross PSR/DSR, CPCV path
  distribution. **Not an executable backtest.**

## Verdict gates (fixed a priori in `config/micro_model.yaml`)
An **edge candidate** requires ALL of: PBO < `max_pbo`, gross DSR > `min_gross_dsr`,
mean gross return > 0, action rate ≥ `min_action_rate`, macro F1 ≥ `min_macro_f1`, and
CPCV median gross Sharpe > 0. Otherwise **no significant edge**.

## Run
```bash
uv run python -m options_system.microstructure_model.run \
    --symbols ES NQ --start 2026-01-26 --end 2026-06-06
# flags: --no-mlflow  --no-interpret  --rebuild-cache  --start/--end
```
Summaries → `data/micro_models/runs/<symbol>.json` (gitignored). Health view:
```bash
uv run python -m options_system.observability.micro_model_health --symbols ES NQ
```

## Files
`model_config.py` (typed config, in `microstructure/`) · `dataset.py` (leak-free
matrix) · `lgbm.py` (3-class wrapper + fold-local weights) · `tune.py` (in-CV search)
· `evaluate.py` (CPCV/PBO/DSR + verdict) · `interpret.py` (optional SHAP) ·
`tracking.py` (MLflow) · `run.py` (CLI).
