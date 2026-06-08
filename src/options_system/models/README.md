# models/

**Trains, evaluates honestly, and explains the directional signal model — entirely
offline.** The model is **LightGBM** (gradient-boosted trees: fast on CPU,
interpretable), and Phase 5 answers one question: *does it beat chance on the
triple-barrier label, after deflation, over and above market beta?* Training is part
of the **offline learning loop**, never the live loop. Overfitting is the enemy: no
result is trusted on in-sample numbers, and a null result is a valid outcome.

## What's here (Phase 5, implemented)

| file | role |
|---|---|
| `config.py` + `../../config/models.yaml` | typed, versioned config (`model_version`): regularisation, search grid, verdict thresholds |
| `dataset.py` | **fast** leak-free `(features@t0, y_dir, t1, weight)` matrix (pushed-down DuckDB read + polars `join_asof`; seconds, exact match to Phase-4) |
| `lgbm.py` | regularized LightGBM **directional** classifier (weight-aware, deterministic) + purged-inner-early-stopping fold fitter |
| `tune.py` | small hyperparameter grid run **inside** the purged CV; `n_trials` counted for the DSR |
| `evaluate_model.py` | honest evaluation through `validation/` (CPCV, PBO, trial-deflated DSR), skill **over beta** → explicit **VERDICT** |
| `interpret.py` | SHAP global + local importances, dominance/leakage sanity check |
| `tracking.py` | local MLflow file store (`data/mlruns`, no cloud) |
| `run.py` | the pipeline: `load → search → evaluate → interpret → track`, per symbol |

```bash
uv run python -m options_system.models.run --symbols MES MNQ
```

Everything is judged **only** through the Phase-4 framework; this module adds no
leakage logic, only the beta-netting and trial-deflation that phase deferred. Method
and the current verdict (**no significant edge** on MES/MNQ — iterate on data/
features before any strategy): `docs/MODEL.md`. Rationale: `docs/DECISIONS.md` Phase 5.

## Deferred (built nothing yet)

The **model registry**, the **champion–challenger promotion** gate, live
retraining, and serving an approved artifact to the live engine are **out of scope**
until a configuration actually clears the verdict's four gates. The live engine will
merely *load* an approved champion and run inference — never train.
