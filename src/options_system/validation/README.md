# `validation/` — leakage-safe evaluation framework (Phase 4)

Model-agnostic machinery that judges any estimator honestly: purged/embargoed
cross-validation, Combinatorial Purged CV (a *distribution* of out-of-sample
paths), walk-forward, and the overfitting statistics (PBO, PSR, DSR, minimum
track-record length). It builds nothing that trades — it is the truth-detector
every model passes through before it earns a place.

Full explanation in plain English: `docs/VALIDATION.md`. Rationale:
`docs/DECISIONS.md` (Phase 4).

## Modules

| File | Responsibility |
|---|---|
| `config.py` | Typed, versioned config (`config/validation.yaml`, `validation_version`). |
| `_purge.py` | Core purge + embargo primitive — the one place leakage logic lives. |
| `purged_kfold.py` | sklearn-compatible purged + embargoed K-fold. |
| `cpcv.py` | Combinatorial Purged CV + OOS path reconstruction (`C(N,k)` splits, `C(N-1,k-1)` paths). |
| `walk_forward.py` | Anchored / rolling sequential validation. |
| `stats.py` | PBO (CSCV), Probabilistic / Deflated Sharpe Ratio, minTRL. |
| `evaluate.py` | The harness: estimator + leak-free matrix → metric distribution + overfitting verdict + effective N. |

## The contract

* Every splitter **purges** training samples whose `[t0, t1]` window overlaps a
  test fold and applies a forward **embargo**, using the labeling layer's `t1`
  resolution times. Leakage is impossible by construction, and proven so by the
  teeth test.
* Scoring honours the Phase-3 uniqueness **sample weights** and reports
  **effective sample size** (Σ uniqueness) — ~11k labels ≈ ~2.7k effective.
* Deterministic under the config seed.

## Run

```bash
uv run python -m options_system.validation.evaluate --symbols MES MNQ
uv run streamlit run src/options_system/observability/validation_health.py
```

## Out of scope (later phases)

No real signal model, no `nautilus` economic backtest, no strategy/risk/execution,
no sentiment, no hyperparameter search. The harness *supports* a search later (it
must run inside the CV and be counted as trials for the DSR) but runs none now.
