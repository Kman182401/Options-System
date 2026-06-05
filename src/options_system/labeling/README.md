# labeling/

**Builds the supervised-learning target — what the signal model will learn to
predict.** This is the **triple-barrier method** (López de Prado, AFML ch. 3–4):
from each sampled event `t0`, does price hit `+pt·σ` (→ `+1`) before `−sl·σ` (→
`−1`) within a max hold, or does the vertical time barrier expire first (→ `0`)?
The barriers encode the system's defined-risk, intraday→3-day style as a
learnable target.

> **Labels ≠ strategy.** These barriers define a *target*, not live entry/exit
> rules. And a **label is allowed to look forward** (that is what a target is) —
> a **feature is not**. The hard rule enforced here: labels never feed back into
> the feature set, and every label records its resolution time `t1` so the
> validation phase (Prompt 4) can purge/embargo overlapping samples.

## What's built (Phase 3, `label_version = v1`)

- **`config.py` + `config/labeling.yaml`** — declarative, validated, versioned
  config: barrier multiples, σ estimator, vertical length, event method, roll
  handling, weighting scheme.
- **`events.py`** — causal `σ_t` (EWM std of 1-bar log returns, √-scaled to a
  session) + the **CUSUM** event sampler (López de Prado), with a `grid`
  alternative. σ sets both the barrier width and the CUSUM threshold.
- **`triple_barrier.py`** — the first-touch generator. Walks the **continuous**
  close in cumulative log-return space (the real front-month, degree-0 →
  back-adjustment-invariant), records `t1`, `barrier`, `ret`, `roll_crossed`,
  `session`, `degraded`. Right-censored tail events are dropped, not mislabeled.
- **`weights.py`** — concurrency → **average uniqueness** → sample weights
  (optional return-attribution + time-decay), AFML ch. 4.
- **`build.py`** — versioned, idempotent label-table writer into
  `data/labels/symbol=…/date=…/` (computed over full history) + leak-free
  retrieval (`read_labels`, `labels_with_features` via the store's `ASOF JOIN`,
  `feature.ts_event ≤ t0`). Carries the **meta-labeling hook** (`side` /
  `meta_label` columns + `apply_meta_labeling` API — structure only, deferred).

Proofs in `tests/test_labeling_*.py`: exact CUSUM crossings, σ causality,
first-touch / `t1` bounds / σ-scaling / roll-crossing / **back-adjustment
invariance**, hand-computed uniqueness, idempotent writes, and the leak-free
features-at-`t0` join.

Run it:

```fish
uv run python -m options_system.labeling.build --symbols MES MNQ
uv run streamlit run src/options_system/observability/labels_health.py
```

Full method + barrier rationale + class balance: `docs/LABELING.md`; decisions:
`docs/DECISIONS.md` Phase 3. No model, cross-validation, strategy, risk, or
execution here — those are later phases.
