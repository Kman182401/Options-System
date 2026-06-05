# Labeling (`label_version = v1`)

The **target** the signal model will learn to predict — not the trading rules.
This is the triple-barrier method (López de Prado, *Advances in Financial Machine
Learning*, ch. 3–4) applied to MES/MNQ, with volatility-scaled barriers, CUSUM
event sampling, and overlap-aware sample weights.

> **Labels ≠ strategy.** The barriers define a *supervised-learning target* —
> "from this moment, does price hit `+kσ` before `−kσ` within a max hold?" They
> are **not** the live entry/exit rules (those come later). A label is *allowed*
> to look forward (that is what a target is); a **feature is not**. The hard rule:
> labels never feed back into the feature set, and every label records its
> resolution time `t1` so the validation phase can purge/embargo overlap.

---

## Why triple-barrier

A fixed-horizon return ("the return 3 days from now") is an unrealistic target: it
ignores that a real, defined-risk position is closed when it hits a profit-take or
a stop, whichever comes first, and otherwise at a time limit. Triple-barrier
encodes exactly that: from each event `t0` we place

- an **upper** (profit-take) barrier at `+pt_mult · σ` cumulative log-return,
- a **lower** (stop-loss) barrier at `−sl_mult · σ`,
- a **vertical** (time) barrier at `t0 + max_hold_bars`,

and the label is decided by the **first barrier touched**:

| first touch | label | `barrier` |
|---|---|---|
| upper | `+1` | `up` |
| lower | `−1` | `dn` |
| vertical (time runs out) | `0` (or `sign(ret)` if `vertical_label_sign`) | `time` |
| roll, in `close` mode | `0` (or `sign(ret)`) | `roll` |

The model then learns "is a tradeable `±kσ` move coming within the hold?" — which
is what the strategy actually cares about.

---

## The instrument and the price path (roll-safe, back-adjustment-invariant)

Barrier touches are computed on the **real tradeable instrument** — the raw
front-month contract a position would hold — **in return / log space**, never on
naively back-adjusted *levels*.

Concretely we walk the production back-adjusted **continuous** close
(`store.get_bars(continuous=True)`) in **cumulative log-return** space from `t0`:
`cr_τ = ln(close_τ) − ln(close_{t0})`. Within one contract that equals the raw
front-month return (the back-adjustment factor cancels in a difference). Across a
roll, ratio back-adjustment makes the seam return ≈ 0 — which *is* the realistic
"rolled position": rolling is not itself a P&L event (the roll spread is a cost,
modelled later). Because we only ever use return **differences**, the labels are
**degree-0 in the price scale** → identical under any global rescale of price.
This is the same back-adjustment-invariance the features rely on, and it is
proven directly: `tests/test_labeling_triple_barrier.py::test_labels_invariant_to_global_price_scale`
rescales every price by a constant and asserts the labels, `t1`, `barrier` and
`ret` are unchanged.

### Roll handling (config `roll.handling`)

A 3-day hold can span a quarterly roll. Two honest options, both **flag**
`roll_crossed = true`:

- **`adjust`** (default) — walk the continuous return path straight through the
  seam (the rolled position). `barrier ∈ {up, dn, time}`.
- **`close`** — cap the path at the roll bar and resolve there: if no price
  barrier was hit yet, `barrier = roll`, `t1 =` the roll bar.

We never silently back-adjust *levels* to compute a touch.

---

## Volatility (sets the barrier width and the CUSUM threshold)

`σ_t` is **causal**: an EWM standard deviation of 1-bar log returns
(`adjust=False`, only past bars), scaled to a session horizon:

```
σ_scaled_t = ewm_std(1-bar log returns, span=ewm_span) · sqrt(barrier_horizon_bars)
```

Scaling a clean 1-bar σ by `√H` (default `H = 390` ≈ one RTH session) makes "1σ" a
meaningful intraday move instead of a one-minute wiggle, while keeping the
estimator transparent and easy to test. σ is `null` during warmup
(`min_samples`); events there are dropped. σ is degree-0 (built from return
differences) so it is back-adjustment-invariant too, and causal (an `adjust=False`
EWM recursion at `t` depends only on bars `≤ t` —
`tests/test_labeling_events.py::test_sigma_is_causal_truncation_invariant`).

The √H scaling is a deliberate, fixed-a-priori modelling choice (it assumes
roughly iid bar returns over the horizon — imperfect intraday, but it introduces
**no look-ahead** and the barriers are a *target definition*, not a P&L claim).

---

## Event sampling (don't label every bar)

Labeling every bar produces enormous, almost-identical overlapping labels. The
**CUSUM filter** (default) emits an event only when the cumulative signed return
since the last event exceeds a σ-scaled threshold, then resets:

```
s_pos = max(0, s_pos + r_t);   s_neg = min(0, s_neg + r_t)
emit & reset s_pos  when  s_pos ≥  cusum_mult · σ_scaled_t
emit & reset s_neg  when  s_neg ≤ −cusum_mult · σ_scaled_t
```

This collapses noise into the handful of bars where something actually moved, so
labels are sparse (see counts below) and not redundant. A deterministic `grid`
alternative (every `grid_step_bars`) is also provided.

> The CUSUM threshold is a **knob, not a result-chaser**. It is set for a sensible
> event frequency, never tuned to make labels "look good".

---

## Sample weights (account for label overlap; AFML ch. 4)

Overlapping labels are correlated — two events a few bars apart share most of
their outcome window. Training on them as independent over-counts that region.

1. **Concurrency** `c_t` — how many labels' `[t0, t1]` windows cover bar `t`.
2. **Average uniqueness** of label *i* — the mean of `1/c_t` over the bars it
   spans (`1.0` if it never overlaps, → 0 as it piles up).
3. **Sample weight** — average uniqueness, optionally × `|realized return|`
   (`scheme = uniqueness_return`, return attribution), then an optional linear
   **time-decay** (`time_decay`, AFML 4.11; `1.0` = none), normalized so weights
   average 1.0.

Hand-computed references in `tests/test_labeling_weights.py` pin the math.

---

## Barriers are fixed *before* the model sees them

`pt_mult` / `sl_mult` default to a **symmetric 1.5σ** — a defined-risk,
regime-adaptive target that matches the system's intraday→3-day style. Asymmetry
is *allowed* but is a knob to fix a priori, **never** to optimize on a backtest:
choosing barriers to maximize results is overfitting. Class imbalance (below) is
**expected** and is surfaced, not "fixed" by retuning barriers — it is handled
later at the model stage (class weights / thresholds).

---

## Output schema (one row per resolved event)

Written to `data/labels/symbol=<SYM>/date=<YYYY-MM-DD>/` (zstd, idempotent on
`t0`). Keyed by `(symbol, t0)`.

| column | meaning |
|---|---|
| `t0` | event time (the bar the label is anchored at) |
| `t1` | **resolution time** (first-touch / vertical / roll). `t0 ≤ t1 ≤ t0+max_hold`. **Not optional** — the validation phase needs it to purge overlap. |
| `symbol` | MES / MNQ |
| `ret` | realized cumulative log-return from `t0` to `t1` |
| `label` | `+1` / `−1` / `0` |
| `barrier` | `up` / `dn` / `time` / `roll` |
| `sigma` | `σ_scaled` used to set this event's barriers |
| `n_bars` | holding length in bars (`t1 − t0`) |
| `contract_id` | raw front-month contract at `t0` |
| `roll_crossed` | the `[t0, t1]` window spanned a contract roll |
| `session` | RTH / ETH at `t0` |
| `degraded` | `t0`…`t1` touched a Databento-flagged degraded day (inherited from `features.yaml`) |
| `avg_uniqueness` | average uniqueness (overlap metric) |
| `weight` | normalized sample weight |
| `side`, `meta_label` | **meta-labeling hook** — null until implemented (structure only) |
| `label_version` | stamped from config |

**Right-censoring is handled honestly.** An event whose vertical window runs past
the end of available data *and* never touches a price barrier is **dropped**, not
resolved early — its outcome is genuinely unknown without future bars.

---

## Retrieval (leak-free features at `t0`)

`labeling/build.py::labels_with_features` attaches the **as-of** feature row at
each `t0` via the store's `asof_join` (`feature.ts_event ≤ t0`), yielding an
aligned `(features@t0, label, t1, weight, …)` matrix with no look-ahead — proven
in `tests/test_labeling_build.py::test_features_at_t0_join_is_leak_free`. A label
may look forward; the features attached to it never do.

---

## Meta-labeling hook (deferred — structure only)

The schema carries `side` and `meta_label`, and `build.py::apply_meta_labeling` is
the reserved API. Meta-labeling adds a *secondary* model that decides whether to
act on (and how to size) a primary signal's `side`. Powerful, but **deferred** —
Phase 3 builds the seat, not the logic.

---

## Build it / look at it

```bash
# build versioned label tables over full history (idempotent; re-run = 0 new rows)
uv run python -m options_system.labeling.build --symbols MES MNQ

# read-only health view: class balance, barrier dist, uniqueness, % roll-crossed
uv run streamlit run src/options_system/observability/labels_health.py
```

### Class balance + barrier distribution (`label_version = v1`, full history 2019→2026)

| symbol | n | `+1` | `−1` | `0` (time) | % roll-crossed | mean avg-uniqueness |
|---|---|---|---|---|---|---|
| MES | 10,835 | 0.495 | 0.469 | 0.036 | 4.7% | 0.235 |
| MNQ | 11,691 | 0.502 | 0.468 | 0.031 | 4.6% | 0.234 |

Timeouts are rare because CUSUM fires on momentum and a ±1.5σ move within ~3 days
usually resolves first; the mild `+1` skew reflects the 2019–2026 bull drift.
This imbalance is reported, not engineered away. Barriers, σ estimator, CUSUM
threshold and weighting are all in `config/labeling.yaml` and versioned by
`label_version`. Rationale: `docs/DECISIONS.md` Phase 3.
