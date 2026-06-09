# Short-horizon labeling (`micro_label_version = ml1`)

The intraday sibling of [`docs/LABELING.md`](LABELING.md). Same López de Prado
triple-barrier method — symmetric CUSUM event sampling, ±σ horizontal barriers, a
vertical time barrier, an honest `t1`, and average-uniqueness sample weights —
**re-scaled from the ~3-day daily horizon to a 30-minute intraday horizon** and run
on the microstructure **dollar bars** (`micro_bars`, see [`MICROSTRUCTURE.md`](MICROSTRUCTURE.md))
instead of the 1-minute price bars.

This is an additive, separately-versioned layer. It does **not** touch the daily
`v1` labels, the price features, the macro layer, the OFI feature layer (`m1`), or
the validation framework. Labels live in their own lake (`data/micro_labels/`),
isolated from `data/labels/`. Config: [`config/micro_labeling.yaml`](../config/micro_labeling.yaml).
Code: `src/options_system/microstructure/labels.py` + `label_config.py`.

## What is reused vs new

**Reused unchanged** (imported from the daily layer, not reimplemented):

- `labeling.events.cusum_events` — the symmetric CUSUM filter.
- `labeling.weights.sample_weights` — concurrency-based average uniqueness +
  normalized weights (and hence effective N).

**New here**, because the dollar bars are not time-uniform and a short intraday
horizon has a session-close leakage trap the daily layer never faces:

1. **σ scaled to a wall-clock horizon** (not a fixed bar count).
2. **A wall-clock (30-min) vertical barrier** mapped to the first bar past the mark.
3. **Session-close guards**: per-session processing + a final-30-min exclusion + a
   hard cap at the close.

## Volatility (sets the barrier width and the CUSUM threshold)

σ is causal: an EWM standard deviation of **per-bar mid (`mid_close`) log returns**
(`adjust=False` → trailing recursion; null during the first `min_samples` bars).
It is then scaled to the 30-minute horizon. Dollar bars are constructed to carry
~equal variance per bar, so the 30-min variance is `(bars per 30 min) · (per-bar
variance)`. The bar rate is itself estimated **causally** from an EWMA of bar
`duration_s`:

```
σ_H = σ_bar · sqrt(vertical_seconds / EWMA(duration_s))
```

This is the direct mirror of the daily estimator (`σ_bar · sqrt(barrier_horizon_bars)`),
with a *causal, time-varying bars-per-horizon* in place of the daily layer's fixed
bar count — the correct generalization for variable-duration bars. Every value at
bar `t` uses only bars with `ts_event ≤ t` (proven in
`tests/test_micro_labeling.py::test_sigma_uses_only_past_bars`). On the 5-day slice
σ_H is stable (ES/NQ ≈ 0.26–0.33% per 30 min), with no non-finite blow-ups.

## Event sampling (don't label every bar)

Symmetric CUSUM on per-bar mid log returns, threshold `h_t = cusum_mult · σ_H,t`,
reset on each event. **Event sampling deliberately uses returns, not the OFI
features**, so the labels are not circular with the order-flow signal we will test
in Prompt 9. CUSUM resets at every session boundary (each session block is
processed independently), so an overnight gap return can never trigger an event.

## The vertical barrier is 30 minutes of wall-clock time

Not a fixed bar count — dollar-bar durations vary. The vertical barrier maps to the
first bar whose close `ts_event ≥ t0 + 30 min`. Order-flow signal is expected to
carry in the minutes-to-~30-min band; 30 min caps the intended max hold while
letting the horizontal barriers trigger earlier.

## Session-boundary handling (the key short-horizon leakage trap)

**No label may cross the RTH session close.** Three guards, all structural:

1. **Per-session processing.** Everything is computed per `(contract_id, ET-date)`
   block. The forward barrier scan is confined to the block, so a label can
   physically never read a bar from the next session (or across a roll).
2. **Final-30-min exclusion.** Events whose `t0 + 30 min` would fall after the
   session close (16:00 ET) are dropped, so every retained label gets its full
   intraday horizon inside one session.
3. **Hard cap at the close.** If no bar reaches the 30-min mark before the session
   ends, the label is resolved at the last in-session bar (`barrier_touched =
   "close"`, `resolved_at_close = True`). On the liquid ES/NQ slice this never
   binds (dollar bars are dense), but the guard is there for thin tails.

The RTH session (09:30–16:00 ET, half-open) is **not redefined** here — it is reused
from `config/microstructure.yaml`'s `session` block. The 16:00-ET close timestamp
is constructed per session exactly as `microstructure.ingest._day_window` does.

## Barriers are fixed *before* the model sees them

`±1.5σ` mirrors the daily multiplier; the vertical is 30 min; the CUSUM `cusum_mult`
is `1.0` (same relationship as the daily config). **All parameters are committed a
priori in `config/micro_labeling.yaml`.** They were not tuned to chase a nicer label
balance or more events — pre-commitment is the anti-snooping guard. The heavy
skew toward the timeout label (below) is reported honestly, not corrected.

## Label value

Sign of the first barrier touched: `+1` upper, `−1` lower, `0` if the vertical or
session close is reached with no horizontal touch (`vertical_label_sign = false`).

## Output schema (one row per resolved event)

Stored at `data/micro_labels/symbol=<SYM>/date=<YYYY-MM-DD>/part-<uuid>.parquet`,
partitioned by `t0`'s date, zstd, idempotent on the natural key `t0`
(re-running writes 0 new rows). Read it back with `labels.read_micro_labels`.

| column | meaning |
|---|---|
| `t0`, `t1` | event bar close / first-barrier-touch (or close-cap) timestamp, UTC |
| `symbol`, `contract_id` | `ES`/`NQ`; the raw contract the window stayed within |
| `session_date` | ET trading date of the session |
| `label` | `−1` / `0` / `+1` (sign of first barrier touched) |
| `barrier_touched` | `upper` / `lower` / `vertical` / `close` |
| `ret_t1` | realized cumulative mid log-return at `t1` |
| `sigma` | σ_H at `t0` (the barrier-width scale) |
| `n_bars` | dollar bars held from `t0` to `t1` |
| `resolved_at_close` | true iff hard-capped at the session close |
| `uniqueness_weight` | average uniqueness (LdP, AFML ch. 4), in (0, 1] |
| `sample_weight` | uniqueness × decay, normalized to mean 1.0 |
| `micro_label_version` | `ml1` |

## QA on the 5-RTH-day validation slice (2026-05-18 → 2026-05-22)

Computed on the bars already on disk; no Databento pulls. `cusum_mult = 1.0`.

| symbol | events sampled | retained | balance +/−/0 | hold (min) median [IQR] | resolved@close | avg uniqueness | **effective N** | **eff N / RTH day** |
|---|---|---|---|---|---|---|---|---|
| ES | 109 | 101 | 0.13 / 0.09 / 0.78 | 30.1 [30.0, 30.2] | 0.000 | 0.602 | 60.8 | **12.2** |
| NQ | 95 | 89 | 0.11 / 0.11 / 0.78 | 30.1 [30.0, 30.3] | 0.000 | 0.667 | 59.3 | **11.9** |

Barrier distribution — ES: vertical 79, upper 13, lower 9. NQ: vertical 69, upper 10,
lower 10.

**Honest reading (this is a label step, not a trading result):**

- **The slice is small.** ~12 effective labels per RTH day per symbol. A short-horizon
  model that must pass the existing overfitting gate (PBO/PSR/DSR) will need far more
  — at ~12 eff/day, reaching ~1,000 effective labels per symbol implies roughly **80
  RTH days (~4 months)** of MBP-1 dollar bars. This is the number that should size the
  Prompt 9 Databento pull.
- **The labels are heavily skewed to the timeout (`0`) class (~78%).** Expected: a
  ±1.5σ_H band is wide relative to the typical 30-min move *after* a CUSUM event has
  already consumed ~1σ. This was **not** tuned away — the ±1.5σ multiplier is the
  a-priori commitment mirrored from the daily layer. A short-horizon model will likely
  need class weighting or meta-labeling, not a relaxed barrier.
- **Average uniqueness ≈ 0.6**, so effective N is ~60% of the raw label count — the
  30-min windows overlap meaningfully and uniqueness weighting is doing real work.
- **`resolved_at_close` is 0** on this liquid slice; the hard cap is a correctness
  guard that simply never binds here.

## Verification (all green)

- **Leakage teeth** (`test_micro_labeling_leakage.py`): a label is invariant to any
  bar strictly after its `t1`, while a value that peeks one bar past `t1` flips under
  the same perturbation (the check has teeth); truncating the bars at `t1` reproduces
  the label, and truncating *inside* the window changes it.
- **Session boundary**: every retained label resolves in its own ET session, strictly
  before 16:00 ET; final-30-min events are excluded; perturbing the next session
  leaves this session's labels byte-identical.
- **σ causality**: σ at `t0` uses only bars `≤ t0`.
- **Determinism**: re-running reproduces identical labels/`t1`/weights; labels are
  invariant to a global price rescale (degree-0 in the price scale).

## Build it / look at it

```bash
# build versioned micro-label tables over all on-disk bars (idempotent; re-run = 0 new rows)
uv run python -m options_system.microstructure.labels
# write-window subset (compute is always full):
uv run python -m options_system.microstructure.labels --start 2026-05-18 --end 2026-05-22
```

Label-config + QA stats (params, events, balance, avg uniqueness, effective N) are
logged to the local MLflow file store (`data/mlruns`, experiment `micro-labels`).
No model is trained here — that is Prompt 9.

## Phase 12 — relabel over the extended dataset (measured 2026-06-09)

After the Phase 12 MBP-1 pull (window `2026-02-16 → 2026-06-06`, ES/NQ, ~79 RTH
sessions with data — see `docs/MICROSTRUCTURE.md`), the `ml1` labels were rebuilt
over **all** on-disk micro bars. No parameters were changed (still `pt=sl=1.5σ`,
30-min vertical, `cusum_mult=1.0`, uniqueness weights). **No model was trained.**

```bash
uv run python -m options_system.microstructure.labels --start 2026-02-16 --end 2026-06-06
```

Re-running the exact command wrote **+0 rows** (idempotent, latest-ingest-wins on
`t0`). Label QA (`observability.micro_label_health`, read from the `micro_labels`
lake):

| | labels | effective N | eff N / RTH day | balance +/−/0 | timeout (label 0) | avg uniqueness | hold median | resolved_at_close | NULL/INF |
|---|---|---|---|---|---|---|---|---|---|
| ES | 1,418 | **916.3** | 11.6 | 0.099 / 0.121 / 0.781 | ~78% | 0.646 | 30.1 min | 0.1% | none |
| NQ | 1,287 | **862.9** | 10.9 | 0.089 / 0.110 / 0.801 | ~80% | 0.670 | 30.2 min | 0.2% | none |

Barriers — ES `{vertical: 1106, lower: 171, upper: 140, close: 1}`; NQ `{vertical:
1029, lower: 142, upper: 114, close: 2}`. `t0` spans 2026-02-16 → 2026-06-05 for
both. Versions present: `ml1` only.

**Target check.** The pull was sized to reach **~1,000 effective labels/symbol**.
Result: **ES 916 (≈92%), NQ 863 (≈86%) — near, but not quite reached.** The 30-min
horizon + CUSUM sampling + ~0.65 average uniqueness yields ~11 effective labels per
RTH day, so the ~79-session window lands just short of 1,000. The class mix is the
expected high-timeout regime (~78–80% label 0), unchanged from Phase 8 — *not*
tuned. Core columns are null/inf-clean.

> Labeling + QA spend **zero** Databento credits (they read the local lake only).

### Read-only label-QA CLI (added Phase 12)

`src/options_system/observability/micro_label_health.py` reads the `micro_labels`
lake and reports the table above (reusing `labels.label_qa`, plus version / `t0`
span / null-inf checks). Pure summarizer unit-tested in
`tests/test_micro_label_health.py`.

```bash
uv run python -m options_system.observability.micro_label_health \
    --symbols ES NQ --start 2026-02-16 --end 2026-06-06
```

## Phase 13 — top-off relabel (clears the ~1,000 effective-N target, measured 2026-06-09)

After the Phase 13 backward MBP-1 top-off (window `2026-01-26 → 2026-02-16`,
non-overlapping, +15 RTH sessions per symbol — see `docs/MICROSTRUCTURE.md`), the
`ml1` labels were rebuilt over **all** on-disk micro bars (the 94-session combined
window `2026-01-26 → 2026-06-06`). No parameters were changed (still `pt=sl=1.5σ`,
30-min vertical, `cusum_mult=1.0`, uniqueness weights). **No model was trained.**

```bash
uv run python -m options_system.microstructure.labels --start 2026-01-26 --end 2026-06-06
```

The first run wrote **+271 (ES) / +234 (NQ)** new labels (the 15 backward
sessions); an immediate second run wrote **+0 / +0** (idempotent, latest-ingest-wins
on `t0`). Label QA (`observability.micro_label_health`, read from the
`micro_labels` lake):

| | labels | effective N | eff N / RTH day | balance +/−/0 | timeout (label 0) | avg uniqueness | hold median | resolved_at_close | NULL/INF |
|---|---|---|---|---|---|---|---|---|---|
| ES | 1,689 | **1,090.0** | 11.6 | 0.098 / 0.124 / 0.778 | 77.8% | 0.645 | 30.1 min | 0.1% | none |
| NQ | 1,521 | **1,024.3** | 10.9 | 0.086 / 0.114 / 0.799 | 79.9% | 0.673 | 30.2 min | 0.1% | none |

Barriers — ES `{vertical: 1313, lower: 210, upper: 165, close: 1}`; NQ `{vertical:
1214, lower: 174, upper: 131, close: 2}`. `t0` spans 2026-01-26 → 2026-06-05 for
both. Versions present: `ml1` only.

**Target check.** The sizing goal was **~1,000 effective labels/symbol**. Result:
**ES 1,090 and NQ 1,024 — both now clear 1,000.** The ~11 effective labels/RTH-day
rate is unchanged from Phase 12 (the labels themselves are untouched); the extra 15
sessions supplied the remaining count. The class mix is the expected high-timeout
regime (~78–80% label 0), unchanged from Phase 8 — *not* tuned. Core columns are
null/inf-clean.

> Labeling + QA spend **zero** Databento credits (they read the local lake only).
> This top-off relabel covers the combined window without re-spending the prior
> Phase 12 pull.

**Next.** With both symbols past the effective-N target and QA clean, the next
phase is **microstructure LightGBM training/validation** — fold-local class
weighting (high-timeout regime), unchanged labels, unchanged validation gates, and
no economic strategy/backtest yet.
