# Decision Log

Running log of non-obvious choices. Newest at the bottom of each section. Keep
entries short: what was decided, and *why*.

---

## Phase 0 — Bootstrap (2026-06-05)

### Python & packaging
- **Python 3.12** (uv-managed CPython **3.12.13**). nautilus_trader supports
  `>=3.12,<3.15`; we pin `>=3.12,<3.13` so the interpreter is unambiguous and
  reproducible. No system `python3.12` existed; uv fetches its own.
- **uv** for env + dependency management. `uv.lock` is committed and is the
  reproducibility source of truth (exact, hashed versions).
- **hatchling** build backend; `src/options_system` installed as an editable
  package so `import options_system` works everywhere. `config/` deliberately
  lives at the repo root (not under `src/`) so it imports as `config.settings`.

### GPU / PyTorch (the one genuinely tricky pin)
- GPU is an **RTX 5070 Ti (Blackwell, sm_120, 16 GB)**; driver reports CUDA UMD
  13.3. Blackwell `sm_120` kernels first shipped in PyTorch's **cu128** builds.
- **torch pinned to `2.9.1+cu128`**, pulled from the dedicated index
  `https://download.pytorch.org/whl/cu128` via `[tool.uv.sources]` +
  `[[tool.uv.index]] explicit = true`. The driver (CUDA 13.3) is
  forward-compatible with the cu128 runtime.
- Chose **cu128 (CUDA 12.8) over newer cu130/cu132 + torch 2.12**: cu128 is the
  most battle-tested Blackwell path and keeps the broader ecosystem
  (transformers, accelerate) on well-trodden ground. The newest CUDA 13.x torch
  buys little here and adds risk. Revisit if a real need appears.

### Library versions (resolved by uv, 2026-06-05)
| Package | Version | Notes |
|---|---|---|
| nautilus_trader | 1.227.0 | engine (backtest == live) |
| ib_async | 2.1.0 | IBKR API (maintained ib_insync successor) |
| lightgbm | 4.6.0 | signal model |
| scikit-learn | 1.9.0 | ML utilities |
| polars | 1.41.2 | fast dataframe |
| pandas | 2.3.3 | nautilus requires `>=2.3.3,<3` |
| numpy | 2.4.6 | numpy 2.x; all deps compatible |
| duckdb | 1.5.3 | local query engine |
| pyarrow | 24.0.0 | nautilus requires `>=23.0.1` |
| torch | 2.9.1+cu128 | Blackwell sm_120 (see above) |
| transformers | 5.10.2 | FinBERT (text-classification) |
| accelerate | 1.13.0 | transformers GPU helper |
| pydantic | 2.13.4 | typed models |
| pydantic-settings | 2.14.1 | typed config + YAML source |
| python-dotenv | 1.2.2 | `.env` loading |
| loguru | 0.7.3 | logging |
| pyyaml | 6.0.3 | config.yaml |
| streamlit | 1.58.0 | dashboard |
| python-telegram-bot | 22.7 | alerts |
| pytest | 9.0.3 | tests (dev) |
| ruff | 0.15.16 | lint + format (dev) |

### Config & safety
- **`mode` hard-locked to `"paper"`** in `config/settings.py` via a validator:
  any other value (e.g. `MODE=live`) makes `Settings()` refuse to load. Live
  trading must never be reachable by configuration alone.
- Config precedence (high→low): init args → env vars → `.env` →
  `config/config.yaml` → field defaults. Secrets are `SecretStr` (never printed).
- Risk-limit fields exist as a **typed placeholder surface only** — no logic
  reads them in Phase 0.

### IBKR defaults
- Default `ibkr_port=4002` (IB Gateway **paper** API). TWS paper would be 7497.
  Documented in `docs/SETUP.md`.

---

## Phase 1.1 — Data layer (2026-06-05)

### Storage
- **Parquet lake** at `data/<dataset>/symbol=<SYM>/date=<YYYY-MM-DD>/`. Datasets:
  `bars_1m`, `bars_5s`, `quotes_l1`, `trades`, `roll_events`. Every market-data
  row carries `ts_event` (exchange) + `ts_ingest` (receipt), both UTC — the
  no-look-ahead foundation.
- **Append-only + idempotent** writer (`lake.py`): natural-key dedupe (e.g. bars
  keyed on `(contract_id, ts_event)`), never overwrites a partition, zstd.
  Re-recording a day cannot duplicate. Small files accumulate; `compact()` merges
  them. `symbol` is kept as a data column (not hive-parsed on read) to avoid a
  path/column clash.
- **DuckDB session pinned to UTC** in `store.py` (`SET TimeZone='UTC'`) so every
  `TIMESTAMPTZ` read comes back UTC-aware (DuckDB otherwise renders in the local
  zone and clashes with our UTC literals).

### Point-in-time correctness
- All retrieval goes through `store.py`. `asof_join` uses DuckDB `ASOF JOIN`
  (`left.ts_event >= right.ts_event`) so a row can never see future auxiliary
  data. Proven leak-free by test.

### Continuous contracts
- **Back-adjustment convention: ratio (multiplicative), default.** Panama
  (difference) also implemented (`continuous_adjustment` setting). Ratio keeps
  returns/percentage moves consistent across the seam, which suits an intraday
  return-based strategy. Raw per-`contract_id` data is never mutated; the
  continuous series is **derived on demand** (not materialized) so we can change
  the convention later without re-recording.
- Roll rule: volume/OI crossover with a calendar fallback `roll_calendar_days`
  (default 5) before expiry. Each roll is recorded in `roll_events`.

### New dependencies
- **`databento==0.79.0`** — historical backfill client (loader is scaffold-only,
  cost-guarded, no-ops without a key; never auto-downloads).
- **`exchange-calendars` deliberately NOT added** — gap detection uses the RTH
  session tag + a threshold (deterministic, dep-free). Calendar-precise
  (holiday-aware) gap detection is deferred until we actually need it, keeping
  the dependency set minimal.

### IBKR / IBC
- **IB Gateway 10.45** installed at `~/ibgateway`; **IBC 3.23.0** at `~/ibc`.
- IBC auto-login is **scaffolded** (`scripts/render_ibc_config.py` +
  `start_gateway.fish` + systemd user units). Credentials come from `.env`
  (`OPTIONS_IBKR_USERNAME` / `OPTIONS_IBKR_PASSWORD`) and are rendered into a
  **tmpfs** config (mode 600) — never persisted to disk or git. Read-only API
  (the recorder never trades). Unverified until first paper login; manual launch
  is the proven fallback.

### Recorder
- Records **L1 + bars only**. L2/market depth is a marked extension point in
  `recorder.py` (deferred — needs the paid CME depth subscription).
- `reqRealTimeBars` gives 5-second bars → stored as `bars_5s` and aggregated to
  `bars_1m` (true 1-second bars aren't available from this API). Session tagged
  RTH/ETH via an America/New_York 09:30–16:00 window.
- New settings: `record_symbols` (MES, MNQ), `recorder_client_id` (11, distinct),
  `recorder_flush_seconds` (30), `roll_calendar_days` (5), `continuous_adjustment`
  (ratio).

### Config isolation (`OPTIONS_` prefix) + stale-env cleanup
- During bootstrap we found the dev machine's shell (`~/.config/fish/conf.d/env.fish`)
  exported global `IBKR_HOST`, `IBKR_PORT=4003`, `IBKR_CLIENT_ID`,
  `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`. Because `pydantic-settings` reads
  same-named env vars, these silently bled into `Settings()` (e.g. the paper
  default 4002 was overridden by ambient 4003).
- **Decision: prefix every env/.env key with `OPTIONS_`** (`env_prefix="OPTIONS_"`).
  The project now reads only `OPTIONS_*` and is fully isolated from ambient/global
  shell state. `config.yaml` keys stay unprefixed (the YAML source matches field
  names). This is a deliberate, documented deviation from the bootstrap prompt's
  literal key names, justified by the paper-only safety posture + "understand the
  whole system" prime directive.
- **Stale-env cleanup:** the `IBKR_*` exports were leftovers from abandoned,
  deleted options projects with **zero remaining consumers** — removed from
  `env.fish` (backup: `env.fish.bak-ibkr-removal-1780656600`). The `TELEGRAM_*`
  exports were **kept**: they are live secrets shared by active systems
  (braxen-app, voice-assistant, security-tools, airbnb-bot), so removing them
  would break those. The `OPTIONS_` prefix is what stops this project from
  inheriting that shared Telegram bot — use a dedicated bot via `.env` instead.

---

## Phase 1.5 — Databento historical backfill (2026-06-05)

### What landed (spend gate honored)
- **Symbols MES + MNQ, schema `ohlcv-1m` only**, dataset **`GLBX.MDP3`**, window
  **2019-05-06 (micros' inception) → 2026-06-05 (dataset end)**. Full history.
- **Cost $26.89** (actual == dry-run estimate), **~0.41 GB uncompressed**,
  **7,364,934 rows**. The mandatory dry-run estimate was presented and explicitly
  approved before any `--confirm` spend. ~21% of the $125 free credit. (Note:
  `ohlcv-1m` bills at **$70/GB** — pricier per GB than the trades feed's volume
  would suggest; that is what made it ~$27 rather than "a few dollars".)

### Symbology — parent `.FUT`, raw per-contract retained
- Pulled with **parent symbology** (`MES.FUT` / `MNQ.FUT`, `stype_in=parent`).
  This returns **individual contract months as raw per-contract rows**
  (`contract_id` = the raw expiry symbol, e.g. `MESM9`), so `continuous.py`
  remains the sole roll authority. Databento `to_df()` defaults used:
  `price_type=float` (prices in dollars), `map_symbols=True` (the `symbol` column
  is the raw contract), `pretty_ts=True`.
- **Parent `.FUT` also returns calendar-spread instruments** (`contract_id` with a
  dash, e.g. `MNQU0-MNQZ0`): MES 76 spreads / 80,267 rows, MNQ 59 spreads /
  58,707 rows. Spread prices legitimately go negative (min −39.9). **Decision:
  keep them in the raw lake (raw retention; already paid), but exclude them
  (`contract_id` containing `-`) from continuous stitching and from the
  "outrights" validation view.** Not deleted, not repaired. Re-evaluate whether to
  purge spreads later.

### Validation result (report-only, nothing repaired)
- **Outright contracts validate clean**: MES 33 outrights / 3,509,950 rows and
  MNQ 33 outrights / 3,716,010 rows — **0** duplicate / monotonic / OHLC / ingest
  errors. Only warnings are `gaps` (RTH minutes > 120s on illiquid back-months —
  expected; gaps stay gaps).
- The **only** error-severity findings on the full set (24,079 OHLC rows) are the
  calendar spreads' negative prices — valid for spreads, excluded above.
- **Databento flagged 11 reduced-quality ("degraded") days**: 2020-02-27,
  2020-02-28, 2020-06-30, 2021-12-05, 2022-01-02, 2025-09-17, 2025-09-24,
  2025-11-28, 2026-03-15, 2026-03-16, 2026-04-10. Recorded, not altered.

### Continuous stitch
- **28 quarterly rolls per symbol** (2019Q2 → 2026Q2), volume/OI crossover with the
  5-day calendar fallback. **Ratio** back-adjustment (the configured default);
  **seam continuity gap = 0.0%** by construction. `roll_events` persisted (28 ×2).
  Continuous series stays **derived on demand** via `store.get_bars(continuous=True)`.
- **Expiry derivation** (no extra `definition`-schema spend): from the raw code
  `{SYM}{H|M|U|Z}{year-digit}` → month ∈ {Mar,Jun,Sep,Dec}, **expiry = 3rd Friday**
  of that month (verified against last-trade dates). Single-digit year decode for
  this 2019-2027 span: `9` → 2019, `0..8` → 2020..2028.

### Loader / lake fixes made against the real API (in-scope, data layer only)
- **`databento_loader.py`**: added **per-year chunking + retry** to
  `_download_and_store` (resumable; the idempotent lake write skips chunks already
  on disk). Removed the now-**deprecated `mode=` arg** from `get_cost`.
- **`lake.py::scan`**: was `Path().glob(<absolute pattern>)`, which raises
  `NotImplementedError: Non-relative patterns are unsupported` on 3.12 (the lake
  root is absolute). Switched to stdlib `glob`, matching `store.py`. It had only
  been exercised on relative paths before.
- **CLIs added** so the steps are reproducible: `python -m options_system.data.validate`
  (`--outrights-only`) and `python -m options_system.data.continuous`.

### Key handling
- `OPTIONS_DATABENTO_API_KEY` was **not** in `.env`/env; the key lives in `pass`
  at `databento/api_key`. The estimate and the download were run by bridging it
  into the process env for that command only (`OPTIONS_DATABENTO_API_KEY="$(pass
  show databento/api_key)" …`) — **no secret written to `.env` or disk**. To make
  future runs (feature phase, re-backfill) turnkey, add it to `.env` or keep
  bridging from `pass`.

---

## Phase 2 — Feature engineering (2026-06-05)

### The leakage principle (the whole design rests on this)
- **Every price feature is degree-0 in the price scale** (returns, ratios,
  z-scores, normalized) — never a raw price/ATR/VWAP **level**. Why: truncating
  the raw history at `t` and rebuilding the ratio-adjusted continuous series only
  multiplies the point-in-time series (rows `<= t`) by a single global constant
  `f_k = ∏ future seam ratios`. A degree-0 feature is invariant to that constant;
  a level feature scales by `f_k` and so secretly encodes the future roll. This is
  the "back-adjustment trap" and it is the reason level features are banned here.
- **Proven, not asserted.** `tests/test_features_leakage.py` rebuilds the
  continuous series point-in-time (raw bars `<= t` + only rolls `<= t`, exactly
  like `store.get_bars(continuous=True)`) and asserts
  `feature(full)[t] == feature(truncated)[t]` for all 45 features, plus a "teeth"
  test showing a raw level genuinely fails. Tolerance is `1e-4` relative: most
  features match to ~1e-12; ADX (a long Wilder-EWM chain over `close*f_k`) drifts
  ~1e-4 from float rounding of the scaling — far below the ~1% a real leak moves.

### Feature set (≈45, interpretable, hand-picked — no kitchen sink)
- Families: returns, trend/momentum (EMA slope, EMA-distance z, MACD on **log**
  price, ROC, ADX), mean-reversion (RSI, Bollinger %B, price z-score), volatility
  (realized vol, ATR%, Parkinson, Garman-Klass, vol-regime), volume (relative
  volume, **time-of-day-baselined** volume, volume z, normalized signed-volume
  flow, session-VWAP distance), time/session (sin/cos of minute-of-day &
  day-of-week, minutes since/to RTH open/close), cross-asset MES↔MNQ (return
  spread, raw-ratio z-score, return correlation). Full table + roll-safety in
  `docs/FEATURES.md`. tsfresh / auto-feature-explosion deliberately avoided.
- **MACD on log price** (not raw) so it stays a difference of log-EMAs → degree-0.
- **Volume is back-adjustment independent** (back-adj only rewrites prices), so
  volume features use raw volume; they are normalized against a **time-of-day
  baseline** (prior-N-days, same minute) because intraday volume is strongly
  seasonal — otherwise the feature mostly encodes "what time it is".
- **Cross-asset ratio uses the raw front price** recovered as `close/adj_factor`
  (the true recorded price, no back-adjustment), since a ratio of two
  independently-back-adjusted continuous series is NOT invariant. The other
  symbol is attached by a **backward as-of join** (contemporaneous-or-earlier).
- **Session-anchored VWAP** resets per CME trade date (overnight session `>= 18:00`
  ET belongs to the next date); emitted only as `close/VWAP − 1` (a ratio).

### Storage, config, flags
- **Declarative config** in `config/features.yaml` → typed `FeatureConfig`
  (`features/config.py`), with a `feature_version` stamped on every row. Windows
  are in bars (= minutes). A disabled `news` seat is a documented placeholder (no
  news/macro data ingested — nothing built).
- Feature tables at `data/features/symbol=…/date=…/` (zstd, idempotent on
  `ts_event`), **computed over full history** so values never depend on the write
  window. Retrieval via the DuckDB store + `asof_join` keeps no-look-ahead
  end-to-end.
- **`degraded` flag** carried on every row = warmup (row index `< max_window`,
  4140) OR a Databento-flagged degraded day. Per-feature nulls during their own
  warmup are kept null — never fabricated. `session` (RTH/ETH) carried too.

### Tooling
- **No new dependency.** Indicators are Polars-native (full causal control, no
  TA-Lib C build), spot-checked in tests against independent pandas/NumPy Wilder
  references (RSI, ATR%, VWAP). All windows trailing; no centered windows, no
  negative shifts.
- **Spreads excluded** end-to-end: the engine reads the outright-only continuous
  series and hard-rejects any `contract_id` containing `-` (tested).

---

## Phase 3 — Labeling (2026-06-05)

### The target is triple-barrier (not a fixed-horizon return)
- **Why:** a fixed "return N days out" target ignores that a real defined-risk
  position is closed at a profit-take, a stop, or a time limit — whichever comes
  first. Triple-barrier (López de Prado, AFML ch. 3) encodes exactly that: from
  each event `t0`, first touch of `+pt·σ` → `+1`, `−sl·σ` → `−1`, vertical (time)
  → `0`. The model learns "is a tradeable `±kσ` move coming within the hold?",
  which is what the strategy actually cares about. Full method: `docs/LABELING.md`.
- **Labels ≠ strategy, and labels may look forward — features may not.** The hard
  invariant enforced here: labels never feed back into features, and **every label
  records `t1`** (resolution time) so the validation phase (Prompt 4) can
  purge/embargo overlapping samples. Without `t1` every CV leaks.

### Barriers fixed a priori; class imbalance surfaced, not tuned away
- **Symmetric ±1.5σ**, vertical = **4140 bars (~3 CME sessions)** — a defined-risk
  target matching the intraday→3-day style. These are set from a vol/risk
  rationale **before** any model sees them. Choosing barriers to maximize a
  backtest is overfitting and is explicitly avoided.
- Observed balance (full 2019→2026): MES `{+1:0.495, −1:0.469, 0:0.036}`, MNQ
  `{+1:0.502, −1:0.468, 0:0.031}`. Timeouts are rare (CUSUM fires on momentum → a
  ±1.5σ move usually resolves within 3 days); mild `+1` skew = the bull drift.
  Imbalance is **reported** (health view) and handled later at the model stage
  (class weights / thresholds), never by retuning barriers to balance classes.

### Volatility: causal 1-bar EWM std, √-scaled to a session
- `σ_scaled = ewm_std(1-bar log returns, span=100) · sqrt(390)`. A clean,
  testably-causal 1-bar σ scaled by `√H` (H = one RTH session) makes "1σ" a
  meaningful intraday move. The √H scaling is a fixed-a-priori modelling choice
  (assumes ~iid bar returns; imperfect intraday but introduces **no look-ahead** —
  the barriers are a target definition, not a P&L claim). σ sets **both** the
  barrier width and the CUSUM threshold from one knob.

### Back-adjustment-invariance (same principle as features)
- Touches are computed on the **real tradeable instrument** in **return/log
  space**, never on back-adjusted *levels*. We walk the continuous close in
  cumulative log-return space from `t0`; within a contract that is the raw
  front-month return, and across a roll the ratio adjustment makes the seam return
  ≈ 0 (the realistic "rolled position" — the roll spread is a cost, modelled
  later). Because only return *differences* are used, labels are **degree-0** →
  invariant to any global price rescale. Proven directly (rescale every price by a
  constant → identical labels/`t1`/`barrier`/`ret`).

### Roll-crossing windows: handled and flagged, never silently back-adjusted
- A 3-day hold can span a quarterly roll (~4.6% of labels do). `roll.handling`:
  **`adjust`** (default) walks the continuous return path through the seam;
  **`close`** caps at the roll bar (`barrier = roll`). Both **flag**
  `roll_crossed`. We never compute a touch on naively back-adjusted levels.

### Event sampling: CUSUM (a knob, not a result-chaser)
- **CUSUM filter** (AFML §2.5.2.1) emits an event only when the cumulative signed
  return since the last event exceeds `cusum_mult · σ_scaled`, then resets —
  collapsing noise into the ~10–12k bars that actually moved (vs millions). Far
  fewer, non-redundant labels. A deterministic `grid` alternative exists. The
  threshold is set for sensible frequency, never to make labels look good.

### Sample weights from overlap (AFML ch. 4)
- Concurrency → **average uniqueness** (mean `1/concurrency` over a label's span)
  → sample weight, optionally × `|return|` (return attribution) and an optional
  linear **time-decay**, normalized to mean 1.0. Mean avg-uniqueness ≈ 0.23 (labels
  do overlap, so this matters). Hand-computed references pin the math in tests.

### Right-censoring handled honestly
- An event whose vertical window runs past the end of available data **and** never
  touches a price barrier is **dropped**, not resolved early — its outcome is
  unknown without future bars. Resolving it at the last bar would be a shorter,
  biased hold.

### Storage / config / hooks
- Own writer at `data/labels/symbol=…/date=…/` (zstd, idempotent on `t0`),
  mirroring the features writer rather than extending `lake.DATASETS` — labels are
  a derived analytic table, not raw market data. Computed over **full history** so
  values never depend on the write window. `label_version` stamped on every row;
  all barrier/vol/sampling/weighting params live in `config/labeling.yaml`.
- `degraded` + `session` inherited from the **feature config** (single source of
  truth for degraded days), carried through, never dropped.
- **Meta-labeling hook is structure only:** the schema carries `side` /
  `meta_label` and `build.apply_meta_labeling` is the reserved API; the secondary
  act/size model is deferred.
- **No new dependency** — Polars + NumPy + the existing DuckDB store. The
  per-event first-touch scan is NumPy (`argmax` on boolean masks); CUSUM is an
  explicit O(n) recursion (path-dependent, can't vectorize).

### Retrieval stays leak-free
- `labels_with_features` attaches the as-of feature row at each `t0` via the
  store's `asof_join` (`feature.ts_event ≤ t0`) — proven by test. A label may look
  forward; the features attached to it never do.

## Phase 4 — Validation framework (the truth-detector)

*Leakage-safe cross-validation + overfitting statistics. Model-agnostic
infrastructure; no real model, backtest, strategy, risk, execution, or sentiment.*

### Purge + embargo via `t1` (not a convenience — the whole point)
- Every splitter purges training samples whose `[t0, t1]` label window overlaps a
  test fold, then applies a forward embargo (`embargo_pct` of total bars). Two
  intervals overlap iff `t0_i ≤ test_end and t1_i ≥ test_start`. Uses the Phase-3
  `t1` resolution times — plain K-fold, which ignores `t1`, leaks overlapping
  labels across the boundary and reports fake skill.
- The logic lives in exactly **one** primitive (`_purge.train_indices`) shared by
  purged K-fold, CPCV and walk-forward, checked against a hand-computed reference.
  Embargo is **forward-only** (purge already covers the overlap/before side);
  double-applying it would silently shrink the train set.

### CPCV is the workhorse, not a single walk-forward
- With average label uniqueness ≈ 0.23 a single OOS path is high-variance and easy
  to fool. CPCV (`N=6, k=2` default → 15 splits, 5 paths) gives a *distribution* of
  OOS performance. Counts are exact: `C(N,k)` splits, `C(N-1,k-1)` paths, proven by
  test. Walk-forward (anchored/rolling) is kept as the realistic chronological
  complement.

### Overfitting statistics are the verdict, accuracy is not
- **PBO** via CSCV (probability the IS-best config is below the OOS median),
  **PSR** and **Deflated Sharpe** (deflating for the number of trials), and
  **minimum track-record length**. A high CV accuracy with PBO→1 or DSR≤0 means we
  found noise — treat the stats as the gate.
- **Conventions fixed once, to avoid silent bugs:** all of `SR`/benchmark/skew/
  kurtosis are per-bar (never mix an annualised Sharpe with per-bar moments);
  kurtosis is **raw** (Normal = 3, not scipy's excess default); Φ/Φ⁻¹ from
  `statistics.NormalDist` (no scipy dependency). Each formula is pinned to a
  closed-form reference value in tests (PSR at the benchmark = 0.5 exactly; Normal
  denominator = `√(1 + ½·SR²)`; DSR threshold via the Gumbel two-term Euler–
  Mascheroni form `Φ⁻¹(1 − 1/N)` and `Φ⁻¹(1 − 1/(N·e))`).

### Effective sample size reported everywhere
- Σ average-uniqueness, per fold and per path. ~11k labels ≈ ~2.7k effective; the
  raw count overstates the information and must temper trust and model complexity.

### Sample weights honoured in fit and score
- The harness passes the Phase-3 `weight` column to every `fit` (routed to a
  pipeline's final step) and weights the accuracy/return metrics. Baselines that
  need scaling use a `StandardScaler` **inside** a pipeline, so the scaler is fit
  per training fold (no test-distribution leakage).

### The teeth test is mandatory
- `tests/test_validation_teeth.py` proves leakage is caught: (a) mechanism — an
  overlapping train sample is purged while a non-overlapping one is retained; (b)
  skill collapse — on random-walk forward-return labels (no global predictability)
  with fold ≤ horizon, a KNN shows inflated OOS skill (~0.78) without purging that
  collapses to chance (~0.50) with purging + embargo. A framework that can't
  demonstrate its teeth is theater.

### Baselines must look unskilled on real data
- The shipped baselines (most-frequent dummy + standardised logistic) are
  deliberately weak. If a dummy ever looks profitable through this harness,
  something leaks — that is a stop-and-investigate signal, not a result.

### Storage / config / scope
- Evaluation runs are saved as JSON under `data/validation/<symbol>.json`
  (consumed by `observability/validation_health.py`); the matrix is assembled by
  the labeling layer's already-leak-tested `labels_with_features` (features as-of
  `t0`). Splits/embargo/CPCV groups/metrics/seed live in `config/validation.yaml`
  with a `validation_version`; fixed seed → identical results.
- **No new dependency** — NumPy + Polars + the existing scikit-learn (already a
  dep) for baselines; `statistics.NormalDist` for Φ/Φ⁻¹.
- Hyperparameter search is **deferred**; the harness must support it later (search
  inside the CV, counted as trials for the DSR) but runs none now. No real model.

---

## Phase 5 — Signal model & honest edge verdict (2026-06-08)

The whole phase answers ONE question — *does a model beat chance on the label,
after deflation, over and above beta?* — and the only acceptable failure mode is a
**fake** edge. Choices are pointed at not manufacturing one. Verdict + method:
`docs/MODEL.md`.

### Fast training matrix (fix the ~30-min load)
- `models/dataset.py` replaces the full-lake materialisation with a **pushed-down
  DuckDB read** (project only `ts_event` + the 45 features, dedup latest-ingest via
  `QUALIFY`) followed by a **polars `join_asof`** (backward, `<= t0`). ~2.5M feature
  rows assemble in **~3s** (was minutes); a version-keyed parquet cache makes a
  re-load ~0.05s. Proven **byte-identical** to Phase-4's `load_matrix` on a bounded
  window (every array equal), so it inherits the already-leak-tested guarantee.
- **Why not a single DuckDB `ASOF JOIN`?** It must sort/materialise the 2.5M-row
  right side inside the join → *minutes*. The projection + merge-asof split is the
  fast, exact equivalent. A nested `SELECT *` also defeats DuckDB's column pushdown
  — the explicit projection is load-bearing.

### Directional target (handle the timeout class)
- Labels are `{-1,0,+1}`; the vertical-timeout `0` (~3-4%) is folded into a
  **direction** by the **sign of its realised return** at `t1` (`sign_return`,
  default) — no rows dropped, weights/normalisation untouched. `drop` is the
  documented alternative. The model predicts up/down so `predict()` IS the trading
  sign and `predict_proba` is a calibrated up-probability (not thresholded here).
  Class imbalance is handled by **weights, never by altering labels**.

### A model that *can't* overfit (effective N ~2.5-2.7k vs 45 features)
- Hard regularisation in `config/models.yaml`: shallow trees (`max_depth 3`,
  `num_leaves 7`), large leaves (`min_child_samples ≥ 100`), L1+L2, sub-1
  bagging/feature fractions, and **early stopping inside each CV fold against a
  *purged* chronological tail** (no inner-val leakage). Deterministic: fixed seeds,
  single-threaded, `deterministic=True` → byte-reproducible. Goal is the *least*
  overfit model, not the highest in-sample accuracy.

### Tuning is in-CV and counts as trials
- A small **2×2×2** grid (`max_depth`/`min_child_samples`/`reg_lambda`, 8 configs)
  runs **inside** the purged K-fold; selection is on pooled OOS weighted directional
  accuracy — no config sees its own test fold. `n_trials = 8` deflates the DSR and
  the 8 configs feed PBO. Kept tiny on purpose: more trials only inflate overfit
  risk at this effective N.

### Skill is reported OVER AND ABOVE beta
- The return metric is the **excess over a perma-long benchmark**
  (`excess = (position − 1)·ret`, information-ratio style); PSR/DSR are computed on
  that excess. A perma-long model has zero excess → no edge, however much beta it
  rode. This is the netting the Phase-4 note flagged (a long dummy's PSR is pure
  beta). PBO is computed on the strategy-return matrix across the 8 configs.
- **Evaluated ONLY through `validation/`.** `evaluate_model.py` composes the
  existing splitters (`PurgedKFold`, `CombinatorialPurgedCV`) and stats
  (PBO/PSR/DSR) — it adds *zero* leakage logic, only the beta-netting and the
  trial-deflation the harness deferred to this phase.

### The VERDICT is an explicit AND
- *edge* requires **all** of: directional accuracy `> 0.52`, PBO `< 0.5`, excess
  **DSR** `> 0.5`, positive mean excess-over-long. Miss one → *no significant edge*.
  Raw accuracy is intentionally the weakest gate. Thresholds are config, the checks
  are recorded per run, and a null result is reported **without massaging** — the
  unacceptable outcome is a fake edge, not an honest "no edge yet".

### Result (full history): no significant edge on MES or MNQ
- Both lose to buy-and-hold (strategy SR < long-only SR), excess DSR ≈0, CPCV excess
  Sharpe negative on every path, PBO 0.70-0.83. MNQ's accuracy nudges past chance
  (0.527) but fails the other three gates. **This routes us to better data/features
  (the deferred macro/sentiment layer) before any strategy** — see `docs/MODEL.md`.

### Interpretability + tracking
- **SHAP** (TreeExplainer) global + local, refit on full history for explanation
  only (not a perf number). Importance is spread thin (top-feature share ~7%, no
  dominance) → consistent with the null and free of leakage smells. Drivers are
  economically plausible (vol, momentum, flow, cross-asset).
- **MLflow local file store** under `data/mlruns` (no cloud, no server). MLflow 3
  put the file store in "maintenance mode", so we set `MLFLOW_ALLOW_FILE_STORE=true`
  to keep it deliberately. `observability/model_health.py` is a read-only view over
  the saved JSON (same payload logged to MLflow).

### Storage / deps / scope
- Runs → `data/models/runs/<symbol>.json` (+ SHAP PNG); matrix cache →
  `data/models/cache/`; MLflow → `data/mlruns` (all gitignored). New deps pinned:
  `shap==0.52.0`, `mlflow==3.13.0` (+ `numba`/`llvmlite` floors so the resolver
  doesn't backtrack onto a pre-3.12 `llvmlite`). Out of scope this phase and **not**
  built: nautilus economic backtest, strategy/entry-exit, risk, execution, the
  registry/champion-challenger **promotion** pipeline, live retraining, sentiment.

---

## Phase 6 — Macro / economic-event layer + re-verdict (2026-06-08)

Price-only features were exhausted (Phase 5 null). Per the pre-registered plan we
add a **macro / economic-event** feature layer and re-run the **identical** verdict
— same labels, same framework, same gates, **inputs only changed**. Method +
comparison: `docs/MACRO.md`. The unacceptable outcome is a *fake* edge; a second
honest null is informative.

### Source: FRED/ALFRED first-print, free, key-gated (verified live)
- **First-print via `output_type=4`** ("Observations, Initial Release Only"). Each
  observation's `realtime_start` is its publication date; `event_time` = that date
  @ the standard release clock (08:30 ET data, 14:00 ET FOMC) → UTC. A later
  revision would leak the future. **Verified live**: June-2024 CPI first-print
  313.049, release 2024-07-11 (the real BLS date); first 2022 hike 0.25→0.50.
- `output_type=4` **requires an explicit ALFRED real-time window**
  (`realtime_start=2000-01-01`, `realtime_end=9999-12-31`); the default (today
  only) 400s with "no vintage dates exist for the specified real-time period".
- **`DFEDTARU` (FOMC rate) uses standard (latest) observations, not `output_type=4`**
  — a daily series has >2000 ALFRED vintages (the file-type cap), and the target
  rate is **never revised**, so latest == first-print; read only as-of a meeting.
- **Network = stdlib `urllib`** (no new dependency; `requests`/`httpx` are only
  transitive). **Key-gated**: with `OPTIONS_FRED_API_KEY` unset, ingestion no-ops
  with a clear message. The key lives in `pass` at `fred/api_key` and is bridged
  into the process env per-command — never written to `.env`/disk. New typed
  surface `Settings.fred_api_key: SecretStr | None`.

### Event set + honest exclusions
- CPI, Core CPI, PCE, Core PCE, NFP, unemployment, initial claims, GDP (advance),
  advance retail sales, PPI (final demand), and **FOMC**. Series ids + clock times
  in `config/macro.yaml`.
- **ISM manufacturing & services EXCLUDED**: ISM data was **removed from FRED on
  2016-06-24** over a licensing dispute → not available from this free source. Not
  fabricated. Documented.
- **`surprise` is null**: FRED has no consensus/expectations series; we never
  fabricate one. The outcome feature is `actual_pit − prior` (change vs the prior
  first-print), not a surprise.
- **FOMC = scheduled meetings only** (8/yr; decision dates verified against
  federalreserve.gov FOMC calendars). **Intermeeting emergency actions excluded**
  (not knowable in advance): March-2020 COVID cuts, the 2019-10-11 balance-sheet
  note; Jackson Hole symposiums are not meetings. Calendared March-2020 = 2020-03-18.

### The timing-vs-outcome leakage rule (the whole game)
- **Timing features look forward at the public schedule** (`event_time` only) —
  "minutes-to-next-CPI/FOMC" is legitimately known ahead. **Outcome features index
  strictly backward** (`event_time <= t0`). Proven by
  `tests/test_macro_features_leakage.py`: outcomes invariant to hiding future
  outcomes; timing invariant to blanking outcomes; and a **teeth** variant — a
  deliberately forward outcome look-up *is* caught by the same backward-invariance
  check, so a real leak could not slip past.
- **Caveat documented**: data-release "next" dates come from the ingested release
  series (FRED only carries past releases); fine for all but the final release
  month (where "minutes-to-next" is NaN). FOMC has an explicit forward calendar.

### Storage + integration (compute-at-`t0`, not a stored grid)
- `macro_events` lake table at `data/macro_events/date=…/` — own idempotent writer
  (key `(event_type, event_time)`), **partitioned by date only, no symbol** (macro
  context is instrument-independent; mirrors the Phase-3 labels decision to not
  extend `lake.DATASETS`).
- Macro features are computed **directly at the label `t0`s** (timing is closed-form
  from the schedule, outcomes a backward look-up over a tiny event table) and
  `hstack`ed onto the matrix — no 2.5M-row stored macro grid, and it generalises to
  the live engine. `macro_feature_version` stamps the matrix + cache key; price
  `feature_version` stays `v1` (clean controlled experiment).
- **Row count unchanged (10,827 / 11,688)**: the null/non-finite **row gate stays
  on the price features**; macro columns may be `NaN` where undefined, which
  LightGBM handles natively. `with_macro=False` reproduces the Phase-5 price-only
  matrix byte-for-byte (the parity test pins this).

### Result: still no significant edge (MES, MNQ)
- Verdict **unchanged** on both. MES price+macro accuracy dipped *below* chance
  (0.498) and excess Sharpe worsened; MNQ barely moved (PBO 0.70→0.52 borderline)
  but excess-over-beta stays negative and excess DSR ≈0. All four gates still fail.
- **SHAP**: macro features **dominate** importance (~66–69 % of total; top drivers
  `macro_chg_ppi/cpi/core_cpi`, `macro_mins_since_nfp/pce`), no single-feature
  dominance, no leakage smell. High in-sample reliance + null OOS skill is exactly
  what the framework exists to expose — and argues *against* leakage (a leak would
  inflate OOS accuracy, not depress it). New deps: **none**. Out of scope and not
  built: news/sentiment, microstructure, backtest, strategy, risk, execution.

## Phase 8 — Short-horizon (micro) labeling (2026-06-08)

Triple-barrier labels (`micro_label_version = ml1`) on the `m1` dollar bars — the
intraday sibling of the daily `v1` labels. Additive and isolated: own config
(`config/micro_labeling.yaml`), own module (`microstructure/labels.py`), own lake
(`data/micro_labels/`). The daily labels, price/macro/OFI features, and the
validation framework are untouched. No model, no verdict, no Databento spend —
that is the next prompt. See `docs/MICRO_LABELING.md`.

### Reuse vs new (don't rebuild the leak-safe primitives)
- **Reused unchanged** (imported): `labeling.events.cusum_events` (symmetric CUSUM)
  and `labeling.weights.sample_weights` (concurrency → average uniqueness → effective N).
- **New**, because dollar bars are not time-uniform and a short horizon has a
  session-close trap the daily layer never faces: a σ estimator scaled to a
  *wall-clock* horizon, a *wall-clock* 30-min vertical barrier, and the session-close
  guards. The first-touch scan mirrors the daily `label_events` argmax logic exactly.

### Label design — fixed a priori (anti-snooping)
- **σ_H = σ_bar · sqrt(vertical_seconds / EWMA(duration_s))** — causal EWM std of
  per-bar mid log returns, scaled to 30 min by the *causally-estimated* bars-per-30min
  (dollar bars carry ~equal variance per bar). Direct generalization of the daily
  `σ_bar · sqrt(barrier_horizon_bars)` to variable-duration bars. Stable on the slice.
- **±1.5σ** horizontal (mirrors daily), **30-min wall-clock vertical** (first bar with
  `ts_event ≥ t0+30min`, not a bar count), **CUSUM `cusum_mult = 1.0`** on mid returns
  (independent of OFI, so labels aren't circular with the signal under test).
- **Session-close handling (the key guard).** Everything is computed per
  `(contract_id, ET-date)` block → a label can never read the next session. Plus a
  final-30-min event exclusion and a hard cap at the 16:00-ET close
  (`barrier_touched = "close"`). RTH boundary reused from `microstructure.yaml`, not
  redefined.

### QA on the 5-RTH-day slice (2026-05-18→22) — sizes the next data pull
- ES: 109 events → 101 labels, balance +/−/0 = 0.13/0.09/0.78, hold ≈30 min,
  avg uniqueness 0.60, **effective N 60.8 (~12.2/RTH day)**.
- NQ: 95 events → 89 labels, balance 0.11/0.11/0.78, avg uniqueness 0.67,
  **effective N 59.3 (~11.9/RTH day)**.
- **Honest notes**: labels skew ~78 % to the timeout (`0`) class (±1.5σ is wide vs the
  typical post-event 30-min move) — reported, **not** tuned away. `resolved_at_close`
  is 0 on this liquid slice (the cap never binds). At ~12 effective labels/RTH day,
  reaching ~1,000 effective labels/symbol implies **~80 RTH days (~4 months)** of
  MBP-1 bars — the number that should size the Prompt 9 pull.

### Verification
- Leakage teeth (invariant past `t1`; a peek-one-past value flips → teeth; PIT
  truncation reproduces, inside-window truncation changes), session-boundary
  (no label crosses the close, final-30-min excluded, next-session perturbation
  leaves this session byte-identical), σ causality, determinism + price-scale
  invariance. Full `pytest` green, ruff/format/ty clean, QA logged to MLflow
  (`micro-labels`). New deps: **none**.

## Phase 18 — Bounded GDELT historical backfill design (2026-06-09)

**Goal:** measure whether enough free, point-in-time-correct sentiment history exists
to justify a Phase 19 model verdict — coverage measurement only, against gates
pre-registered before any data arrived (see `docs/SENTIMENT.md`). No model, no
strategy, no paid data (`OPTIONS_DATABENTO_SPEND_OK` stayed unset throughout).

### Backfill orchestrator (`sentiment/backfill.py`) — key decisions
- **One UTC calendar day per topic per slice**, from a pure, deterministic slicer.
  GDELT ArtList returns ≤250 records with **no pagination**, so a slice returning
  exactly 250 is *truncated*: it is **bisected** and both halves re-fetched,
  recursively, until halves would drop below **1 hour**; a floor slice still
  returning 250 is recorded `truncated: true` and left incomplete (disclosed, not
  hidden). Truncation barely affects coverage (has_any needs one event), only depth.
- **Archive honesty.** GDELT officially supports only ~the last 3 months
  (`ARCHIVE_SUPPORTED_DAYS = 92`). Every slice is classified
  `supported`/`unsupported_archive` at plan time; both are attempted, but the
  **supported region runs first** so a request-capped run spends its budget where
  the pre-registered gates are evaluated, and coverage is reported per region.
- **Pacing/backoff as protocol facts, not tunables**: ≥5 s + 0–1 s jitter between
  requests (GDELT enforces ~1 req/5 s per IP); on HTTP 429 exponential backoff
  10 s → doubling, capped 120 s, honoring `Retry-After`, ≤5 attempts per slice, then
  `failed: rate_limited` and the run continues. Clock/sleep/RNG/fetcher injectable —
  every test runs offline with zero real sleeps.
- **Resumable JSON manifest** under `data/sentiment_backfill/` (gitignored, atomic
  writes): per-slice window, HTTP status, records returned/written, truncated/
  bisected/failed flags, timestamps, plus per-run counters. On `--resume`, completed
  slices are skipped, failed slices retried, and pending bisection children
  re-derived — an interrupt loses nothing. The manifest is the audit trail the
  coverage report cites.
- **Fail-closed hard caps** in `config/sentiment.yaml` (`backfill.max_requests: 2500`,
  `backfill.max_wall_clock_minutes: 240`); hitting either stops cleanly (exit 3)
  with the checkpoint intact. `--plan` prints the full request plan with zero
  network; `--probe` runs exactly one slice as a live canary. All real fetches
  require `--allow-network` and route through the existing fail-closed
  `external_data_policy` (free_no_auth only). **No second GDELT client** — the
  existing `gdelt.py` builder/parser/lake are reused.
- **Topic label ≠ query text.** Several stable topic labels (`ai_capex`,
  `megacap_tech`, `risk_off`) are not usable as literal GDELT search tokens, so
  `backfill.topic_queries` maps each label to the query text actually sent (keys
  validated ⊆ `query_topics`; the s1 `query_topic` event attribute remains the
  label). Disclosed in `docs/SENTIMENT.md`.
- **Topic attribution undercount (disclosed, unchanged).** `content_hash` excludes
  `query_topic`, so an article matched by several topic queries persists once with
  first-write-wins attribution — by-topic breakdowns undercount multi-topic
  articles. The schema was deliberately NOT changed (s1 stability).

### FinBERT provisioning + batch scoring
- `scripts/download_finbert.py` is the **single deliberate download path** (prints
  model/destination/size, then resolves the snapshot revision hash). The scorer
  keeps `local_files_only=True` and fails with instructions — scoring can never
  trigger a download. `FinbertScorer.score_batch` adds batched
  `torch.inference_mode()` scoring (titles ≪ 512 tokens), and the resolved revision
  hash is stamped as `model_version_or_hash` so a future re-score with different
  weights is distinguishable. `sentiment/score_backfill.py` scores unscored raw
  rows idempotently on `(content_hash, model_name)` in deterministic order.

### Coverage split
- `sentiment/coverage.py` gained `--archive-cutoff`: the report splits per-label
  coverage into `supported` vs `unsupported_archive` regions so the 3-month archive
  limit's effect is visible instead of silently pooled. Tested on fixtures.

### Verification
- 28 new offline tests (slicer determinism/UTC boundaries, bisection floor, backoff
  sequence + retry cap with injected clock, checkpoint resume incl. re-derived
  bisection children, hard caps, archive classification, plan-mode zero-network
  guarantee, scoring idempotency, coverage split). Full `pytest` green (397),
  ruff format/check + ty clean, zero `# type: ignore`. New deps: **none**
  (`huggingface_hub` was already a transformers dependency).

### Execution + coverage verdict (2026-06-13)
- The detached chain ran to the 240-min wall-clock cap (clean stop, checkpoint intact).
  Manifest: 1,172 slices (1,100 ok / 72 rate-limited), 839 supported / 333 unsupported;
  185,533 records returned → **145,654 written** after `content_hash` dedup; 396 truncated;
  5,623 requests / 4,537 HTTP 429s. The supported region (where the gates live) was covered;
  the full plan was not (capped on breadth).
- **Local** FinBERT (`ProsusAI/finbert`, rev `4556d130…`, CUDA, `local_files_only`) scored
  140,764 unscored rows in 54.9 s, idempotent on `(content_hash, model_name)`; lake now holds
  145,629 scored rows (29 degraded). `network_used=false`; `OPTIONS_DATABENTO_SPEND_OK`
  unset throughout — **zero spend**.
- **Pre-registered gates** (fixed 2026-06-10, pre-data; transcribed verbatim into
  `docs/SENTIMENT.md`), evaluated over the supported archive region (`t0` ≥ 2026-03-10),
  pooled ES+NQ: **G1** sent_1d has_any 98.3% (≥60) · **G2** sent_240m 98.3% (≥35) · **G3**
  pooled sent_1d has_any 2,131 (≥1,400) — **ALL PASS**.
- **Verdict: Phase 19 (sentiment micro-model A/B edge verdict) is authorized.** Coverage ≠
  edge — Phase 19 must still clear the unchanged edge bar; an honest null remains acceptable.
  No model/strategy/backtest/risk/execution/live trading was run or authorized by this phase.
  Coverage JSONs saved under `data/sentiment_backfill/coverage_{pooled,ES,NQ}.json` (gitignored).

## Phase 19 — pre-registration committed before any modeling (2026-06-13)
- The full Phase 19 contract — the A/B arms (`mm1` baseline vs `mm2` = `mm1` + the `s2`
  sentiment block), the frozen row set (supported region `t0 ≥ 2026-03-10`, per symbol),
  every inherited `mm1` parameter, the six **unchanged** verdict gates, the attribution
  logic, the decision rule, and the anti-snooping commitments — is frozen verbatim in
  **`docs/PHASE19_PREREGISTRATION.md`** (single source of truth; gates not restated here).
- It was committed **before any Phase 19 model was trained and before any data reached a
  model** — the commit date of that file is the pre-registration timestamp. This is the
  deliberate corrective to the **Phase 18 provenance gap** (Phase 18’s gates were pre-
  registered upstream but only transcribed into a committed doc on the same date the data
  was scored, so the repo alone could not prove the goalposts predated the data). Docs-only;
  no model, no data, no network; `OPTIONS_DATABENTO_SPEND_OK` unset.

## Phase 19 — sentiment micro-model A/B edge verdict (2026-06-13)
- Ran the pre-registered A/B exactly as frozen: an **opt-in** `with_sentiment` block on the
  micro-model dataset (Phase-10 `with_ta` pattern — additive, default off, baseline byte-
  identical; the row gate stays on the m1 features so both arms admit the identical rows;
  sentiment nulls KEPT, never imputed). Baseline (`mm1`, OFI-only) and treatment (`mm2` =
  `mm1` + the 80 `s2` `sent_*` features) run through the **unchanged** search / fold-local
  weighting / CPCV / PBO / DSR and the **six unchanged gates** read from
  `config/micro_model.yaml`. New: `config/phase19.yaml` + `phase19_config.py` (A/B scaffolding
  only — no gates/params duplicated), `microstructure_model/phase19_ab.py` (orchestrator +
  pure attribution/decision), `observability/phase19_ab_health.py`. Reads only local lakes —
  no Databento/IBKR/network; `OPTIONS_DATABENTO_SPEND_OK` unset; zero spend.
- **Verdict: no significant edge — both symbols. Sentiment is the fifth honest null.**
  Window `t0 ∈ [2026-03-10, 2026-06-06]`, per symbol (never pooled); rows ES 1,132 / NQ 1,036,
  effective N 735.5 / 696.3. Treatment ES failed 1/6 (only PBO 0.734 — a selection overfit;
  gross DSR 0.84 + CPCV median +0.010 did not survive PBO), NQ failed 3/6 (gross DSR 0.081,
  macro F1 0.177, CPCV median −0.034). Attribution `null` on both. SHAP shows the treatment
  arm leaned heavily on sentiment (`sent_1d_topic_rates_mean_score` #1 on both) — used, not
  ignored, and still no OOS edge. No model promoted; no strategy/backtest/risk/execution/live
  trading authorized. Remaining forks: MBP-10 (paid, blocked) or a horizon/regime redesign —
  no sentiment re-tuning. Method + numbers: `docs/PHASE19_AB.md`; verdict table updated in
  `docs/RESEARCH_VERDICTS.md`. New tests: `tests/test_phase19_ab.py` (11).

## Phase 20 — pre-registration committed before any modeling (2026-06-13)
- The full Phase 20 **meta-labeling** contract — the fixed causal primary side rule
  (`sign(ofi_top)` at `t0`), the binary meta-label, the meta-model and its feature set, the
  fixed decision threshold (τ = 0.5), the frozen row set/window, every inherited `mm1`
  parameter, the verdict gates (the five inherited unchanged + the one disclosed binary
  meta-skill substitute for the 3-class F1 gate), the attribution logic, the decision rule,
  and the anti-snooping commitments — is frozen verbatim in
  **`docs/PHASE20_PREREGISTRATION.md`** (single source of truth; gates not restated here).
- Committed **before any Phase 20 model was trained and before any data reached a model** — the
  commit date of that file is the pre-registration timestamp — following the Phase 19
  pre-registration discipline (`docs/PHASE19_PREREGISTRATION.md`). Docs-only; no model, no
  data, no network; `OPTIONS_DATABENTO_SPEND_OK` unset.

## Phase 20 — meta-labeling edge verdict (2026-06-13)
- Ran the pre-registered meta-labeling A/B exactly as frozen. A fixed, deterministic, causal
  **primary** picks the side (`sign(ofi_top)` as-of `t0` — not a fitted model ⇒ no
  primary-model leakage, no nested OOF), and a **binary LightGBM meta-model** over the m1 OFI
  block + the `s2` sentiment block + `|ofi_top|` predicts `P(meta_label = 1)`; act when
  `P > τ` (τ = 0.5, fixed). The meta-model is the SOLE fitted component and runs through the
  **unchanged** purged K-fold / CPCV / PBO / trial-deflated DSR / fold-local class weighting.
  New code: `config/phase20.yaml` + `phase20_config.py` (NEW knobs only — τ, the meta-skill
  floor, the window, the primary feature; no gates/params duplicated), `meta_labeling.py`
  (pure primary-side/meta-label/gating), `meta_lgbm.py` (binary `RegularizedLGBMBinary` +
  `fit_meta_fold`, reusing the mm1 weighting primitives; mm1 left byte-identical),
  `microstructure_model/phase20_meta.py` (orchestrator + meta search/eval + B0 reference +
  pure attribution/decision), `observability/phase20_meta_health.py`. Five inherited mm1
  gross gates applied unchanged; the 3-class macro-F1 gate (undefined for a binary gate)
  replaced by the disclosed **meta-skill** gate (OOF balanced accuracy ≥ 0.52 AND acted-on
  hit-rate > always-act hit-rate). Reads only local lakes — no Databento/IBKR/network;
  `OPTIONS_DATABENTO_SPEND_OK` unset; zero spend.
- **Verdict: no significant edge — both symbols. Meta-labeling is the sixth honest null.**
  Full window `t0 ∈ [2026-01-26, 2026-06-06]`, per symbol; meta-set ES 1,688 (effN 1,089.3) /
  NQ 1,519 (1,023.1). Arm M: ES fails 2/6 (gross DSR 0.060, CPCV median −0.010), NQ fails 4/6
  (PBO 0.722, gross DSR 0.247, mean gross −1.2e-5, CPCV median −0.027). Notably the
  **meta-skill gate PASSED on both** (acted-hit > always-hit: ES 0.122→0.134, NQ 0.103→0.130;
  balAcc 0.52/0.57) — the gate *did* add precision, just far too little to survive deflation.
  B0 (always-act reference) failed both, as expected. SHAP shows the gate leaned ≈70–75% on
  the `s2` block (used, not ignored). No model promoted; no strategy/backtest/risk/execution/
  live trading authorized. Because meta-labeling is the canonical remedy for this regime, the
  null points the binding constraint at sample size / edge existence at this horizon, not
  model framing — favouring the data-changing forks (MBP-10, paid/blocked; or horizon/regime
  redesign). Method + numbers: `docs/PHASE20_META.md`; verdict table updated in
  `docs/RESEARCH_VERDICTS.md`. New tests: `tests/test_phase20_meta.py` (22).
