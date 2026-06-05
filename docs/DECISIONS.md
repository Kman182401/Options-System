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
