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
