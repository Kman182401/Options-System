# CLAUDE.md — Options-System

## What this is
An autonomous, production-grade futures-trading system I build *with* Claude Code and run locally. Phase 1 trades CME micro futures (MES, MNQ) intraday (max ~3-day holds). Phase 2 adds options vertical spreads. The repo is named "Options-System" because options are the end goal.

## Environment
Runs locally on **CachyOS** (Arch-based Linux) with the **fish** shell. Hardware: AMD Ryzen 9800X3D, NVIDIA RTX 5070 Ti (Blackwell, 16 GB VRAM), 32 GB DDR5. Everything — engine, training, dashboard — runs on this one machine; no VPS/cloud. Always use fish-compatible commands and the Arch package manager (`pacman`/AUR). No Windows.

## Prime directive
The human must be able to understand the entire system at all times. Clarity beats cleverness. Every module is independently explainable in plain English. If a change would make the system harder to understand, flag it.

## Two brains (never mix them)
1. **Live engine** — deterministic Python. Loads a pre-trained model, runs inference, decides, risk-checks, executes. Fast and boring. **No LLM in this loop, ever** → zero per-trade AI cost.
2. **Offline learning loop** — backtest → train → gate → deploy, run on this GPU box. Where improvement happens.
Claude Code is the **builder** (interactive) and is **never** in the live loop.

## Architecture (local-first, one machine)
Data inputs (IBKR live, news/macro) → Ingestion → Features → ML Inference (LightGBM + sentiment) → Strategy decision → **Risk Manager (can veto/flatten; always rests a broker-side stop)** → Execution (nautilus_trader → ib_async → IBKR paper) → Log (DuckDB/Parquet) + Streamlit dashboard + Telegram alerts.
Learning loop: Databento history + logged live data → nautilus backtest/walk-forward → train (LightGBM, FinBERT/8B-LLM on GPU) → champion-challenger gate → model registry → deploy to live inference.

## Tech stack
Python 3.12 (uv) · ib_async · nautilus_trader · lightgbm · polars/pandas · duckdb/pyarrow · transformers+torch (FinBERT) / Ollama (optional 8B) · streamlit · python-telegram-bot · pydantic-settings · loguru · pytest · ruff.

## Rules of engagement
- Build ONE section at a time, only when prompted. Never scaffold the whole system at once.
- Each module gets a plain-English README and docstrings explaining *why*, not just *what*.
- Ask before adding any dependency or expanding scope.
- Safety: paper trading only until the human explicitly approves live. Never write real-money connection code without that approval. The Risk Manager is sacrosanct.
- Overfitting is the enemy: no strategy/model is accepted on in-sample results. Always walk-forward + realistic costs/slippage + a champion-challenger gate before anything reaches money.
- Determinism & tests: every component unit-tested; rely on nautilus for backtest=live parity.
- Keep `docs/DECISIONS.md` (decision log) and this file updated as the system evolves.
- If unsure, ask — do not guess.

## Phase roadmap
- **Phase 0 (done once bootstrapped):** project skeleton, tooling, docs, smoke tests.
- **Phase 1 (futures):** data layer → features → backtest harness → **strategy research & selection (Claude-driven)** → signal model → sentiment → risk manager → paper execution → observability → paper-trading hardening.
- **Phase 2 (later):** options vertical spreads adapter on the same engine.

## Directory map
`config/` typed config + yaml · `src/options_system/{common,data,features,macro,labeling,microstructure,sentiment,models,strategy,risk,execution,backtest,observability,validation}/` · `scripts/` smoke tests · `tests/` · `data|models|logs/` (gitignored) · `docs/` (ARCHITECTURE, DECISIONS, GLOSSARY, SETUP, MICROSTRUCTURE, research/).

## Feature layers (versioned, isolated, additive)
- `feature_version=v1` — price features (1-min bars). `macro_feature_version` — macro/event layer. `microstructure_feature_version=m1` — order-flow/OFI on **dollar bars** from CME L2 data, separate `data/micro_bars/` table; see `docs/MICROSTRUCTURE.md`.
- **Databento cost is real money, billed per byte.** Always `metadata.get_cost`/`get_billable_size` (free) BEFORE any `get_range`/`to_file`. Microstructure ingest is gated by `databento_budget_usd_cap` (estimate→cap→abort). Key in `pass` (`databento/api_key_2` live; original `databento/api_key` removed 2026-06-08 after depletion; `databento/api_key_3` reserved for next backup — `pass insert` it, no code change), never `.env`. MBP-1 (top-of-book) now; MBP-10 (multi-level OFI) is a later escalation.

## How to run
- Install: `uv sync`
- GPU check: `uv run python scripts/smoke_test_gpu.py`
- IBKR paper check (Gateway must be running): `uv run python scripts/smoke_test_ibkr.py`
- Tests/lint: `uv run pytest -q` · `uv run ruff check .`

## Git & remote
- **Remote:** `origin` → `git@github.com:Kman182401/Options-System.git` (private, SSH). Default branch `master`.
- **Commits = automatic, local.** After any code/config edit in this repo, stage + commit without asking. Only files changed in the session; never sweep unrelated/untracked WIP (e.g. in-progress `microstructure/`) into a commit unless explicitly told.
- **Pushes = approval-gated.** NEVER `git push` without Karson's explicit OK each time. Commits accumulate locally; he approves when to publish. Enforced in `.claude/settings.json` (`git push` is in `permissions.ask`).
- Never `--force`/`--amend`/`--no-verify` or rewrite shared history without permission.
- Commit trailer: `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
