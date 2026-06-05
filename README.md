# Options-System

An autonomous, production-grade **futures-trading system**, built locally with
Claude Code and run on one machine. **Phase 1** trades CME micro futures
(**MES**, **MNQ**) intraday (holds of at most ~3 days). **Phase 2** (later) adds
options **vertical spreads** on the same engine — hence the repo name. It is
**paper-trading only** until a human explicitly approves otherwise, and a
sacrosanct **Risk Manager** sits between every decision and every order, always
resting a hard stop-loss at the broker so an outage can't leave a position naked.

## Two brains (kept separate)

- **Live engine** — deterministic Python: load an approved model, run inference,
  decide, risk-check, execute. **No LLM in this loop, ever** → zero per-trade cost.
- **Offline learning loop** — on this GPU box: backtest → train → champion-
  challenger gate → deploy. Where improvement happens. Claude Code is the
  **builder** here and is **never** in the live loop.

The only thing crossing between them is a vetted model artifact. Full narrative:
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Setup (CachyOS / Linux / fish)

```fish
cd ~/Options-System
uv sync                                      # Python 3.12 env + all deps
uv run python scripts/smoke_test_gpu.py      # CUDA + RTX 5070 Ti + FinBERT
uv run python scripts/smoke_test_ibkr.py     # IBKR paper (needs IB Gateway running)
```

Copy `.env.example` → `.env` for secrets/overrides (keys are `OPTIONS_`-prefixed;
non-secret defaults live in `config/config.yaml`). Full install + IB Gateway +
GPU + Ollama steps: [`docs/SETUP.md`](docs/SETUP.md).

## Develop

```fish
uv run pytest -q          # tests
uv run ruff check .       # lint
uv run ruff format .      # format
```

## Phase roadmap

- **Phase 0 — bootstrap (this commit):** skeleton, tooling, docs, smoke tests.
- **Phase 1 — futures:** data layer → features → backtest harness → strategy
  research & selection (Claude-driven) → signal model → sentiment → risk manager
  → paper execution → observability → paper-trading hardening.
- **Phase 2 — options:** vertical-spread adapter on the same engine.

## Tech stack

Python 3.12 (uv) · `nautilus_trader` · `ib_async` · `lightgbm` · `polars`/`pandas`
· `duckdb`/`pyarrow` · `transformers`+`torch` (FinBERT) / Ollama (optional 8B) ·
`streamlit` · `python-telegram-bot` · `pydantic-settings` · `loguru` · `pytest` ·
`ruff`.

## Where to read next

- [`CLAUDE.md`](CLAUDE.md) — the project anchor (rules of engagement, prime directive).
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — data-flow narrative.
- [`docs/DECISIONS.md`](docs/DECISIONS.md) — decision log (versions, trade-offs).
- [`docs/GLOSSARY.md`](docs/GLOSSARY.md) — plain-English terms.
- [`docs/SETUP.md`](docs/SETUP.md) — install + connectivity steps.
- Each `src/options_system/<module>/README.md` — that module's future job.

> Prime directive: a human must be able to understand the entire system at all
> times. Clarity beats cleverness.
