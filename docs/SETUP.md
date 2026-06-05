# Setup (CachyOS / Linux / fish)

Everything runs locally on one machine. Commands are **fish**-compatible. System
packages use `pacman` / AUR (`paru`/`yay`). No Windows, no Docker (unless asked).

---

## 1. Python environment (uv)

```fish
cd ~/Options-System
uv sync                      # creates .venv on Python 3.12 and installs everything
```

`uv` is already installed on this box. If it ever isn't:
`paru -S uv`  (or the official installer: `curl -LsSf https://astral.sh/uv/install.sh | sh`).

Run anything inside the env with `uv run …` (no manual venv activation needed).

---

## 2. GPU + sentiment check

```fish
uv run python scripts/smoke_test_gpu.py
```

Expected: `cuda.is_available(): True`, device `NVIDIA GeForce RTX 5070 Ti`,
compute capability `sm_120`, a passing GPU matmul, and a FinBERT sentiment
result for three sample sentences (first run downloads the ~440 MB model).

Notes:
- The installed torch is the **cu128** build (`2.9.1+cu128`); Blackwell `sm_120`
  kernels ship from cu128 onward. The driver (CUDA 13.3) is forward-compatible.
- Confirm the driver is healthy with `nvidia-smi` (should list the 5070 Ti).
- **Only relevant when running through the Claude Code agent sandbox:** that
  sandbox may inject a stale `CUDA_VISIBLE_DEVICES` UUID that hides the GPU.
  Your normal fish shell does **not** set it (see `~/.config/fish/conf.d/env.fish`),
  so a direct `uv run …` works. If you ever see "No CUDA GPUs are available"
  despite a working `nvidia-smi`, check `echo $CUDA_VISIBLE_DEVICES`.

---

## 3. IBKR paper connectivity

We connect **only to the paper account**. Default port is IB Gateway paper.

### 3.1 Install IB Gateway (Linux)
1. Download **IB Gateway** (the standalone, Java-based Linux build) from
   Interactive Brokers: <https://www.interactivebrokers.com/en/trading/ibgateway-stable.php>
   (the "stable" channel is fine). It's a `.sh` installer:
   ```fish
   sh ~/Downloads/ibgateway-stable-standalone-linux-x64.sh
   ```
2. Launch IB Gateway, choose **IB API** (not FIX), and log in with your
   **paper** credentials (paper usernames/accounts start with `DU`).

### 3.2 Enable the API
In IB Gateway: **Configure → Settings → API → Settings**:
- ☑ **Enable ActiveX and Socket Clients**
- **Socket port**: `4002` (IB Gateway paper). *(TWS paper would be `7497`.)*
- **Trusted IPs**: add `127.0.0.1`.
- Leave **Read-Only API** checked for now — the smoke test connects read-only
  and never places orders. (We'll revisit when execution is built, paper-only.)

### 3.3 Point the project at it
Copy the example env file and set the IBKR values (keys are `OPTIONS_`-prefixed):
```fish
cp .env.example .env
# then edit .env:
#   OPTIONS_IBKR_HOST=127.0.0.1
#   OPTIONS_IBKR_PORT=4002
#   OPTIONS_IBKR_CLIENT_ID=1
```
Non-secret defaults also live in `config/config.yaml`; `.env` overrides them.

### 3.4 Verify
With IB Gateway running and logged into paper:
```fish
uv run python scripts/smoke_test_ibkr.py
```
Expected: a connection confirmation, account summary rows, the resolved
front-month MES contract, and one recent 1-hour bar. If Gateway isn't running
the script prints this checklist and exits non-zero — that's correct.

Paper market data may be **delayed** without a subscription; delayed is fine for
Phase 0 (the script requests delayed data and historical bars).

### 3.5 Later: headless auto-login (IBC)
For unattended running, **IBC** (<https://github.com/IbcAlpha/IBC>) can auto-login
and auto-restart IB Gateway on Linux. **Do not set this up yet** — note it for the
paper-hardening step.

---

## 4. Telegram alerts (optional, later)

When observability is built you'll want a dedicated bot:
1. In Telegram, message **@BotFather** → `/newbot` → get the **bot token**.
2. Get your **chat id** (e.g. message the bot, then read
   `https://api.telegram.org/bot<token>/getUpdates`).
3. Put them in `.env` as `OPTIONS_TELEGRAM_BOT_TOKEN` and `OPTIONS_TELEGRAM_CHAT_ID`.

Use a **bot dedicated to this system** — the project intentionally does not read
any global `TELEGRAM_*` shell variable.

---

## 5. Ollama 8B sentiment (optional upgrade path, later)

FinBERT is the baseline. A local ~8B LLM is the optional upgrade for richer
sentiment. Inference stays **local and offline** — never in the live trade loop.

```fish
paru -S ollama-cuda          # GPU build for the 5070 Ti (or `ollama` for CPU)
sudo systemctl enable --now ollama
ollama pull qwen3:8b         # or: ollama pull llama3.1:8b
ollama run qwen3:8b "Classify the sentiment of: 'Profits beat estimates.'"
```

Not needed for Phase 0; documented so it's ready when we get there.

---

## 6. Day-to-day commands

```fish
uv sync                              # install / update deps
uv run pytest -q                     # tests
uv run ruff check .                  # lint
uv run ruff format .                 # format
uv run python scripts/smoke_test_gpu.py
uv run python scripts/smoke_test_ibkr.py   # needs IB Gateway running
```
