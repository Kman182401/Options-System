"""Typed application configuration.

A single ``Settings`` object is the one place the whole system reads
configuration from. Values are layered, highest priority first:

1. explicit keyword args to ``Settings(...)``
2. environment variables (all prefixed ``OPTIONS_``, e.g. ``OPTIONS_IBKR_PORT``)
3. a local ``.env`` file (gitignored — holds secrets, also ``OPTIONS_``-prefixed)
4. ``config/config.yaml`` (non-secret, human-editable defaults; keys are the
   bare field names, no prefix)
5. the field defaults defined below

The ``OPTIONS_`` prefix keeps this project's configuration isolated from
ambient global shell variables, so nothing leaks in by accident.

Secrets (API keys / tokens) are typed as ``SecretStr`` so they never print in
logs or ``model_dump()``. Nothing here *uses* the values yet — Phase 0 only
establishes the typed surface. The risk limits in particular are placeholders
with conservative defaults and are not wired to any logic.

SAFETY: ``mode`` is hard-locked to ``"paper"``. Any other value (e.g. trying to
set ``OPTIONS_MODE=live``) makes ``Settings()`` refuse to load. Live trading must never
be enabled by configuration alone — it requires an explicit, deliberate code
change that the human approves.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

# Repo root = parent of this file's directory (config/ lives at the repo root).
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    """Typed, validated configuration for the whole system."""

    model_config = SettingsConfigDict(
        # All env / .env keys are prefixed OPTIONS_ so this project's config is
        # fully isolated from ambient/global shell vars (e.g. a shared
        # TELEGRAM_BOT_TOKEN used by other systems never bleeds in here).
        env_prefix="OPTIONS_",
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        yaml_file=str(PROJECT_ROOT / "config" / "config.yaml"),
        case_sensitive=False,
        extra="ignore",
    )

    # --- Trading mode (PAPER ONLY — see SAFETY note in module docstring) ---
    mode: str = "paper"

    # --- Instrument (Phase 1: CME micro futures) ---
    instrument: str = "MES"

    # --- Local store / artifact paths (absolute, anchored to the repo root) ---
    data_dir: Path = PROJECT_ROOT / "data"
    models_dir: Path = PROJECT_ROOT / "models"
    logs_dir: Path = PROJECT_ROOT / "logs"

    # --- IBKR connection (paper Gateway/TWS). Defaults = IB Gateway paper. ---
    ibkr_host: str = "127.0.0.1"
    ibkr_port: int = 4002  # IB Gateway paper API; TWS paper is 7497
    ibkr_client_id: int = 1

    # --- IBKR paper login for IBC auto-login (loaded from .env; never committed). ---
    # Leave unset to log in to IB Gateway by hand instead. The username is not a
    # secret; the password is a SecretStr and is only ever written to a tmpfs file
    # at launch (see scripts/start_gateway.fish), never to disk or git.
    ibkr_username: str | None = None
    ibkr_password: SecretStr | None = None

    # --- Data layer ---
    record_symbols: list[str] = ["MES", "MNQ"]  # CME micro futures to record live
    recorder_client_id: int = 11  # distinct from smoke-test/engine client ids
    recorder_flush_seconds: int = 30  # how often the recorder flushes buffers to Parquet
    roll_calendar_days: int = 5  # calendar fallback: roll this many days before expiry
    continuous_adjustment: str = "ratio"  # back-adjustment: "ratio" or "panama"

    # --- Secrets (loaded from .env; never committed) ---
    databento_api_key: SecretStr | None = None
    finnhub_api_key: SecretStr | None = None
    telegram_bot_token: SecretStr | None = None
    telegram_chat_id: str | None = None  # an ID, not a secret, but optional

    # --- Risk limits (PLACEHOLDERS — typed surface only, not wired to logic) ---
    max_daily_loss_usd: float = Field(default=500.0, gt=0)
    risk_per_trade_usd: float = Field(default=100.0, gt=0)
    max_position_size: int = Field(default=1, gt=0)
    max_concurrent_positions: int = Field(default=1, gt=0)
    max_account_drawdown_pct: float = Field(default=0.10, gt=0, le=1)

    @field_validator("mode")
    @classmethod
    def _enforce_paper_only(cls, v: str) -> str:
        """Reject any mode other than paper. Live trading is forbidden by config."""
        normalized = v.strip().lower()
        if normalized != "paper":
            raise ValueError(
                f"mode={v!r} is not permitted. Only 'paper' is allowed in this phase. "
                "Live trading must never be enabled via configuration — it requires an "
                "explicit, human-approved code change."
            )
        return normalized

    @field_validator("continuous_adjustment")
    @classmethod
    def _validate_adjustment(cls, v: str) -> str:
        normalized = v.strip().lower()
        if normalized not in {"ratio", "panama"}:
            raise ValueError(f"continuous_adjustment={v!r} invalid; use 'ratio' or 'panama'.")
        return normalized

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Insert config.yaml as a low-priority source (below env/.env)."""
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            YamlConfigSettingsSource(settings_cls),
            file_secret_settings,
        )


if __name__ == "__main__":  # pragma: no cover - manual smoke check
    print(Settings().model_dump())
