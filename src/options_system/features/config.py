"""Typed, declarative feature configuration.

Loads ``config/features.yaml`` into a validated :class:`FeatureConfig`. Feature
families, windows (in bars = minutes), and parameters live here — never as magic
numbers in :mod:`options_system.features.compute`. The ``feature_version`` string
is stamped onto every emitted feature row so stored tables are self-describing.

Windows are expressed in **bars** (the base series is 1-minute, so 1 bar = 1
minute). All windows are *trailing* by construction in the engine; this config
only declares sizes, never look-ahead.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

# config/ lives at the repo root (see config/__init__.py); features.yaml sits beside it.
_DEFAULT_PATH = Path(__file__).resolve().parents[3] / "config" / "features.yaml"


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Windows(_Base):
    short: list[int]
    session: list[int]
    multiday: list[int]


class ReturnsCfg(_Base):
    horizons: list[int]


class MomentumCfg(_Base):
    ema_windows: list[int]
    slope_lookback: int = Field(gt=0)
    macd: tuple[int, int, int]  # fast, slow, signal
    roc_windows: list[int]
    adx_window: int = Field(gt=0)

    @field_validator("macd")
    @classmethod
    def _macd_ordered(cls, v: tuple[int, int, int]) -> tuple[int, int, int]:
        fast, slow, signal = v
        if not (0 < fast < slow) or signal <= 0:
            raise ValueError(f"macd must be (fast<slow, signal>0); got {v}")
        return v


class MeanReversionCfg(_Base):
    rsi_window: int = Field(gt=1)
    bb_window: int = Field(gt=1)
    bb_std: float = Field(gt=0)
    zscore_windows: list[int]


class VolatilityCfg(_Base):
    rv_windows: list[int]
    atr_window: int = Field(gt=0)
    parkinson_window: int = Field(gt=1)
    gk_window: int = Field(gt=1)
    regime_window: int = Field(gt=1)
    regime_baseline: int = Field(gt=1)


class VolumeCfg(_Base):
    rvol_windows: list[int]
    tod_baseline_days: int = Field(gt=0)
    zscore_window: int = Field(gt=1)
    obv_window: int = Field(gt=1)


class CrossAssetCfg(_Base):
    pair: list[str]
    ratio_zscore_window: int = Field(gt=1)
    corr_window: int = Field(gt=1)

    @field_validator("pair")
    @classmethod
    def _pair_of_two(cls, v: list[str]) -> list[str]:
        if len(v) != 2 or v[0] == v[1]:
            raise ValueError(f"cross_asset.pair must be two distinct symbols; got {v}")
        return v


class SessionCfg(_Base):
    tz: str
    rth_open_min: int = Field(ge=0, lt=1440)
    rth_close_min: int = Field(ge=0, le=1440)
    session_roll_hour_et: int = Field(ge=0, lt=24)


class NewsHookCfg(_Base):
    # Placeholder only — no news/macro data is ingested yet. The engine builds
    # nothing while this is disabled; the seat exists so the schema is forward-ready.
    enabled: bool = False


class FeatureConfig(_Base):
    """Validated feature configuration (one object, loaded once, shared)."""

    feature_version: str
    windows: Windows
    returns: ReturnsCfg
    momentum: MomentumCfg
    mean_reversion: MeanReversionCfg
    volatility: VolatilityCfg
    volume: VolumeCfg
    cross_asset: CrossAssetCfg
    session: SessionCfg
    degraded_days: list[date]
    news: NewsHookCfg = NewsHookCfg()

    @classmethod
    def load(cls, path: str | Path | None = None) -> FeatureConfig:
        """Load and validate the feature config from YAML (defaults to config/features.yaml)."""
        p = Path(path) if path is not None else _DEFAULT_PATH
        with p.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        return cls.model_validate(data)

    def to_dict(self) -> dict:
        """Round-trippable plain dict (dates as ISO strings)."""
        return self.model_dump(mode="json")

    def degraded_day_set(self) -> set[date]:
        return set(self.degraded_days)

    def max_window(self) -> int:
        """Largest trailing window anywhere in the config (drives the warmup flag)."""
        candidates: list[int] = []
        candidates += self.windows.short + self.windows.session + self.windows.multiday
        candidates += self.returns.horizons
        candidates += self.momentum.ema_windows + list(self.momentum.macd)
        candidates += self.momentum.roc_windows + [self.momentum.adx_window]
        candidates += [self.mean_reversion.rsi_window, self.mean_reversion.bb_window]
        candidates += self.mean_reversion.zscore_windows
        candidates += self.volatility.rv_windows
        candidates += [
            self.volatility.atr_window,
            self.volatility.parkinson_window,
            self.volatility.gk_window,
            self.volatility.regime_window,
            self.volatility.regime_baseline,
        ]
        candidates += self.volume.rvol_windows
        candidates += [self.volume.zscore_window, self.volume.obv_window]
        candidates += [self.cross_asset.ratio_zscore_window, self.cross_asset.corr_window]
        return max(candidates)


if __name__ == "__main__":  # pragma: no cover - manual smoke check
    cfg = FeatureConfig.load()
    print(f"feature_version={cfg.feature_version} max_window={cfg.max_window()}")
