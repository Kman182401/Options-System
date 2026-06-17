"""Typed, declarative volatility-forecast configuration (Phase 21).

Loads ``config/volatility.yaml`` into a validated :class:`VolatilityConfig`. Every knob the
frozen contract (``docs/PHASE21_PREREGISTRATION.md``) fixes a priori lives here — the RV
estimator params, the horizon set, the HAR lags, the single fixed LightGBM regressor config,
the anchored walk-forward window, the regime rule, and the Diebold-Mariano level — never as
magic numbers in code. Mirrors the frozen, ``extra='forbid'`` pydantic pattern used across the
project. ``volatility_version`` is stamped onto every emitted run.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

_DEFAULT_PATH = Path(__file__).resolve().parents[3] / "config" / "volatility.yaml"

# The two allowed training objectives (the contract's QLIKE + its disclosed fallback).
_KNOWN_OBJECTIVES = frozenset({"qlike", "l2_log_rv"})

# LightGBM regressor constructor knobs (the subset the wrapper takes; seed handled separately).
_LGBM_ESTIMATOR_PARAMS = (
    "n_estimators",
    "learning_rate",
    "max_depth",
    "num_leaves",
    "min_child_samples",
    "subsample",
    "subsample_freq",
    "colsample_bytree",
    "reg_alpha",
    "reg_lambda",
    "max_bin",
)


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RvCfg(_Base):
    sampling_minutes: int = Field(gt=0)
    n_subsample_grids: int = Field(gt=0)
    min_5min_returns_per_session: int = Field(ge=0)

    @field_validator("n_subsample_grids")
    @classmethod
    def _grids_le_sampling(cls, v: int, info: Any) -> int:  # noqa: ANN401 - pydantic hook
        # the offset grids have origins 0..sampling_minutes-1; can't have more grids than that.
        sm = info.data.get("sampling_minutes")
        if sm is not None and v > sm:
            raise ValueError(f"n_subsample_grids ({v}) cannot exceed sampling_minutes ({sm})")
        return v


class HorizonsCfg(_Base):
    primary: int = Field(gt=0)
    diagnostic: list[int]

    @field_validator("diagnostic")
    @classmethod
    def _positive(cls, v: list[int]) -> list[int]:
        if any(h <= 0 for h in v):
            raise ValueError(f"diagnostic horizons must all be > 0, got {v}")
        return v

    @property
    def all(self) -> list[int]:
        """Primary + diagnostics, deduped, primary first (deterministic order)."""
        out = [self.primary]
        for h in self.diagnostic:
            if h not in out:
                out.append(h)
        return out


class HarCfg(_Base):
    lags: list[int]

    @field_validator("lags")
    @classmethod
    def _three_increasing(cls, v: list[int]) -> list[int]:
        if v != [1, 5, 22]:
            raise ValueError(f"har.lags must be the Corsi daily/weekly/monthly [1, 5, 22], got {v}")
        return v


class LgbmCfg(_Base):
    objective: str
    n_estimators: int = Field(gt=0)
    learning_rate: float = Field(gt=0.0)
    max_depth: int = Field(gt=0)
    num_leaves: int = Field(gt=1)
    min_child_samples: int = Field(gt=0)
    subsample: float = Field(gt=0.0, le=1.0)
    subsample_freq: int = Field(ge=0)
    colsample_bytree: float = Field(gt=0.0, le=1.0)
    reg_alpha: float = Field(ge=0.0)
    reg_lambda: float = Field(ge=0.0)
    max_bin: int = Field(gt=1)
    early_stopping_rounds: int = Field(gt=0)
    inner_val_fraction: float = Field(gt=0.0, lt=1.0)

    @field_validator("objective")
    @classmethod
    def _known_objective(cls, v: str) -> str:
        s = v.strip().lower()
        if s not in _KNOWN_OBJECTIVES:
            raise ValueError(
                f"lgbm.objective={v!r} invalid; use one of {sorted(_KNOWN_OBJECTIVES)}"
            )
        return s

    def estimator_params(self) -> dict[str, Any]:
        """The LGBMRegressor constructor hyperparameters (regularization knobs only)."""
        return {k: getattr(self, k) for k in _LGBM_ESTIMATOR_PARAMS}


class WalkForwardCfg(_Base):
    oos_start: str
    n_steps: int = Field(gt=1)

    @field_validator("oos_start")
    @classmethod
    def _valid_date(cls, v: str) -> str:
        datetime.fromisoformat(v)
        return v

    def oos_start_dt(self) -> datetime:
        return datetime.fromisoformat(self.oos_start).replace(tzinfo=UTC)


class RegimeCfg(_Base):
    trailing_days: int = Field(gt=0)


class DmCfg(_Base):
    alpha: float = Field(gt=0.0, lt=0.5)


class FeaturesCfg(_Base):
    with_price: bool
    with_macro: bool
    with_sentiment: bool
    # Phase-22 opt-in blocks (default OFF -> the frozen Phase-21 feature set is unchanged):
    with_marketdata: bool = False  # x1 daily VIX/VXN + cross-asset features
    with_gkg: bool = False  # s3 multi-year GKG news-tone daily aggregates


class VolatilityConfig(_Base):
    """Validated volatility-forecast configuration (loaded once, shared)."""

    volatility_version: str
    seed: int = Field(ge=0)
    symbols: list[str]
    rv: RvCfg
    horizons: HorizonsCfg
    har: HarCfg
    lgbm: LgbmCfg
    walk_forward: WalkForwardCfg
    regime: RegimeCfg
    dm: DmCfg
    features: FeaturesCfg

    @field_validator("symbols")
    @classmethod
    def _nonempty_symbols(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("symbols must be non-empty")
        return v

    @classmethod
    def load(cls, path: str | Path | None = None) -> VolatilityConfig:
        """Load + validate the volatility config from YAML (default: config/volatility.yaml)."""
        p = Path(path) if path is not None else _DEFAULT_PATH
        with p.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        return cls.model_validate(data)

    def to_dict(self) -> dict:
        return self.model_dump(mode="json")
