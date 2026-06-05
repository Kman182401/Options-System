"""Typed, declarative labeling configuration.

Loads ``config/labeling.yaml`` into a validated :class:`LabelConfig`. Barrier
multiples, the volatility estimator, the vertical-barrier length, the
event-sampling method/threshold, the roll-handling choice and the weighting
scheme all live here — never as magic numbers in the generators. The
``label_version`` string is stamped onto every emitted label row so stored
tables are self-describing.

Mirrors the pattern of :mod:`options_system.features.config`: a frozen,
``extra='forbid'`` pydantic tree loaded once and shared. Windows / horizons are
in **bars** (the base series is 1-minute, so 1 bar = 1 minute).
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

# config/ lives at the repo root (see config/__init__.py); labeling.yaml sits beside it.
_DEFAULT_PATH = Path(__file__).resolve().parents[3] / "config" / "labeling.yaml"


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class VolatilityCfg(_Base):
    ewm_span: int = Field(gt=1)
    min_samples: int = Field(gt=1)
    barrier_horizon_bars: int = Field(gt=0)


class BarriersCfg(_Base):
    pt_mult: float = Field(gt=0)
    sl_mult: float = Field(gt=0)
    max_hold_bars: int = Field(gt=0)
    vertical_label_sign: bool = False


class EventsCfg(_Base):
    method: str
    cusum_mult: float = Field(gt=0)
    grid_step_bars: int = Field(gt=0)

    @field_validator("method")
    @classmethod
    def _known_method(cls, v: str) -> str:
        m = v.strip().lower()
        if m not in {"cusum", "grid"}:
            raise ValueError(f"events.method={v!r} invalid; use 'cusum' or 'grid'")
        return m


class RollCfg(_Base):
    handling: str

    @field_validator("handling")
    @classmethod
    def _known_handling(cls, v: str) -> str:
        h = v.strip().lower()
        if h not in {"adjust", "close"}:
            raise ValueError(f"roll.handling={v!r} invalid; use 'adjust' or 'close'")
        return h


class WeightsCfg(_Base):
    scheme: str
    time_decay: float = Field(ge=-1.0, le=1.0)

    @field_validator("scheme")
    @classmethod
    def _known_scheme(cls, v: str) -> str:
        s = v.strip().lower()
        if s not in {"uniqueness", "uniqueness_return"}:
            raise ValueError(
                f"weights.scheme={v!r} invalid; use 'uniqueness' or 'uniqueness_return'"
            )
        return s


class LabelConfig(_Base):
    """Validated labeling configuration (one object, loaded once, shared)."""

    label_version: str
    volatility: VolatilityCfg
    barriers: BarriersCfg
    events: EventsCfg
    roll: RollCfg
    weights: WeightsCfg

    @classmethod
    def load(cls, path: str | Path | None = None) -> LabelConfig:
        """Load and validate the labeling config from YAML (defaults to config/labeling.yaml)."""
        p = Path(path) if path is not None else _DEFAULT_PATH
        with p.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        return cls.model_validate(data)

    def to_dict(self) -> dict:
        """Round-trippable plain dict."""
        return self.model_dump(mode="json")


if __name__ == "__main__":  # pragma: no cover - manual smoke check
    cfg = LabelConfig.load()
    print(
        f"label_version={cfg.label_version} barriers={cfg.barriers.pt_mult}/{cfg.barriers.sl_mult}"
    )
