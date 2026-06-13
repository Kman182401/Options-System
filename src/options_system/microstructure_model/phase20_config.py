"""Declarative config for the Phase-20 meta-labeling edge verdict.

Loads ``config/phase20.yaml`` into a validated :class:`Phase20Config`. Holds ONLY the
NEW pre-registered knobs (the full-window row set, the supported-region cutoff, the
symbols, the fixed primary side feature, the fixed decision threshold ``tau``, the
meta-skill balanced-accuracy floor, the MLflow experiment). The model hyperparameters,
the 8-config search grid, the seed, and the FIVE inherited verdict gates are read from
``config/micro_model.yaml`` (mm1) — never duplicated here. The frozen contract is
``docs/PHASE20_PREREGISTRATION.md``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# config/ lives at the repo root; phase20.yaml sits beside micro_model.yaml.
_DEFAULT_PATH = Path(__file__).resolve().parents[3] / "config" / "phase20.yaml"


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class WindowCfg(_Base):
    """The frozen label window (``t0`` in [start, end])."""

    start: str
    end: str

    @field_validator("start", "end")
    @classmethod
    def _valid_date(cls, v: str) -> str:
        datetime.fromisoformat(v)  # raises ValueError if not an ISO date
        return v

    @model_validator(mode="after")
    def _ordered(self) -> WindowCfg:
        if datetime.fromisoformat(self.start) >= datetime.fromisoformat(self.end):
            raise ValueError(f"window start {self.start} must precede end {self.end}")
        return self

    def start_dt(self) -> datetime:
        return datetime.fromisoformat(self.start).replace(tzinfo=UTC)

    def end_dt(self) -> datetime:
        return datetime.fromisoformat(self.end).replace(tzinfo=UTC)


class Phase20Config(_Base):
    """The Phase-20 meta-labeling scaffolding + NEW knobs (no inherited gates/params)."""

    phase20_version: str
    window: WindowCfg
    supported_region_start: str
    symbols: list[str]
    primary_feature: str
    decision_threshold: float = Field(gt=0.0, lt=1.0)
    meta_skill_min_balanced_accuracy: float = Field(ge=0.5, le=1.0)
    sentiment_feature_version: str
    mlflow_experiment: str

    @field_validator("supported_region_start")
    @classmethod
    def _valid_date(cls, v: str) -> str:
        datetime.fromisoformat(v)
        return v

    @field_validator("symbols")
    @classmethod
    def _nonempty_symbols(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("symbols must be non-empty")
        return v

    @model_validator(mode="after")
    def _supported_region_within_window(self) -> Phase20Config:
        srs = datetime.fromisoformat(self.supported_region_start)
        if not (
            datetime.fromisoformat(self.window.start)
            <= srs
            <= datetime.fromisoformat(self.window.end)
        ):
            raise ValueError(
                f"supported_region_start {self.supported_region_start} must fall within the "
                f"window [{self.window.start}, {self.window.end}]"
            )
        return self

    def supported_region_start_dt(self) -> datetime:
        return datetime.fromisoformat(self.supported_region_start).replace(tzinfo=UTC)

    @classmethod
    def load(cls, path: str | Path | None = None) -> Phase20Config:
        """Load + validate the Phase-20 config from YAML (default: config/phase20.yaml)."""
        p = Path(path) if path is not None else _DEFAULT_PATH
        with p.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        return cls.model_validate(data)
