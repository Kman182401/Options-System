"""Declarative config for the Phase-19 sentiment A/B edge verdict.

Loads ``config/phase19.yaml`` into a validated :class:`Phase19Config`. Holds ONLY the
A/B scaffolding (the frozen supported-region window, the two arms, the symbols, the
MLflow experiment). Model hyperparameters, the 8-config search grid, the seed, and the
six verdict gates are read from ``config/micro_model.yaml`` (mm1) — never duplicated
here. The frozen contract is ``docs/PHASE19_PREREGISTRATION.md``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

# config/ lives at the repo root; phase19.yaml sits beside micro_model.yaml.
_DEFAULT_PATH = Path(__file__).resolve().parents[3] / "config" / "phase19.yaml"


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ArmCfg(_Base):
    """One A/B arm: a name, the sentiment toggle, and the model-version stamp."""

    name: str
    with_sentiment: bool
    model_version: str


class WindowCfg(_Base):
    """The frozen supported-region label window (``t0`` in [start, end])."""

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


class Phase19Config(_Base):
    """The Phase-19 A/B scaffolding (no gates/params — those stay in mm1)."""

    phase19_version: str
    window: WindowCfg
    symbols: list[str]
    arms: list[ArmCfg]
    sentiment_feature_version: str
    mlflow_experiment: str

    @field_validator("symbols")
    @classmethod
    def _nonempty_symbols(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("symbols must be non-empty")
        return v

    @model_validator(mode="after")
    def _two_arms_one_each(self) -> Phase19Config:
        if len(self.arms) != 2:
            raise ValueError(
                f"expected exactly 2 arms (baseline + treatment); got {len(self.arms)}"
            )
        sent = [a for a in self.arms if a.with_sentiment]
        base = [a for a in self.arms if not a.with_sentiment]
        if len(sent) != 1 or len(base) != 1:
            raise ValueError(
                "arms must be exactly one baseline (with_sentiment=false) and one treatment (true)"
            )
        if base[0].model_version == sent[0].model_version:
            raise ValueError("baseline and treatment must have distinct model_version stamps")
        return self

    @property
    def baseline(self) -> ArmCfg:
        return next(a for a in self.arms if not a.with_sentiment)

    @property
    def treatment(self) -> ArmCfg:
        return next(a for a in self.arms if a.with_sentiment)

    @classmethod
    def load(cls, path: str | Path | None = None) -> Phase19Config:
        """Load + validate the Phase-19 config from YAML (default: config/phase19.yaml)."""
        p = Path(path) if path is not None else _DEFAULT_PATH
        with p.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        return cls.model_validate(data)
