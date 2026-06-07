"""Typed, declarative validation configuration.

Loads ``config/validation.yaml`` into a validated :class:`ValidationConfig`.
Split counts, embargo sizes, CPCV group counts, the metric set and the random
seed all live here — never as magic numbers in the splitters or harness. The
``validation_version`` string is stamped onto every emitted evaluation summary so
saved runs are self-describing.

Mirrors :mod:`options_system.features.config` and
:mod:`options_system.labeling.config`: a frozen, ``extra='forbid'`` pydantic tree
loaded once and shared. Embargo lengths are expressed as a *fraction of total
bars* (``embargo_pct``); a splitter turns that into an integer bar count with
``ceil(embargo_pct * n)``.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

# config/ lives at the repo root (see config/__init__.py); validation.yaml sits beside it.
_DEFAULT_PATH = Path(__file__).resolve().parents[3] / "config" / "validation.yaml"

# Metric names the harness knows how to compute (see validation/evaluate.py).
_KNOWN_METRICS = frozenset({"accuracy", "auc", "weighted_return"})


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class KFoldCfg(_Base):
    n_splits: int = Field(gt=1)
    embargo_pct: float = Field(ge=0.0, lt=1.0)


class CPCVCfg(_Base):
    n_groups: int = Field(gt=1)
    test_groups: int = Field(gt=0)
    embargo_pct: float = Field(ge=0.0, lt=1.0)


class WalkForwardCfg(_Base):
    scheme: str
    n_splits: int = Field(gt=1)
    min_train_bars: int = Field(ge=0)
    embargo_pct: float = Field(ge=0.0, lt=1.0)

    @field_validator("scheme")
    @classmethod
    def _known_scheme(cls, v: str) -> str:
        s = v.strip().lower()
        if s not in {"anchored", "rolling"}:
            raise ValueError(f"walk_forward.scheme={v!r} invalid; use 'anchored' or 'rolling'")
        return s


class EvaluationCfg(_Base):
    metrics: list[str]
    primary_metric: str
    seed: int = Field(ge=0)

    @field_validator("metrics")
    @classmethod
    def _known_metrics(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("evaluation.metrics must list at least one metric")
        unknown = [m for m in v if m not in _KNOWN_METRICS]
        if unknown:
            raise ValueError(
                f"evaluation.metrics has unknown entries {unknown}; known={sorted(_KNOWN_METRICS)}"
            )
        return v


class ValidationConfig(_Base):
    """Validated validation-framework configuration (one object, loaded once, shared)."""

    validation_version: str
    kfold: KFoldCfg
    cpcv: CPCVCfg
    walk_forward: WalkForwardCfg
    evaluation: EvaluationCfg

    @field_validator("cpcv")
    @classmethod
    def _cpcv_k_lt_n(cls, v: CPCVCfg) -> CPCVCfg:
        if v.test_groups >= v.n_groups:
            raise ValueError(
                f"cpcv.test_groups ({v.test_groups}) must be < cpcv.n_groups ({v.n_groups})"
            )
        return v

    @field_validator("evaluation")
    @classmethod
    def _primary_in_metrics(cls, v: EvaluationCfg) -> EvaluationCfg:
        if v.primary_metric not in v.metrics:
            raise ValueError(
                f"evaluation.primary_metric={v.primary_metric!r} must be one of "
                f"evaluation.metrics {v.metrics}"
            )
        return v

    @classmethod
    def load(cls, path: str | Path | None = None) -> ValidationConfig:
        """Load + validate the validation config from YAML (default: config/validation.yaml)."""
        p = Path(path) if path is not None else _DEFAULT_PATH
        with p.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        return cls.model_validate(data)

    def to_dict(self) -> dict:
        """Round-trippable plain dict."""
        return self.model_dump(mode="json")


if __name__ == "__main__":  # pragma: no cover - manual smoke check
    cfg = ValidationConfig.load()
    print(
        f"validation_version={cfg.validation_version} "
        f"kfold={cfg.kfold.n_splits} cpcv={cfg.cpcv.n_groups}/{cfg.cpcv.test_groups}"
    )
