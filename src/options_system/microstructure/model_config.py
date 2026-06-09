"""Typed, declarative microstructure signal-model configuration.

Loads ``config/micro_model.yaml`` into a validated :class:`MicroModelConfig`. This
is the **short-horizon microstructure** sibling of :mod:`options_system.models.config`
(the daily directional model): a 3-class target, fold-local class weighting for the
~78-80% timeout regime, a gross-signal-return selection metric, and verdict gates
fixed a priori.

Mirrors the pattern of :mod:`options_system.models.config` and
:mod:`options_system.validation.config`: a frozen, ``extra='forbid'`` pydantic tree
loaded once and shared. The ``micro_model_version`` string is stamped onto every
saved run so artifacts are self-describing and never collide with the daily model.
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

# config/ lives at the repo root; micro_model.yaml sits beside it.
_DEFAULT_PATH = Path(__file__).resolve().parents[3] / "config" / "micro_model.yaml"

# Grid / param keys LightGBM requires to be integers (cast on apply).
_INT_PARAMS = frozenset(
    {"n_estimators", "max_depth", "num_leaves", "min_child_samples", "subsample_freq", "max_bin"}
)
# Estimator-constructor hyperparameters (the subset of the lgbm block the wrapper
# takes). objective/num_class/n_jobs/seed are handled separately (see lgbm.py).
_ESTIMATOR_PARAMS = (
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
_KNOWN_SELECTION = frozenset({"gross_signal_sharpe", "excess_signal_sharpe", "macro_f1"})
_KNOWN_TARGET_MODE = frozenset({"multiclass_3"})
_KNOWN_WEIGHT_METHOD = frozenset({"balanced_fold_local"})


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TargetCfg(_Base):
    mode: str
    classes: list[int]

    @field_validator("mode")
    @classmethod
    def _known_mode(cls, v: str) -> str:
        s = v.strip().lower()
        if s not in _KNOWN_TARGET_MODE:
            raise ValueError(f"target.mode={v!r} invalid; use one of {sorted(_KNOWN_TARGET_MODE)}")
        return s

    @field_validator("classes")
    @classmethod
    def _three_class(cls, v: list[int]) -> list[int]:
        if v != [-1, 0, 1]:
            raise ValueError(f"target.classes must be exactly [-1, 0, 1], got {v}")
        return v


class ClassWeightingCfg(_Base):
    enabled: bool = True
    method: str
    use_sample_weight_in_balance: bool = True

    @field_validator("method")
    @classmethod
    def _known_method(cls, v: str) -> str:
        s = v.strip().lower()
        if s not in _KNOWN_WEIGHT_METHOD:
            raise ValueError(
                f"class_weighting.method={v!r} invalid; use one of {sorted(_KNOWN_WEIGHT_METHOD)}"
            )
        return s


class LgbmCfg(_Base):
    objective: str
    num_class: int = Field(gt=1)
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
    n_jobs: int = Field(ge=1)
    seed: int = Field(ge=0)

    @field_validator("objective")
    @classmethod
    def _multiclass(cls, v: str) -> str:
        if v.strip().lower() != "multiclass":
            raise ValueError(f"lgbm.objective must be 'multiclass', got {v!r}")
        return v.strip().lower()

    @field_validator("num_class")
    @classmethod
    def _three(cls, v: int) -> int:
        if v != 3:
            raise ValueError(f"lgbm.num_class must be 3 for the 3-class target, got {v}")
        return v

    def as_params(self, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
        """The estimator-constructor hyperparameter dict (the wrapper's args), with
        optional grid overrides merged + int-cast. objective/num_class/n_jobs/seed
        are NOT included here — they are fixed by the wrapper or passed separately."""
        params: dict[str, Any] = {k: getattr(self, k) for k in _ESTIMATOR_PARAMS}
        for k, v in (overrides or {}).items():
            params[k] = int(v) if k in _INT_PARAMS else v
        return params


class EarlyStoppingCfg(_Base):
    enabled: bool = True
    rounds: int = Field(gt=0)
    inner_val_fraction: float = Field(gt=0.0, lt=1.0)


class SearchCfg(_Base):
    enabled: bool = True
    selection_metric: str
    grid: dict[str, list[Any]]

    @field_validator("selection_metric")
    @classmethod
    def _known_metric(cls, v: str) -> str:
        s = v.strip().lower()
        if s not in _KNOWN_SELECTION:
            raise ValueError(
                f"search.selection_metric={v!r} invalid; use one of {sorted(_KNOWN_SELECTION)}"
            )
        return s

    @field_validator("grid")
    @classmethod
    def _non_empty_axes(cls, v: dict[str, list[Any]]) -> dict[str, list[Any]]:
        for key, vals in v.items():
            if not vals:
                raise ValueError(f"search.grid[{key!r}] must list at least one value")
        return v

    def configs(self) -> list[dict[str, Any]]:
        """Cartesian product of the grid → list of per-config override dicts.

        Deterministic order (sorted axis names). An empty/disabled grid yields a
        single empty override (the base params), so ``n_trials`` is always >= 1.
        """
        if not self.enabled or not self.grid:
            return [{}]
        axes = sorted(self.grid)
        combos = itertools.product(*(self.grid[a] for a in axes))
        return [dict(zip(axes, combo, strict=True)) for combo in combos]

    @property
    def n_trials(self) -> int:
        """Number of configurations tried (fed to the Deflated Sharpe Ratio + PBO)."""
        return len(self.configs())


class VerdictCfg(_Base):
    max_pbo: float = Field(ge=0.0, le=1.0)
    min_gross_dsr: float = Field(ge=0.0, le=1.0)
    require_positive_gross_return: bool = True
    min_action_rate: float = Field(ge=0.0, le=1.0)
    min_macro_f1: float = Field(ge=0.0, le=1.0)
    require_cpcv_median_gross_sharpe_positive: bool = True


class MicroModelConfig(_Base):
    """Validated microstructure signal-model configuration (loaded once, shared)."""

    micro_model_version: str
    target: TargetCfg
    class_weighting: ClassWeightingCfg
    lgbm: LgbmCfg
    early_stopping: EarlyStoppingCfg
    search: SearchCfg
    verdict: VerdictCfg
    seed: int = Field(ge=0)

    @classmethod
    def load(cls, path: str | Path | None = None) -> MicroModelConfig:
        """Load + validate the micro-model config from YAML (default: config/micro_model.yaml)."""
        p = Path(path) if path is not None else _DEFAULT_PATH
        with p.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        return cls.model_validate(data)

    def to_dict(self) -> dict:
        """Round-trippable plain dict."""
        return self.model_dump(mode="json")


if __name__ == "__main__":  # pragma: no cover - manual smoke check
    cfg = MicroModelConfig.load()
    print(
        f"micro_model_version={cfg.micro_model_version} mode={cfg.target.mode} "
        f"n_trials={cfg.search.n_trials} metric={cfg.search.selection_metric} seed={cfg.seed}"
    )
