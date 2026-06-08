"""Typed, declarative signal-model configuration.

Loads ``config/models.yaml`` into a validated :class:`ModelConfig`. The directional
target policy, the LightGBM regularisation, the early-stopping rule, the
(small) hyperparameter grid and the verdict thresholds all live here — never as
magic numbers in the estimator or evaluator. The ``model_version`` string is
stamped onto every saved run and MLflow log so artifacts are self-describing.

Mirrors :mod:`options_system.validation.config` and
:mod:`options_system.labeling.config`: a frozen, ``extra='forbid'`` pydantic tree
loaded once and shared. Tree depth is in LightGBM units; windows/horizons that
matter to the data live in the labeling/feature configs, not here.
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

# config/ lives at the repo root (see config/__init__.py); models.yaml sits beside it.
_DEFAULT_PATH = Path(__file__).resolve().parents[3] / "config" / "models.yaml"

# Grid / param keys that LightGBM requires to be integers (cast on apply).
_INT_PARAMS = frozenset(
    {"n_estimators", "max_depth", "num_leaves", "min_child_samples", "subsample_freq", "max_bin"}
)
_KNOWN_SELECTION = frozenset({"directional_accuracy", "excess_sharpe"})
_KNOWN_TIMEOUT = frozenset({"sign_return", "drop"})


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TargetCfg(_Base):
    timeout_handling: str

    @field_validator("timeout_handling")
    @classmethod
    def _known(cls, v: str) -> str:
        s = v.strip().lower()
        if s not in _KNOWN_TIMEOUT:
            raise ValueError(
                f"target.timeout_handling={v!r} invalid; use one of {sorted(_KNOWN_TIMEOUT)}"
            )
        return s


class LgbmCfg(_Base):
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
    min_split_gain: float = Field(ge=0.0)
    max_bin: int = Field(gt=1)

    def as_params(self, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
        """Plain LightGBM param dict, with optional grid overrides merged + int-cast."""
        params: dict[str, Any] = self.model_dump()
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
        """Number of configurations tried (fed to the Deflated Sharpe Ratio)."""
        return len(self.configs())


class VerdictCfg(_Base):
    min_directional_accuracy: float = Field(gt=0.0, lt=1.0)
    max_pbo: float = Field(ge=0.0, le=1.0)
    min_excess_dsr: float = Field(ge=0.0, le=1.0)
    require_positive_excess: bool = True


class ModelConfig(_Base):
    """Validated signal-model configuration (one object, loaded once, shared)."""

    model_version: str
    target: TargetCfg
    lgbm: LgbmCfg
    early_stopping: EarlyStoppingCfg
    search: SearchCfg
    verdict: VerdictCfg
    seed: int = Field(ge=0)

    @classmethod
    def load(cls, path: str | Path | None = None) -> ModelConfig:
        """Load + validate the model config from YAML (default: config/models.yaml)."""
        p = Path(path) if path is not None else _DEFAULT_PATH
        with p.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        return cls.model_validate(data)

    def to_dict(self) -> dict:
        """Round-trippable plain dict."""
        return self.model_dump(mode="json")


if __name__ == "__main__":  # pragma: no cover - manual smoke check
    cfg = ModelConfig.load()
    print(
        f"model_version={cfg.model_version} n_trials={cfg.search.n_trials} "
        f"timeout={cfg.target.timeout_handling} seed={cfg.seed}"
    )
