"""Phase-23 config loader — the frozen 1-day RV-forecast confirmation contract.

Loads ``config/phase23_vol_h1.yaml`` (frozen by ``docs/PHASE23_PREREGISTRATION.md``). The core
estimator/model/walk-forward/regime/feature knobs are validated by **reusing the Phase-21
:class:`VolatilityConfig`** unchanged (so the confirmation runs the identical pipeline); the new
Phase-23 sections — the benchmark battery and the G3/G4 gate thresholds — are parsed here.

Descriptive ``rule``/``spec`` strings in the YAML are documentation of the frozen logic (the code is
the source of truth); the parsers ``ignore`` them rather than re-encode them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from .config import VolatilityConfig

_DEFAULT_PATH = Path(__file__).resolve().parents[3] / "config" / "phase23_vol_h1.yaml"

# The top-level keys VolatilityConfig (the frozen Phase-21 core) owns.
_CORE_KEYS = (
    "volatility_version",
    "seed",
    "symbols",
    "rv",
    "horizons",
    "har",
    "lgbm",
    "walk_forward",
    "regime",
    "dm",
    "features",
)


class _Lenient(BaseModel):
    # ignore the YAML's descriptive rule/spec strings — the code owns the logic.
    model_config = ConfigDict(extra="ignore", frozen=True, populate_by_name=True)


class EwmaCfg(_Lenient):
    enabled: bool = True
    lam: float = Field(default=0.94, alias="lambda", gt=0.0, lt=1.0)


class GarchCfg(_Lenient):
    enabled: bool = True


class BenchmarksCfg(_Lenient):
    har: bool = True
    random_walk: bool = True
    ewma: EwmaCfg = EwmaCfg()
    garch: GarchCfg = GarchCfg()


class G3Cfg(_Lenient):
    challengers: list[str] = Field(default_factory=lambda: ["random_walk", "ewma", "garch"])


class G4Cfg(_Lenient):
    n_folds: int = Field(default=18, gt=1)
    min_folds_beating_har: int = Field(default=13, gt=0)
    min_folds_beating_rw: int = Field(default=13, gt=0)


class GatesCfg(_Lenient):
    g3_benchmark_hardness: G3Cfg = G3Cfg()
    g4_temporal_stability: G4Cfg = G4Cfg()


class Phase23Config(_Lenient):
    """The full Phase-23 contract: the reused Phase-21 core + the new battery/gate knobs."""

    core: VolatilityConfig
    benchmarks: BenchmarksCfg
    gates: GatesCfg
    artifacts: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path | None = None) -> Phase23Config:
        p = Path(path) if path is not None else _DEFAULT_PATH
        with p.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        core = VolatilityConfig.model_validate({k: data[k] for k in _CORE_KEYS})
        return cls(
            core=core,
            benchmarks=BenchmarksCfg.model_validate(data.get("benchmarks", {})),
            gates=GatesCfg.model_validate(data.get("gates", {})),
            artifacts=data.get("artifacts", {}) or {},
        )
