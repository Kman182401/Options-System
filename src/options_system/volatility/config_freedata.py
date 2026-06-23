"""Phase-24 config loader — the frozen free-data incremental-value contract.

Loads ``config/phase24_freedata.yaml`` (frozen by ``docs/PHASE24_PREREGISTRATION.md``). The entire
**modeling core** (RV target, the fixed LightGBM, h = 1, walk-forward, regime, DM) is **inherited
verbatim** from the Phase-23 contract (``config/phase23_vol_h1.yaml``) via :class:`Phase23Config` —
so the Phase-24 baseline IS the confirmed Phase-23 model with zero drift. This file parses only
the Phase-24-specific knobs (the additive arms, the coverage threshold, the gate thresholds) and
exposes the baseline / augmented :class:`VolatilityConfig` for each arm.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from .config import VolatilityConfig
from .config_h1 import Phase23Config

_DEFAULT_PATH = Path(__file__).resolve().parents[3] / "config" / "phase24_freedata.yaml"


class _Lenient(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True, populate_by_name=True)


class Arm(_Lenient):
    key: str
    block: str  # the features toggle flipped ON: "marketdata" -> with_marketdata, "gkg" -> with_gkg
    col_prefix: str
    description: str = ""

    @property
    def feature_flag(self) -> str:
        return f"with_{self.block}"


class G3Cfg(_Lenient):
    n_folds: int = Field(default=18, gt=1)
    min_folds_beating_baseline: int = Field(default=13, gt=0)


class GatesCfg(_Lenient):
    g3_temporal_stability: G3Cfg = G3Cfg()


class CoverageCfg(_Lenient):
    min_oos_fraction: float = Field(default=0.80, gt=0.0, le=1.0)


class Phase24Config(_Lenient):
    """The Phase-24 contract: the inherited Phase-23 core + the additive arms / coverage / gates."""

    freedata_version: str
    core: VolatilityConfig  # = the confirmed Phase-23 core (h = 1)
    arms: list[Arm]
    coverage: CoverageCfg
    gates: GatesCfg
    artifacts: dict = Field(default_factory=dict)

    def baseline_core(self) -> VolatilityConfig:
        """The baseline arm = the Phase-23 core with both free-data blocks OFF (confirmed model)."""
        feats = self.core.features.model_copy(update={"with_marketdata": False, "with_gkg": False})
        return self.core.model_copy(update={"features": feats})

    def augmented_core(self, arm: Arm) -> VolatilityConfig:
        """The baseline with the arm's block toggled ON (the only delta from the baseline)."""
        base = self.baseline_core()
        feats = base.features.model_copy(update={arm.feature_flag: True})
        return base.model_copy(update={"features": feats})

    @classmethod
    def load(cls, path: str | Path | None = None) -> Phase24Config:
        p = Path(path) if path is not None else _DEFAULT_PATH
        with p.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        core = Phase23Config.load().core  # inherit the frozen Phase-23 modeling core verbatim
        return cls(
            freedata_version=data["freedata_version"],
            core=core,
            arms=[Arm.model_validate(a) for a in data.get("arms", [])],
            coverage=CoverageCfg.model_validate(data.get("coverage", {})),
            gates=GatesCfg.model_validate(data.get("gates", {})),
            artifacts=data.get("artifacts", {}) or {},
        )
