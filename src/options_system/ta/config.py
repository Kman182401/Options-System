"""Typed, declarative technical-analysis (TA) feature configuration.

Loads ``config/ta.yaml`` into a validated :class:`TaConfig`. Indicator families,
windows (in bars = minutes), and parameters live here — never as magic numbers in
:mod:`options_system.ta.compute`. The ``ta_feature_version`` string is stamped
onto every emitted row so stored tables are self-describing.

This is the additive ``feature_version = v2`` layer: a curated set of classic
oscillators (Stochastic, CCI, MFI, Vortex, TRIX) that complement — and never
duplicate — the price ``feature_version = v1`` layer. Windows are *trailing* by
construction in the engine; this config only declares sizes, never look-ahead.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

# config/ lives at the repo root (see config/__init__.py); ta.yaml sits beside it.
_DEFAULT_PATH = Path(__file__).resolve().parents[3] / "config" / "ta.yaml"


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class StochCfg(_Base):
    k_window: int = Field(gt=1)
    d_smooth: int = Field(gt=0)


class CciCfg(_Base):
    window: int = Field(gt=1)


class MfiCfg(_Base):
    window: int = Field(gt=1)


class VortexCfg(_Base):
    window: int = Field(gt=1)


class TrixCfg(_Base):
    window: int = Field(gt=1)


class TaConfig(_Base):
    """Validated TA feature configuration (one object, loaded once, shared)."""

    ta_feature_version: str
    stoch: StochCfg
    cci: CciCfg
    mfi: MfiCfg
    vortex: VortexCfg
    trix: TrixCfg
    degraded_days: list[date]

    @classmethod
    def load(cls, path: str | Path | None = None) -> TaConfig:
        """Load and validate the TA config from YAML (defaults to config/ta.yaml)."""
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
        """Largest trailing lookback anywhere in the config (drives the warmup flag).

        TRIX is a triple-nested EWM, so its effective warmup is ~3x its span; CCI's
        mean-absolute-deviation chains two rolling windows (~2x). The values below
        upper-bound each family's first fully-populated bar.
        """
        return max(
            self.stoch.k_window + self.stoch.d_smooth,
            2 * self.cci.window,
            self.mfi.window,
            self.vortex.window,
            3 * self.trix.window,
        )


if __name__ == "__main__":  # pragma: no cover - manual smoke check
    cfg = TaConfig.load()
    print(f"ta_feature_version={cfg.ta_feature_version} max_window={cfg.max_window()}")
