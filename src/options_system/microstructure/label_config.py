"""Typed, declarative short-horizon (micro) labeling configuration.

Loads ``config/micro_labeling.yaml`` into a validated :class:`MicroLabelConfig`.
This is the **short-horizon** sibling of :mod:`options_system.labeling.config`
(the daily triple-barrier config): same López de Prado methodology, re-scaled to
an intraday horizon and run on the microstructure **dollar bars** instead of the
1-minute price bars.

Why a separate, frozen config (mirrors ``labeling/config.py`` /
``microstructure/config.py``):

* Every horizon-specific number lives here, never as a magic constant in the
  generator, so the label definition is auditable in one place.
* The ``micro_label_version`` string is stamped onto every emitted label row so
  stored tables are self-describing and never collide with the daily ``v1``
  labels or with each other across versions.
* ANTI-SNOOPING: these parameters are fixed **a priori**, before any model is
  trained. Do NOT tune them to produce a nicer label balance or more events —
  pre-commitment is the guard against data-snooping (see docs/MICRO_LABELING.md).

Horizons here are in **wall-clock minutes** (``barriers.vertical_minutes``) and
**bars** (the EWMA spans), because the dollar bars are NOT time-uniform — a fixed
bar count would not be a fixed amount of time. The RTH session boundary
(09:30-16:00 ET) is NOT redefined here; it is reused from
:class:`options_system.microstructure.config.SessionCfg`.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

# config/ lives at the repo root; micro_labeling.yaml sits beside it.
_DEFAULT_PATH = Path(__file__).resolve().parents[3] / "config" / "micro_labeling.yaml"


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class MicroVolatilityCfg(_Base):
    """Causal volatility estimator, scaled from per-bar to the 30-min horizon.

    σ is built from per-bar mid-price log returns (EWM std, ``adjust=False`` →
    causal recursion). Because dollar bars carry ~equal variance per bar, the
    30-min variance is ``(bars per 30 min) · (per-bar variance)``; the bar rate is
    itself estimated causally as ``vertical_seconds / EWMA(duration_s)``. So
    ``σ_H = σ_bar · sqrt(vertical_seconds / dur_ewma)``.
    """

    ewm_span: int = Field(gt=1)  # smoothing span for the EWM std of per-bar mid log returns
    min_samples: int = Field(gt=1)  # warmup: bars required before σ is defined
    dur_ewm_span: int = Field(gt=1)  # smoothing span for the EWMA of bar duration_s (bar-rate)


class MicroBarriersCfg(_Base):
    pt_mult: float = Field(gt=0)  # upper (profit-take) barrier = +pt_mult · σ_H
    sl_mult: float = Field(gt=0)  # lower (stop-loss)  barrier = −sl_mult · σ_H
    vertical_minutes: float = Field(gt=0)  # vertical (time) barrier, wall-clock minutes from t0
    vertical_label_sign: bool = False  # timeout/close label: false → 0; true → sign(ret at t1)


class MicroEventsCfg(_Base):
    method: str
    cusum_mult: float = Field(gt=0)  # CUSUM threshold h_t = cusum_mult · σ_H,t
    grid_step_bars: int = Field(gt=0)  # grid alternative: emit an event every k bars

    @field_validator("method")
    @classmethod
    def _known_method(cls, v: str) -> str:
        m = v.strip().lower()
        if m not in {"cusum", "grid"}:
            raise ValueError(f"events.method={v!r} invalid; use 'cusum' or 'grid'")
        return m


class MicroWeightsCfg(_Base):
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


class MicroLabelConfig(_Base):
    """Validated short-horizon labeling configuration (one object, loaded once)."""

    micro_label_version: str
    volatility: MicroVolatilityCfg
    barriers: MicroBarriersCfg
    events: MicroEventsCfg
    weights: MicroWeightsCfg

    @classmethod
    def load(cls, path: str | Path | None = None) -> MicroLabelConfig:
        """Load and validate from YAML (defaults to config/micro_labeling.yaml)."""
        p = Path(path) if path is not None else _DEFAULT_PATH
        with p.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        return cls.model_validate(data)

    def to_dict(self) -> dict:
        """Round-trippable plain dict."""
        return self.model_dump(mode="json")


if __name__ == "__main__":  # pragma: no cover - manual smoke check
    cfg = MicroLabelConfig.load()
    print(
        f"micro_label_version={cfg.micro_label_version} "
        f"vertical={cfg.barriers.vertical_minutes}min "
        f"barriers={cfg.barriers.pt_mult}/{cfg.barriers.sl_mult} "
        f"cusum_mult={cfg.events.cusum_mult}"
    )
