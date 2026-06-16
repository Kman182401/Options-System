"""Typed, declarative config for the daily market-state layer (System B).

Loads ``config/marketdata.yaml`` into a validated :class:`MarketDataConfig`, mirroring
the discipline of the macro/sentiment configs (frozen, ``extra='forbid'``, declarative
``source_policy`` cross-checked against the authoritative
:data:`options_system.common.external_data_policy.DEFAULT_REGISTRY`).

These are continuous **daily** series (FRED ``output_type=1``), used as leak-safe
cross-asset features (``x1``), distinct from the macro layer's first-print release events.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from options_system.common.external_data_policy import DEFAULT_REGISTRY, SourcePolicy

_DEFAULT_PATH = Path(__file__).resolve().parents[3] / "config" / "marketdata.yaml"


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Series(_Base):
    id: str  # FRED series id, e.g. VIXCLS
    label: str  # stable human token used in feature column names, e.g. vix
    group: str  # documentation grouping, e.g. volatility

    @field_validator("id", "label", "group")
    @classmethod
    def _nonblank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must be non-blank")
        return v


class Storage(_Base):
    dataset: str


class Features(_Base):
    change_horizons_days: tuple[int, ...]
    zscore_window_days: int = Field(gt=1)
    curve_pairs: tuple[tuple[str, str], ...] = ()

    @field_validator("change_horizons_days")
    @classmethod
    def _positive_horizons(cls, v: tuple[int, ...]) -> tuple[int, ...]:
        if not v or any(h <= 0 for h in v):
            raise ValueError("change_horizons_days must be non-empty positive integers")
        return v


class MarketDataConfig(_Base):
    """Validated daily market-state configuration (loaded once, shared)."""

    marketdata_feature_version: str
    observation_start: date
    series: tuple[Series, ...]
    storage: Storage
    features: Features
    source_policy: dict[str, SourcePolicy]

    @field_validator("series")
    @classmethod
    def _series_nonempty_unique(cls, v: tuple[Series, ...]) -> tuple[Series, ...]:
        if not v:
            raise ValueError("series must be non-empty")
        labels = [s.label for s in v]
        ids = [s.id for s in v]
        if len(set(labels)) != len(labels):
            raise ValueError(f"series labels must be unique; got {labels}")
        if len(set(ids)) != len(ids):
            raise ValueError(f"series ids must be unique; got {ids}")
        return v

    @model_validator(mode="after")
    def _curve_pairs_reference_known_labels(self) -> MarketDataConfig:
        known = {s.label for s in self.series}
        for long_label, short_label in self.features.curve_pairs:
            missing = {long_label, short_label} - known
            if missing:
                raise ValueError(
                    f"features.curve_pairs references unknown label(s) {sorted(missing)}; "
                    f"known labels: {sorted(known)}"
                )
        return self

    @model_validator(mode="after")
    def _policy_agrees_with_code(self) -> MarketDataConfig:
        for src, declared in self.source_policy.items():
            canonical = DEFAULT_REGISTRY.get(src.strip().lower(), SourcePolicy.UNKNOWN_BLOCKED)
            if declared != canonical:
                raise ValueError(
                    f"source_policy[{src!r}]={declared.value!r} disagrees with the authoritative "
                    f"external_data_policy registry ({canonical.value!r}); the code is the truth."
                )
        return self

    def label_for(self, series_id: str) -> str:
        for s in self.series:
            if s.id == series_id:
                return s.label
        raise KeyError(series_id)

    @classmethod
    def load(cls, path: str | Path | None = None) -> MarketDataConfig:
        p = Path(path) if path is not None else _DEFAULT_PATH
        with p.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        return cls.model_validate(data)


if __name__ == "__main__":  # pragma: no cover - manual smoke check
    cfg = MarketDataConfig.load()
    print(
        f"version={cfg.marketdata_feature_version} "
        f"series={[s.label for s in cfg.series]} curves={list(cfg.features.curve_pairs)}"
    )
