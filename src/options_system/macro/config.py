"""Typed, declarative macro/economic-event configuration.

Loads ``config/macro.yaml`` into a validated :class:`MacroConfig`. The event
calendar (FRED series ids + standard release clock times), the FOMC scheduled
meeting dates, and the leak-safe feature parameters all live here — never as
magic values in :mod:`options_system.macro.ingest` or
:mod:`options_system.features.macro_features`.

Two version stamps make stored artifacts self-describing:

* ``macro_version`` — bumped if a change would alter ingested event *rows*.
* ``features.macro_feature_version`` — bumped if a change would alter emitted
  feature *values*.

Mirrors :mod:`options_system.features.config` and the other config trees: a
frozen, ``extra='forbid'`` pydantic model loaded once and shared.
"""

from __future__ import annotations

from datetime import date, time
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

# config/ lives at the repo root (see config/__init__.py); macro.yaml sits beside it.
_DEFAULT_PATH = Path(__file__).resolve().parents[3] / "config" / "macro.yaml"


def _parse_et(v: str | time) -> time:
    """Parse an ``"HH:MM"`` Eastern release clock string into a ``datetime.time``."""
    if isinstance(v, time):
        return v
    hh, mm = (int(x) for x in str(v).split(":", 1))
    return time(hour=hh, minute=mm)


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class EventSpec(_Base):
    """One FRED-sourced economic-data release type."""

    series_id: str
    release_et: time
    label: str
    high_impact: bool = True

    @field_validator("release_et", mode="before")
    @classmethod
    def _coerce_time(cls, v: str | time) -> time:
        return _parse_et(v)


class FomcCfg(_Base):
    """FOMC scheduled meetings: timing calendar + the rate-outcome series."""

    release_et: time
    rate_series_id: str
    high_impact: bool = True
    decision_dates: list[date]

    @field_validator("release_et", mode="before")
    @classmethod
    def _coerce_time(cls, v: str | time) -> time:
        return _parse_et(v)

    @field_validator("decision_dates")
    @classmethod
    def _sorted_unique(cls, v: list[date]) -> list[date]:
        if len(set(v)) != len(v):
            raise ValueError("fomc.decision_dates contains duplicates")
        if v != sorted(v):
            raise ValueError("fomc.decision_dates must be in ascending order")
        return v


class MacroFeaturesCfg(_Base):
    """Parameters for the leak-safe macro features."""

    macro_feature_version: str
    timing_types: list[str]
    outcome_types: list[str]
    tendency_types: list[str]
    tendency_window: int = Field(gt=0)
    blackout_lead_minutes: int = Field(ge=0)
    next_horizon_hours: int = Field(gt=0)


class MacroConfig(_Base):
    """Validated macro configuration (one object, loaded once, shared)."""

    macro_version: str
    timezone: str
    events: dict[str, EventSpec]
    fomc: FomcCfg
    features: MacroFeaturesCfg

    @field_validator("events")
    @classmethod
    def _non_empty(cls, v: dict[str, EventSpec]) -> dict[str, EventSpec]:
        if not v:
            raise ValueError("events must declare at least one economic-data series")
        return v

    def event_types(self) -> list[str]:
        """All event-type keys (the FRED-data events plus ``fomc``)."""
        return [*self.events.keys(), "fomc"]

    def high_impact_types(self) -> list[str]:
        """Event types flagged high-impact (drives the aggregate timing features)."""
        hi = [k for k, e in self.events.items() if e.high_impact]
        if self.fomc.high_impact:
            hi.append("fomc")
        return hi

    @classmethod
    def load(cls, path: str | Path | None = None) -> MacroConfig:
        """Load + validate the macro config from YAML (default: config/macro.yaml)."""
        p = Path(path) if path is not None else _DEFAULT_PATH
        with p.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        cfg = cls.model_validate(data)
        cfg._validate_feature_types()
        return cfg

    def _validate_feature_types(self) -> None:
        """Every feature-referenced type must be a known event type."""
        known = set(self.event_types())
        for field, types in (
            ("timing_types", self.features.timing_types),
            ("outcome_types", self.features.outcome_types),
            ("tendency_types", self.features.tendency_types),
        ):
            unknown = [t for t in types if t not in known]
            if unknown:
                raise ValueError(f"features.{field} references unknown event types {unknown}")

    def to_dict(self) -> dict:
        """Round-trippable plain dict (dates/times as ISO strings)."""
        return self.model_dump(mode="json")


if __name__ == "__main__":  # pragma: no cover - manual smoke check
    cfg = MacroConfig.load()
    print(
        f"macro_version={cfg.macro_version} events={len(cfg.events)} "
        f"fomc_meetings={len(cfg.fomc.decision_dates)} "
        f"macro_feature_version={cfg.features.macro_feature_version}"
    )
