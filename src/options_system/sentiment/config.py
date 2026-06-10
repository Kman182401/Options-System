"""Typed, declarative sentiment-layer configuration.

Loads ``config/sentiment.yaml`` into a validated :class:`SentimentConfig`, mirroring
the pattern of :mod:`options_system.microstructure.config` (frozen, ``extra='forbid'``
pydantic tree loaded once and shared). The declarative ``source_policy`` block is
validated against the authoritative
:data:`options_system.common.external_data_policy.DEFAULT_REGISTRY` so the YAML can
never silently widen access beyond what the code allows.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from options_system.common.external_data_policy import (
    DEFAULT_REGISTRY,
    SourcePolicy,
)

# config/ lives at the repo root; sentiment.yaml sits beside it.
_DEFAULT_PATH = Path(__file__).resolve().parents[3] / "config" / "sentiment.yaml"


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Window(_Base):
    start: date
    end: date

    @field_validator("end")
    @classmethod
    def _ordered(cls, v: date, info) -> date:  # noqa: ANN001 - pydantic validation info
        start = info.data.get("start")
        if start is not None and v <= start:
            raise ValueError(f"windows.end ({v}) must be after windows.start ({start})")
        return v


class Scoring(_Base):
    model: str
    labels: tuple[str, ...]
    local_only: bool = True

    @field_validator("labels")
    @classmethod
    def _three_labels(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        if set(v) != {"positive", "negative", "neutral"}:
            raise ValueError(f"scoring.labels must be the FinBERT 3-class set; got {list(v)}")
        return v


class Storage(_Base):
    raw_dataset: str
    scored_dataset: str


class FetchLimits(_Base):
    max_records: int = Field(gt=0, le=10000)
    default_language: str | None = None
    sec_user_agent: str


# The nine aggregate fields the layer knows how to compute (the authoritative set;
# config may select a subset/order but never invent a name). Mirrors features.py.
_KNOWN_AGG_FIELDS = frozenset(
    {
        "event_count",
        "degraded_count",
        "mean_sentiment_score",
        "sum_sentiment_score",
        "mean_positive_score",
        "mean_negative_score",
        "mean_neutral_score",
        "max_abs_sentiment_score",
        "latest_observed_age_minutes",
    }
)
_KNOWN_AGG_GROUPS = frozenset({"all_sources_all_topics", "by_source", "by_topic"})


class Aggregation(_Base):
    """Point-in-time sentiment feature-aggregation config (Phase 17).

    A separate version axis from the event schema: ``feature_version`` (s2) versions
    the AGGREGATE layer, while the top-level ``sentiment_feature_version`` (s1) versions
    the raw/scored event schema. ``windows`` maps a window name to its length in minutes;
    aggregates use the half-open window ``(t - window, t]`` on ``observed_at``.
    """

    feature_version: str
    windows: dict[str, int]
    groups: tuple[str, ...]
    fields: tuple[str, ...]
    breakdown_fields: tuple[str, ...]
    emit_has_any: bool = True
    breakdown_sources: tuple[str, ...] = ()
    breakdown_topics: tuple[str, ...] = ()

    @field_validator("windows")
    @classmethod
    def _windows_positive(cls, v: dict[str, int]) -> dict[str, int]:
        if not v:
            raise ValueError("aggregation.windows must be non-empty")
        for name, minutes in v.items():
            if minutes <= 0:
                raise ValueError(f"aggregation.windows[{name!r}]={minutes} must be > 0 minutes")
        return v

    @field_validator("groups")
    @classmethod
    def _known_groups(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        if not v:
            raise ValueError("aggregation.groups must be non-empty")
        bad = sorted(set(v) - _KNOWN_AGG_GROUPS)
        if bad:
            raise ValueError(
                f"aggregation.groups has unknown groups {bad}; known: {sorted(_KNOWN_AGG_GROUPS)}"
            )
        return v

    @field_validator("fields", "breakdown_fields")
    @classmethod
    def _known_fields(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        if not v:
            raise ValueError("aggregation field list must be non-empty")
        bad = sorted(set(v) - _KNOWN_AGG_FIELDS)
        if bad:
            raise ValueError(
                f"unknown aggregate field(s) {bad}; known: {sorted(_KNOWN_AGG_FIELDS)}"
            )
        return v


class SentimentConfig(_Base):
    """Validated sentiment configuration (one object, loaded once, shared)."""

    sentiment_feature_version: str
    default_sources: tuple[str, ...]
    source_policy: dict[str, SourcePolicy]
    windows: Window
    query_topics: tuple[str, ...]
    scoring: Scoring
    storage: Storage
    fetch_limits: FetchLimits
    aggregation: Aggregation

    @field_validator("default_sources", "query_topics")
    @classmethod
    def _nonempty(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        if not v:
            raise ValueError("must be non-empty")
        return v

    @model_validator(mode="after")
    def _policy_agrees_with_code(self) -> SentimentConfig:
        """The YAML source_policy must match the authoritative code registry exactly
        for every source it declares — config can never widen (or contradict) the
        fail-closed policy enforced in external_data_policy."""
        for src, declared in self.source_policy.items():
            canonical = DEFAULT_REGISTRY.get(src.strip().lower(), SourcePolicy.UNKNOWN_BLOCKED)
            if declared != canonical:
                raise ValueError(
                    f"source_policy[{src!r}]={declared.value!r} disagrees with the authoritative "
                    f"external_data_policy registry ({canonical.value!r}). Fix the YAML; the code "
                    f"registry is the source of truth."
                )
        # Every default source must be one we actually allow (free or local).
        for src in self.default_sources:
            pol = DEFAULT_REGISTRY.get(src.strip().lower(), SourcePolicy.UNKNOWN_BLOCKED)
            if pol in (SourcePolicy.PAID_BLOCKED, SourcePolicy.UNKNOWN_BLOCKED):
                raise ValueError(
                    f"default_sources lists {src!r} with blocked policy {pol.value!r}."
                )
        return self

    @model_validator(mode="after")
    def _aggregation_keys_are_known(self) -> SentimentConfig:
        """Aggregation breakdown keys must be subsets of the declared sources/topics, and
        breakdown_fields a subset of the global fields — so feature columns can only ever
        come from vetted, sanitizable names, never arbitrary raw strings."""
        agg = self.aggregation
        unknown_src = sorted(set(agg.breakdown_sources) - set(self.default_sources))
        if unknown_src:
            raise ValueError(
                f"aggregation.breakdown_sources {unknown_src} not in default_sources "
                f"{list(self.default_sources)}"
            )
        unknown_topic = sorted(set(agg.breakdown_topics) - set(self.query_topics))
        if unknown_topic:
            raise ValueError(
                f"aggregation.breakdown_topics {unknown_topic} not in query_topics "
                f"{list(self.query_topics)}"
            )
        unknown_bf = sorted(set(agg.breakdown_fields) - set(agg.fields))
        if unknown_bf:
            raise ValueError(
                f"aggregation.breakdown_fields {unknown_bf} not in aggregation.fields "
                f"{list(agg.fields)}"
            )
        return self

    @classmethod
    def load(cls, path: str | Path | None = None) -> SentimentConfig:
        """Load and validate from YAML (defaults to config/sentiment.yaml)."""
        p = Path(path) if path is not None else _DEFAULT_PATH
        with p.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        return cls.model_validate(data)

    def to_dict(self) -> dict:
        """Round-trippable plain dict (dates as ISO strings)."""
        return self.model_dump(mode="json")


if __name__ == "__main__":  # pragma: no cover - manual smoke check
    cfg = SentimentConfig.load()
    print(
        f"sentiment_feature_version={cfg.sentiment_feature_version} "
        f"sources={list(cfg.default_sources)} topics={len(cfg.query_topics)} "
        f"window={cfg.windows.start}..{cfg.windows.end} model={cfg.scoring.model}"
    )
