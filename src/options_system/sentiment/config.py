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
