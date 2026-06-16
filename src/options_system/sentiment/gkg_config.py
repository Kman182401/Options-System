"""Typed, declarative config for the GDELT GKG bulk-archive ingestion (System A).

Loads ``config/sentiment_gkg.yaml`` into a validated :class:`GkgConfig`, mirroring the
discipline of :class:`options_system.sentiment.config.SentimentConfig` (frozen,
``extra='forbid'``, declarative ``source_policy`` cross-checked against the
authoritative :data:`options_system.common.external_data_policy.DEFAULT_REGISTRY` so the
YAML can never silently widen access).

GKG is a separate source (``gdelt_gkg``) with its own lake datasets and its own raw
event-schema axis (``gkg_event_version``, the g-series). This config governs the bulk
downloader + the theme row-filter; the point-in-time *aggregate* feature layer (s3) is
configured later (System D).
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import ClassVar

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from options_system.common.external_data_policy import DEFAULT_REGISTRY, SourcePolicy

# config/ lives at the repo root; sentiment_gkg.yaml sits beside it.
_DEFAULT_PATH = Path(__file__).resolve().parents[3] / "config" / "sentiment_gkg.yaml"


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Window(_Base):
    start: date
    end: date

    @field_validator("end")
    @classmethod
    def _ordered(cls, v: date, info) -> date:  # noqa: ANN001 - pydantic validation info
        start = info.data.get("start")
        if start is not None and v < start:
            raise ValueError(f"window.end ({v}) must be on/after window.start ({start})")
        return v


class Storage(_Base):
    raw_dataset: str
    scored_dataset: str

    @model_validator(mode="after")
    def _distinct_from_finbert(self) -> Storage:
        """GKG must never write into the FinBERT (s2) lake — keep the datasets isolated."""
        if self.raw_dataset in {"sentiment_raw", "sentiment_scores"} or self.scored_dataset in {
            "sentiment_raw",
            "sentiment_scores",
        }:
            raise ValueError(
                "GKG storage datasets must be distinct from the FinBERT sentiment lake "
                "(sentiment_raw / sentiment_scores) to keep the s2 pipeline uncontaminated."
            )
        return self


class Backfill(_Base):
    """Bounded GKG bulk-download caps + pacing. Every cap is fail-closed: hitting it
    stops the run cleanly with the resumable manifest intact."""

    max_files: int = Field(gt=0)
    max_wall_clock_minutes: int = Field(gt=0)
    max_bytes: int = Field(gt=0)
    workers: int = Field(gt=0, le=16)
    politeness_delay_s: float = Field(ge=0.0, le=60.0)
    retry_max: int = Field(ge=0, le=20)


class GkgConfig(_Base):
    """Validated GKG ingestion configuration (one object, loaded once, shared)."""

    gkg_event_version: str
    tone_model_name: str
    query_topic: str
    window: Window
    theme_prefixes: tuple[str, ...]
    storage: Storage
    source_policy: dict[str, SourcePolicy]
    backfill: Backfill

    SOURCE: ClassVar[str] = "gdelt_gkg"

    @field_validator("theme_prefixes")
    @classmethod
    def _nonempty_prefixes(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        if not v:
            raise ValueError("theme_prefixes must be non-empty (the filter would keep nothing)")
        if any(not p.strip() for p in v):
            raise ValueError("theme_prefixes must not contain blank entries")
        return v

    @field_validator("gkg_event_version", "tone_model_name", "query_topic")
    @classmethod
    def _nonblank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must be non-blank")
        return v

    @model_validator(mode="after")
    def _policy_agrees_with_code(self) -> GkgConfig:
        """The declared source_policy must match the authoritative code registry exactly,
        and the GKG source itself must be a free, network-eligible source."""
        for src, declared in self.source_policy.items():
            canonical = DEFAULT_REGISTRY.get(src.strip().lower(), SourcePolicy.UNKNOWN_BLOCKED)
            if declared != canonical:
                raise ValueError(
                    f"source_policy[{src!r}]={declared.value!r} disagrees with the authoritative "
                    f"external_data_policy registry ({canonical.value!r}); the code is the truth."
                )
        canonical_gkg = DEFAULT_REGISTRY.get(self.SOURCE, SourcePolicy.UNKNOWN_BLOCKED)
        if canonical_gkg is not SourcePolicy.FREE_NO_AUTH:
            raise ValueError(
                f"source {self.SOURCE!r} must be FREE_NO_AUTH in the registry; got {canonical_gkg}."
            )
        return self

    @classmethod
    def load(cls, path: str | Path | None = None) -> GkgConfig:
        """Load and validate from YAML (defaults to config/sentiment_gkg.yaml)."""
        p = Path(path) if path is not None else _DEFAULT_PATH
        with p.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        return cls.model_validate(data)


if __name__ == "__main__":  # pragma: no cover - manual smoke check
    cfg = GkgConfig.load()
    print(
        f"gkg_event_version={cfg.gkg_event_version} themes={list(cfg.theme_prefixes)} "
        f"window={cfg.window.start}..{cfg.window.end} workers={cfg.backfill.workers}"
    )
