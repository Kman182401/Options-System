"""Typed, declarative microstructure configuration.

Loads ``config/microstructure.yaml`` into a validated :class:`MicrostructureConfig`.
Instruments, the dollar-bar thresholds, the session policy, the order-flow
parameters, the hard ``databento_budget_usd_cap`` and the ingestion controls live
here — never as magic numbers in the reducer or the loader. The
``microstructure_feature_version`` string is stamped onto every emitted bar row so
stored tables are self-describing.

Mirrors the pattern of :mod:`options_system.features.config` and
:mod:`options_system.labeling.config`: a frozen, ``extra='forbid'`` pydantic tree
loaded once and shared.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

# config/ lives at the repo root; microstructure.yaml sits beside it.
_DEFAULT_PATH = Path(__file__).resolve().parents[3] / "config" / "microstructure.yaml"


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
            raise ValueError(f"window.end ({v}) must be after window.start ({start})")
        return v


class SessionCfg(_Base):
    tz: str
    rth_only: bool = True
    rth_open_min: int = Field(ge=0, lt=1440)
    rth_close_min: int = Field(ge=0, le=1440)

    @field_validator("rth_close_min")
    @classmethod
    def _close_after_open(cls, v: int, info) -> int:  # noqa: ANN001
        open_m = info.data.get("rth_open_min")
        if open_m is not None and v <= open_m:
            raise ValueError(f"rth_close_min ({v}) must be after rth_open_min ({open_m})")
        return v


class Instrument(_Base):
    symbol: str
    continuous_symbol: str
    exec_symbol: str
    multiplier: float = Field(gt=0)
    tick_size: float = Field(gt=0)
    dollar_threshold: float = Field(gt=0)


class OFICfg(_Base):
    rolling_bars: int = Field(ge=1)


class IngestCfg(_Base):
    chunk: str
    retries: int = Field(gt=0)
    backoff_s: float = Field(gt=0)

    @field_validator("chunk")
    @classmethod
    def _known_chunk(cls, v: str) -> str:
        c = v.strip().lower()
        if c != "day":
            raise ValueError(f"ingest.chunk={v!r} invalid; only 'day' is supported")
        return c


class MicrostructureConfig(_Base):
    """Validated microstructure configuration (one object, loaded once, shared)."""

    microstructure_feature_version: str
    dataset: str
    schema_: str = Field(alias="schema")
    window: Window
    databento_budget_usd_cap: float = Field(gt=0)
    session: SessionCfg
    instruments: list[Instrument]
    ofi: OFICfg
    ingest: IngestCfg

    # ``schema`` is a BaseModel attribute name, so the YAML key 'schema' is mapped
    # to ``schema_`` via the alias above; allow population by either.
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    @field_validator("instruments")
    @classmethod
    def _nonempty_unique(cls, v: list[Instrument]) -> list[Instrument]:
        if not v:
            raise ValueError("instruments must be non-empty")
        syms = [i.symbol for i in v]
        if len(set(syms)) != len(syms):
            raise ValueError(f"duplicate instrument symbols: {syms}")
        return v

    @classmethod
    def load(cls, path: str | Path | None = None) -> MicrostructureConfig:
        """Load and validate from YAML (defaults to config/microstructure.yaml)."""
        p = Path(path) if path is not None else _DEFAULT_PATH
        with p.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        return cls.model_validate(data)

    def to_dict(self) -> dict:
        """Round-trippable plain dict (dates as ISO strings, 'schema' key restored)."""
        return self.model_dump(mode="json", by_alias=True)

    def symbols(self) -> list[str]:
        return [i.symbol for i in self.instruments]

    def instrument(self, symbol: str) -> Instrument:
        for i in self.instruments:
            if i.symbol == symbol:
                return i
        raise KeyError(f"unknown instrument {symbol!r}; known: {self.symbols()}")


if __name__ == "__main__":  # pragma: no cover - manual smoke check
    cfg = MicrostructureConfig.load()
    print(
        f"microstructure_feature_version={cfg.microstructure_feature_version} "
        f"schema={cfg.schema_} symbols={cfg.symbols()} "
        f"window={cfg.window.start}..{cfg.window.end} cap=${cfg.databento_budget_usd_cap}"
    )
