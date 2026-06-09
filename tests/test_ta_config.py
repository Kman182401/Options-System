"""TA config loads, validates, and round-trips (feature_version = v2)."""

from __future__ import annotations

import pytest

from options_system.ta.config import TaConfig


def test_loads_and_records_version():
    cfg = TaConfig.load()
    assert cfg.ta_feature_version == "v2"
    assert cfg.max_window() > 0


def test_round_trips():
    cfg = TaConfig.load()
    assert TaConfig.model_validate(cfg.to_dict()) == cfg


def test_rejects_non_positive_window():
    cfg = TaConfig.load().model_dump()
    cfg["cci"]["window"] = 0  # must be > 1
    with pytest.raises(ValueError):
        TaConfig.model_validate(cfg)


def test_rejects_unknown_key():
    cfg = TaConfig.load().model_dump()
    cfg["bogus"] = 1  # extra='forbid'
    with pytest.raises(ValueError):
        TaConfig.model_validate(cfg)
