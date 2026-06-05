"""Labeling config loads, validates, round-trips, and records its version (Task 1)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from options_system.labeling.config import LabelConfig


def test_loads_and_records_version():
    cfg = LabelConfig.load()
    assert cfg.label_version  # non-empty
    assert cfg.barriers.pt_mult > 0 and cfg.barriers.sl_mult > 0
    assert cfg.barriers.max_hold_bars > 0
    assert cfg.volatility.barrier_horizon_bars > 0
    assert cfg.events.method in {"cusum", "grid"}
    assert cfg.roll.handling in {"adjust", "close"}
    assert cfg.weights.scheme in {"uniqueness", "uniqueness_return"}


def test_round_trips():
    cfg = LabelConfig.load()
    again = LabelConfig.model_validate(cfg.to_dict())
    assert again == cfg


def test_rejects_unknown_event_method():
    cfg = LabelConfig.load().to_dict()
    cfg["events"]["method"] = "magic"
    with pytest.raises(ValidationError):
        LabelConfig.model_validate(cfg)


def test_rejects_unknown_roll_handling():
    cfg = LabelConfig.load().to_dict()
    cfg["roll"]["handling"] = "teleport"
    with pytest.raises(ValidationError):
        LabelConfig.model_validate(cfg)


def test_rejects_extra_keys():
    cfg = LabelConfig.load().to_dict()
    cfg["surprise"] = 1
    with pytest.raises(ValidationError):
        LabelConfig.model_validate(cfg)


def test_rejects_negative_barrier():
    cfg = LabelConfig.load().to_dict()
    cfg["barriers"]["pt_mult"] = -1.0
    with pytest.raises(ValidationError):
        LabelConfig.model_validate(cfg)


def test_time_decay_bounds():
    cfg = LabelConfig.load().to_dict()
    cfg["weights"]["time_decay"] = 2.0  # out of [-1, 1]
    with pytest.raises(ValidationError):
        LabelConfig.model_validate(cfg)
