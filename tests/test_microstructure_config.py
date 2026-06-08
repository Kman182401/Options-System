"""MicrostructureConfig loads, validates, and rejects bad inputs."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from options_system.microstructure.config import MicrostructureConfig


def test_default_config_loads():
    cfg = MicrostructureConfig.load()
    assert cfg.microstructure_feature_version == "m1"
    assert cfg.dataset == "GLBX.MDP3"
    assert cfg.schema_ == "mbp-1"  # cheap top-of-book stage; MBP-10 is a later escalation
    assert set(cfg.symbols()) == {"ES", "NQ"}
    es = cfg.instrument("ES")
    assert es.multiplier == 50.0 and es.tick_size == 0.25
    assert es.dollar_threshold > 0
    assert cfg.window.end > cfg.window.start
    assert cfg.databento_budget_usd_cap > 0
    assert cfg.ofi.rolling_bars >= 1


def test_to_dict_restores_schema_key_and_roundtrips():
    cfg = MicrostructureConfig.load()
    d = cfg.to_dict()
    assert d["schema"] == "mbp-1"  # alias restored, not "schema_"
    again = MicrostructureConfig.model_validate(d)
    assert again.microstructure_feature_version == cfg.microstructure_feature_version


def test_unknown_instrument_raises():
    cfg = MicrostructureConfig.load()
    with pytest.raises(KeyError):
        cfg.instrument("ZZ")


def _base_dict() -> dict:
    return MicrostructureConfig.load().to_dict()


def test_nonpositive_budget_cap_rejected():
    d = _base_dict()
    d["databento_budget_usd_cap"] = 0
    with pytest.raises(ValidationError):
        MicrostructureConfig.model_validate(d)


def test_window_end_must_follow_start():
    d = _base_dict()
    d["window"]["end"] = d["window"]["start"]
    with pytest.raises(ValidationError):
        MicrostructureConfig.model_validate(d)


def test_rth_close_after_open():
    d = _base_dict()
    d["session"]["rth_close_min"] = d["session"]["rth_open_min"]
    with pytest.raises(ValidationError):
        MicrostructureConfig.model_validate(d)


def test_duplicate_symbols_rejected():
    d = _base_dict()
    d["instruments"].append(dict(d["instruments"][0]))
    with pytest.raises(ValidationError):
        MicrostructureConfig.model_validate(d)


def test_extra_keys_forbidden():
    d = _base_dict()
    d["surprise"] = 1
    with pytest.raises(ValidationError):
        MicrostructureConfig.model_validate(d)
