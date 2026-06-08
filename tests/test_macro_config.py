"""Macro config (config/macro.yaml → MacroConfig) validation."""

from __future__ import annotations

import copy
from datetime import time

import pytest
import yaml

from options_system.macro.config import _DEFAULT_PATH, MacroConfig


def test_loads_and_has_expected_shape():
    cfg = MacroConfig.load()
    assert cfg.macro_version
    assert (
        len(cfg.events) >= 8
    )  # CPI, core CPI, PCE, core PCE, NFP, unrate, claims, GDP, retail, PPI
    # FOMC: 8 scheduled meetings/year over 2019-2026 (plus tentative future dates).
    assert len(cfg.fomc.decision_dates) >= 56
    assert cfg.fomc.release_et == time(14, 0)  # statement at 14:00 ET
    assert all(spec.release_et == time(8, 30) for spec in cfg.events.values())  # data at 08:30 ET


def test_event_types_include_fomc_and_high_impact_subset():
    cfg = MacroConfig.load()
    assert "fomc" in cfg.event_types()
    assert "cpi" in cfg.event_types()
    hi = cfg.high_impact_types()
    assert "fomc" in hi and "cpi" in hi
    assert "claims" not in hi  # claims is flagged low-impact in config


def test_fomc_dates_sorted_unique_enforced():
    raw = yaml.safe_load(_DEFAULT_PATH.read_text())
    bad = copy.deepcopy(raw)
    bad["fomc"]["decision_dates"] = ["2020-02-02", "2020-01-01"]  # out of order
    with pytest.raises(ValueError, match="ascending"):
        MacroConfig.model_validate(bad)


def test_feature_types_must_be_known(tmp_path):
    raw = yaml.safe_load(_DEFAULT_PATH.read_text())
    bad = copy.deepcopy(raw)
    bad["features"]["outcome_types"] = ["cpi", "not_a_real_event"]
    p = tmp_path / "macro_bad.yaml"
    p.write_text(yaml.safe_dump(bad))
    with pytest.raises(ValueError, match="unknown event types"):
        MacroConfig.load(p)
