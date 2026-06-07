"""ValidationConfig loads, validates, round-trips, and rejects bad input."""

from __future__ import annotations

import pytest

from options_system.validation.config import ValidationConfig


def test_default_config_loads_and_round_trips():
    cfg = ValidationConfig.load()
    assert cfg.validation_version == "v1"
    assert cfg.cpcv.test_groups < cfg.cpcv.n_groups
    assert cfg.evaluation.primary_metric in cfg.evaluation.metrics
    # to_dict is a faithful, re-loadable snapshot
    d = cfg.to_dict()
    again = ValidationConfig.model_validate(d)
    assert again == cfg


def test_unknown_walk_forward_scheme_rejected():
    cfg = ValidationConfig.load().to_dict()
    cfg["walk_forward"]["scheme"] = "sideways"
    with pytest.raises(ValueError, match="scheme"):
        ValidationConfig.model_validate(cfg)


def test_cpcv_test_groups_must_be_less_than_n_groups():
    cfg = ValidationConfig.load().to_dict()
    cfg["cpcv"]["test_groups"] = cfg["cpcv"]["n_groups"]
    with pytest.raises(ValueError, match="must be < cpcv.n_groups"):
        ValidationConfig.model_validate(cfg)


def test_primary_metric_must_be_in_metrics():
    cfg = ValidationConfig.load().to_dict()
    cfg["evaluation"]["primary_metric"] = "sharpe_ratio_annualised"
    with pytest.raises(ValueError, match="primary_metric"):
        ValidationConfig.model_validate(cfg)


def test_unknown_metric_rejected():
    cfg = ValidationConfig.load().to_dict()
    cfg["evaluation"]["metrics"] = ["accuracy", "magic"]
    with pytest.raises(ValueError, match="unknown entries"):
        ValidationConfig.model_validate(cfg)


def test_extra_key_forbidden():
    cfg = ValidationConfig.load().to_dict()
    cfg["surprise"] = 1
    with pytest.raises(ValueError):
        ValidationConfig.model_validate(cfg)
