"""Feature config loads, validates, and round-trips (Task 1)."""

from __future__ import annotations

import pytest

from options_system.features.config import FeatureConfig


def test_loads_and_records_version():
    cfg = FeatureConfig.load()
    assert cfg.feature_version
    assert cfg.max_window() > 0
    assert cfg.cross_asset.pair == ["MES", "MNQ"]


def test_round_trips():
    cfg = FeatureConfig.load()
    assert FeatureConfig.model_validate(cfg.to_dict()) == cfg


def test_rejects_bad_macd():
    cfg = FeatureConfig.load().model_dump()
    cfg["momentum"]["macd"] = [26, 12, 9]  # fast >= slow
    with pytest.raises(ValueError):
        FeatureConfig.model_validate(cfg)


def test_rejects_non_distinct_pair():
    cfg = FeatureConfig.load().model_dump()
    cfg["cross_asset"]["pair"] = ["MES", "MES"]
    with pytest.raises(ValueError):
        FeatureConfig.model_validate(cfg)


def test_news_hook_disabled_by_default():
    assert FeatureConfig.load().news.enabled is False
