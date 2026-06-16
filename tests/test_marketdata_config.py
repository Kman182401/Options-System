"""MarketDataConfig validation — offline."""

from __future__ import annotations

import pytest

from options_system.marketdata.config import MarketDataConfig


def test_loads_default():
    cfg = MarketDataConfig.load()
    assert cfg.marketdata_feature_version == "x1"
    assert "vix" in {s.label for s in cfg.series}
    assert cfg.label_for("VIXCLS") == "vix"


def test_rejects_duplicate_labels():
    cfg = MarketDataConfig.load()
    data = cfg.model_dump(mode="json")
    data["series"][1]["label"] = data["series"][0]["label"]  # force a dup label
    with pytest.raises(ValueError, match="labels must be unique"):
        MarketDataConfig.model_validate(data)


def test_rejects_curve_pair_unknown_label():
    cfg = MarketDataConfig.load()
    data = cfg.model_dump(mode="json")
    data["features"]["curve_pairs"] = [["ust_10y", "nonexistent"]]
    with pytest.raises(ValueError, match="unknown label"):
        MarketDataConfig.model_validate(data)


def test_rejects_policy_disagreement():
    cfg = MarketDataConfig.load()
    data = cfg.model_dump(mode="json")
    data["source_policy"] = {"fred": "free_no_auth"}  # wrong: fred is FREE_AUTH
    with pytest.raises(ValueError, match="disagrees with the authoritative"):
        MarketDataConfig.model_validate(data)
