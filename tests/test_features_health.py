"""Features-health view core (gather_feature_health) renders against the lake."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from options_system.data.store import DuckStore
from options_system.features.build import read_features
from options_system.features.compute import feature_names
from options_system.features.config import FeatureConfig


def test_gather_feature_health_summarizes_lake():
    cfg = FeatureConfig.load()
    store = DuckStore()
    try:
        # require features to have been built for the window (Task 3 wrote 2026-05+)
        probe = read_features(
            "MES", datetime(2026, 5, 1, tzinfo=UTC), datetime(2026, 6, 5, tzinfo=UTC), store=store
        )
        if probe.is_empty():
            pytest.skip("no features in lake (run features.build first)")
        from options_system.observability.features_health import gather_feature_health

        health = gather_feature_health(
            store, ["MES"], cfg, as_of=datetime(2026, 6, 5, tzinfo=UTC), lookback_days=40
        )
    finally:
        store.close()

    info = next(h for h in health if h["symbol"] == "MES")
    assert info["rows"] > 0
    assert info["feature_version"] == ["v1"]
    assert info["last_ts"] is not None
    # every feature has a null-rate entry, all within [0, 1]
    assert set(info["null_rate"]) == set(feature_names(cfg))
    assert all(0.0 <= r <= 1.0 for r in info["null_rate"].values())
