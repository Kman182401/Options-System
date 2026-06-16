"""Cross-asset (x1) feature tests — point-in-time leakage is the headline case."""

from __future__ import annotations

from datetime import UTC, date, datetime

import numpy as np
import polars as pl
import pytest

from options_system.marketdata.config import MarketDataConfig
from options_system.marketdata.features import (
    _change,
    _level,
    _SeriesArrays,
    _zscore,
    attach_market_asof,
    build_market_features_for_times,
    market_feature_names,
)
from options_system.marketdata.lake import build_rows, pit_observed_at

INGEST = datetime(2026, 1, 1, tzinfo=UTC)


def _vix_frame(vals: dict[date, float]) -> pl.DataFrame:
    obs = pl.DataFrame({"obs_date": list(vals), "value": [float(v) for v in vals.values()]}).sort(
        "obs_date"
    )
    return build_rows("VIXCLS", obs, ingested_at=INGEST)


# --- helpers ------------------------------------------------------------------ #


def test_asof_index_and_helpers():
    obs = np.array([10, 20, 30], dtype=np.int64)
    val = np.array([1.0, 2.0, 4.0], dtype=np.float64)
    arr = _SeriesArrays(obs, val)
    assert arr.asof_index(25) == 1  # latest <= 25 is index 1
    assert arr.asof_index(5) == -1  # nothing known yet
    assert _level(arr, 2) == 4.0
    assert _level(arr, -1) is None
    assert _change(arr, 2, 1) == 2.0  # 4 - 2
    assert _change(arr, 1, 5) is None  # not enough history
    assert _zscore(arr, 2, 3) is not None
    assert _zscore(arr, 0, 3) is None  # window longer than history


def test_zscore_constant_series_is_zero():
    arr = _SeriesArrays(np.array([1, 2, 3], np.int64), np.array([5.0, 5.0, 5.0], np.float64))
    assert _zscore(arr, 2, 3) == 0.0


# --- the leakage guarantee ---------------------------------------------------- #


def test_level_never_uses_a_same_or_future_day_value():
    cfg = MarketDataConfig.load()
    frame = _vix_frame({date(2024, 1, 2): 10.0, date(2024, 1, 3): 20.0, date(2024, 1, 4): 15.0})

    # A label at NOON on Jan 3: Jan-3's value is observed at END of Jan 3 (23:59:59),
    # so it must NOT be visible yet — the level must be Jan-2's 10.0, never 20.0.
    feats = build_market_features_for_times(frame, [datetime(2024, 1, 3, 12, tzinfo=UTC)], cfg)
    assert feats["mkt_vix_level"][0] == 10.0

    # At noon Jan 4, the latest knowable value is Jan-3's 20.0; the 1-day change is 20-10.
    feats2 = build_market_features_for_times(frame, [datetime(2024, 1, 4, 12, tzinfo=UTC)], cfg)
    assert feats2["mkt_vix_level"][0] == 20.0
    assert feats2["mkt_vix_chg_1d"][0] == 10.0


def test_observed_at_is_end_of_day():
    assert pit_observed_at(date(2024, 1, 2)) == datetime(2024, 1, 2, 23, 59, 59, tzinfo=UTC)


def test_missing_series_is_null_not_zero():
    cfg = MarketDataConfig.load()
    frame = _vix_frame({date(2024, 1, 2): 10.0})
    feats = build_market_features_for_times(frame, [datetime(2024, 6, 1, tzinfo=UTC)], cfg)
    # vxn has no data in this frame -> null level (not a fabricated 0).
    assert feats["mkt_vxn_level"][0] is None
    # before VIX's first obs -> null too.
    pre = build_market_features_for_times(frame, [datetime(2023, 1, 1, tzinfo=UTC)], cfg)
    assert pre["mkt_vix_level"][0] is None


def test_curve_spread_is_long_minus_short():
    cfg = MarketDataConfig.load()
    rows = pl.concat(
        [
            build_rows(
                "DGS10",
                pl.DataFrame({"obs_date": [date(2024, 1, 2)], "value": [4.5]}),
                ingested_at=INGEST,
            ),
            build_rows(
                "DGS2",
                pl.DataFrame({"obs_date": [date(2024, 1, 2)], "value": [4.0]}),
                ingested_at=INGEST,
            ),
        ]
    )
    feats = build_market_features_for_times(rows, [datetime(2024, 1, 3, tzinfo=UTC)], cfg)
    assert feats["mkt_curve_ust_10y_ust_2y"][0] == pytest.approx(0.5)


def test_feature_names_deterministic_and_stamped():
    cfg = MarketDataConfig.load()
    names = market_feature_names(cfg)
    assert names[0] == "mkt_vix_level"
    assert "mkt_vix_chg_5d" in names and f"mkt_vix_z_{cfg.features.zscore_window_days}d" in names
    feats = build_market_features_for_times(
        _vix_frame({date(2024, 1, 2): 10.0}), [datetime(2024, 1, 4, tzinfo=UTC)], cfg
    )
    assert feats["marketdata_feature_version"][0] == cfg.marketdata_feature_version
    assert set(names).issubset(feats.columns)


def test_attach_preserves_rows_and_aligns():
    cfg = MarketDataConfig.load()
    frame = _vix_frame({date(2024, 1, 2): 10.0, date(2024, 1, 3): 20.0})
    labels = pl.DataFrame(
        {
            "t0": [datetime(2024, 1, 4, tzinfo=UTC), datetime(2024, 1, 3, 12, tzinfo=UTC)],
            "y": [1, 0],
        }
    )
    out = attach_market_asof(labels, frame, cfg)
    assert out.height == 2 and "y" in out.columns
    assert out["mkt_vix_level"].to_list() == [20.0, 10.0]


def test_empty_target_times():
    cfg = MarketDataConfig.load()
    out = build_market_features_for_times(_vix_frame({date(2024, 1, 2): 10.0}), [], cfg)
    assert out.height == 0 and "mkt_vix_level" in out.columns
