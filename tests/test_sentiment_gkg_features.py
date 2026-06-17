"""GKG s3 daily tone aggregation — point-in-time correctness — offline."""

from __future__ import annotations

from datetime import UTC, datetime

import polars as pl
import pytest

from options_system.sentiment.gkg_config import GkgConfig
from options_system.sentiment.gkg_features import (
    attach_gkg_asof,
    build_gkg_features_for_times,
    gkg_feature_names,
)


def _scored(events: list[tuple[datetime, float, float]]) -> pl.DataFrame:
    """events = [(observed_at, tone, positive_share)] -> a minimal scored frame."""
    return pl.DataFrame(
        {
            "content_hash": [f"h{i}" for i in range(len(events))],
            "observed_at": [e[0] for e in events],
            "sentiment_score": [e[1] for e in events],
            "positive_score": [e[2] for e in events],
        }
    )


def test_feature_names_window_major():
    cfg = GkgConfig.load()
    names = gkg_feature_names(cfg)
    assert names[0] == "gkgtone_d1_count"
    assert "gkgtone_d1_mean_tone" in names and "gkgtone_d7_has_any" in names


def test_window_aggregation_and_leakage():
    cfg = GkgConfig.load()
    frame = _scored(
        [
            (datetime(2024, 1, 2, 0, 0, tzinfo=UTC), 0.1, 0.05),
            (datetime(2024, 1, 2, 12, 0, tzinfo=UTC), 0.3, 0.10),
            (datetime(2024, 1, 3, 6, 0, tzinfo=UTC), -0.2, 0.02),  # FUTURE vs the t0 below
        ]
    )
    # t0 = Jan-2 18:00. The d1 window (t0-1d, t0] includes the two Jan-2 events but NOT
    # the Jan-3 event (observed after t0) — that is the leakage guarantee.
    feats = build_gkg_features_for_times(frame, [datetime(2024, 1, 2, 18, 0, tzinfo=UTC)], cfg)
    assert feats["gkgtone_d1_count"][0] == 2
    assert feats["gkgtone_d1_has_any"][0] == 1
    assert feats["gkgtone_d1_mean_tone"][0] == pytest.approx(0.2)
    assert feats["gkgtone_d1_pos_share"][0] == pytest.approx(0.075)


def test_empty_window_is_zero_count_null_mean():
    cfg = GkgConfig.load()
    frame = _scored([(datetime(2024, 1, 2, 0, 0, tzinfo=UTC), 0.1, 0.05)])
    # Far-future target: nothing in the 1-day window.
    feats = build_gkg_features_for_times(frame, [datetime(2024, 6, 1, tzinfo=UTC)], cfg)
    assert feats["gkgtone_d1_count"][0] == 0
    assert feats["gkgtone_d1_has_any"][0] == 0
    assert feats["gkgtone_d1_mean_tone"][0] is None


def test_dedup_by_content_hash():
    cfg = GkgConfig.load()
    t = datetime(2024, 1, 2, 0, 0, tzinfo=UTC)
    frame = pl.DataFrame(
        {
            "content_hash": ["dup", "dup", "other"],  # same article re-emitted
            "observed_at": [t, t, t],
            "sentiment_score": [0.5, 0.5, -0.5],
            "positive_score": [0.1, 0.1, 0.0],
        }
    )
    feats = build_gkg_features_for_times(frame, [datetime(2024, 1, 2, 1, tzinfo=UTC)], cfg)
    assert feats["gkgtone_d1_count"][0] == 2  # dup collapsed to one


def test_empty_scored_frame():
    cfg = GkgConfig.load()
    feats = build_gkg_features_for_times(_scored([]), [datetime(2024, 1, 2, tzinfo=UTC)], cfg)
    assert feats["gkgtone_d1_count"][0] == 0 and feats["gkgtone_d1_mean_tone"][0] is None


def test_attach_preserves_rows_and_stamps_version():
    cfg = GkgConfig.load()
    frame = _scored([(datetime(2024, 1, 2, 0, 0, tzinfo=UTC), 0.1, 0.05)])
    labels = pl.DataFrame({"t0": [datetime(2024, 1, 2, 1, tzinfo=UTC)], "y": [1]})
    out = attach_gkg_asof(labels, frame, cfg)
    assert out.height == 1 and "y" in out.columns
    assert out["gkg_feature_version"][0] == cfg.aggregation.feature_version
    assert set(gkg_feature_names(cfg)).issubset(out.columns)
