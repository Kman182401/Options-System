"""Point-in-time label-join tests for sentiment features (offline, no network)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl

from options_system.observability.sentiment_health import gather_sentiment_feature_health
from options_system.sentiment.config import SentimentConfig
from options_system.sentiment.features import sentiment_feature_names
from options_system.sentiment.join import attach_to_daily_labels, attach_to_micro_labels

CFG = SentimentConfig.load()
T0_A = datetime(2026, 2, 3, 15, 30, tzinfo=UTC)  # has prior sentiment
T0_B = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)  # no prior sentiment


def _scored() -> pl.DataFrame:
    rows = [
        {
            "content_hash": "e1",
            "source": "gdelt",
            "query_topic": "fed",
            "published_at": T0_A - timedelta(minutes=10),
            "observed_at": T0_A - timedelta(minutes=10),
            "positive_score": 0.7,
            "negative_score": 0.1,
            "neutral_score": 0.2,
            "sentiment_score": 0.6,
            "model_name": "fake-lexicon-v1",
            "scored_at": datetime(2026, 2, 3, 17, tzinfo=UTC),
        },
        {
            "content_hash": "e2",
            "source": "sec_edgar",
            "query_topic": "inflation",
            "published_at": T0_A - timedelta(hours=2),
            "observed_at": T0_A - timedelta(hours=2),
            "positive_score": 0.2,
            "negative_score": 0.5,
            "neutral_score": 0.3,
            "sentiment_score": -0.3,
            "model_name": "fake-lexicon-v1",
            "scored_at": datetime(2026, 2, 3, 17, tzinfo=UTC),
        },
    ]
    return pl.DataFrame(rows)


def _micro_labels() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "t0": [T0_A, T0_B],
            "t1": [T0_A + timedelta(minutes=30), T0_B + timedelta(minutes=30)],
            "symbol": ["ES", "ES"],
            "label": [1, -1],
            "ret_t1": [0.0012, -0.0008],
        }
    )


def _daily_labels() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "t0": [T0_A, T0_B],
            "t1": [T0_A + timedelta(days=2), T0_B + timedelta(days=2)],
            "symbol": ["ES", "ES"],
            "ret": [0.0031, -0.0021],
            "label": [1, -1],
            "barrier": ["upper", "lower"],
        }
    )


# --- 6. micro-label join ----------------------------------------------------- #


def test_micro_join_uses_t0_preserves_rows_and_stamps_version():
    attached, cov = attach_to_micro_labels(_micro_labels(), _scored(), CFG)
    assert attached.height == 2  # no rows dropped
    assert "sent_1d_count" in attached.columns
    assert attached["sentiment_feature_version"][0] == CFG.aggregation.feature_version
    # outcome columns survive untouched (passengers, not inputs)
    assert attached["ret_t1"].to_list() == [0.0012, -0.0008]
    assert cov["rows"] == 2
    assert cov["rows_with_any_sentiment"] == 1
    assert abs(cov["coverage_rate"] - 0.5) < 1e-9


def test_micro_join_features_independent_of_label_outcome_columns():
    # Dropping t1/ret_t1/label must not change any sentiment feature -> they are not inputs.
    full = _micro_labels()
    minimal = full.select("t0")
    a_full, _ = attach_to_micro_labels(full, _scored(), CFG)
    a_min, _ = attach_to_micro_labels(minimal, _scored(), CFG)
    names = sentiment_feature_names(CFG)
    assert a_full.select(names).equals(a_min.select(names))


def test_micro_join_label_with_no_prior_sentiment_is_kept():
    attached, _ = attach_to_micro_labels(_micro_labels(), _scored(), CFG)
    # row 1 is T0_B (no prior sentiment) -> kept with zero counts / null score
    assert attached["sent_1d_count"][1] == 0
    assert attached["sent_1d_mean_score"][1] is None


# --- 7. daily-label join ----------------------------------------------------- #


def test_daily_join_uses_t0_and_does_not_leak_returns():
    full = _daily_labels()
    minimal = full.select("t0")
    a_full, cov = attach_to_daily_labels(full, _scored(), CFG)
    a_min, _ = attach_to_daily_labels(minimal, _scored(), CFG)
    names = sentiment_feature_names(CFG)
    assert a_full.select(names).equals(a_min.select(names))  # ret/label/t1 not used
    assert a_full.height == 2
    assert cov["rows_with_any_sentiment"] == 1


# --- 8. coverage summary ----------------------------------------------------- #


def test_coverage_summary_metrics():
    _, cov = attach_to_micro_labels(_micro_labels(), _scored(), CFG)
    assert cov["events_used"] == 2  # both events land inside T0_A's 1d window
    assert cov["scored_rows"] == 2
    assert cov["duplicate_count"] == 0
    assert cov["degraded_count"] == 0
    assert cov["observed_at"]["min"] is not None
    assert cov["observed_at"]["max"] is not None
    assert cov["label_time"]["min"] is not None
    assert cov["feature_version"] == CFG.aggregation.feature_version
    assert cov["windows"] == list(CFG.aggregation.windows)
    # 15m window: only e1 (10m before T0_A) qualifies; e2 (2h) does not.
    assert cov["coverage_by_window"]["15m"] == 1
    assert cov["coverage_by_window"]["1d"] == 1


def test_feature_health_summary():
    attached, _ = attach_to_micro_labels(_micro_labels(), _scored(), CFG)
    health = gather_sentiment_feature_health(attached, CFG)
    assert health["feature_rows"] == 2
    assert health["feature_columns"] == len(sentiment_feature_names(CFG))
    assert health["feature_columns_stable"] is True
    assert abs(health["coverage_rate"] - 0.5) < 1e-9
    assert health["non_null_score_by_window"]["1d"] == 1
    assert health["coverage_by_source"]["gdelt"] == 1
