"""Point-in-time sentiment feature aggregation tests (fixture/offline, no network)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl

from options_system.sentiment.config import SentimentConfig
from options_system.sentiment.features import (
    build_sentiment_features_for_times,
    sentiment_feature_names,
)
from options_system.sentiment.lake import _SCORED_SCHEMA

CFG = SentimentConfig.load()
T = datetime(2026, 2, 3, 15, 30, tzinfo=UTC)


def _row(
    content_hash: str,
    source: str,
    topic: str,
    observed_at: datetime,
    sentiment: float | None,
    *,
    pos: float | None = None,
    neg: float | None = None,
    neu: float | None = None,
    published_at: datetime | None = None,
    degraded: bool = False,
    model: str = "fake-lexicon-v1",
) -> dict:
    return {
        "content_hash": content_hash,
        "source": source,
        "query_topic": topic,
        "published_at": published_at or observed_at,
        "observed_at": observed_at,
        "positive_score": pos,
        "negative_score": neg,
        "neutral_score": neu,
        "sentiment_score": sentiment,
        "model_name": model,
        "scored_at": datetime(2026, 2, 3, 17, 0, tzinfo=UTC),
        "degraded": degraded,
    }


def _scored(rows: list[dict]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(schema={**_SCORED_SCHEMA, "degraded": pl.Boolean})
    return pl.DataFrame(rows)


def _build(rows: list[dict], times: list[datetime] | None = None) -> pl.DataFrame:
    return build_sentiment_features_for_times(_scored(rows), times or [T], CFG)


# --- 1. point-in-time aggregation ------------------------------------------- #


def test_event_exactly_at_target_is_included():
    out = _build([_row("h1", "gdelt", "fed", T, 0.6, pos=0.7, neg=0.1, neu=0.2)])
    assert out["sent_15m_count"][0] == 1
    assert out["sent_15m_has_any"][0] == 1
    assert abs(out["sent_15m_mean_score"][0] - 0.6) < 1e-9


def test_event_observed_after_target_is_excluded():
    out = _build(
        [_row("h1", "gdelt", "fed", T + timedelta(minutes=1), 0.6, pos=0.7, neg=0.1, neu=0.2)]
    )
    assert out["sent_15m_count"][0] == 0
    assert out["sent_1d_count"][0] == 0
    assert out["sent_15m_has_any"][0] == 0
    assert out["sent_15m_mean_score"][0] is None


def test_published_before_but_observed_after_is_excluded():
    # observed_at (not published_at) governs causality: published earlier, observed later.
    row = _row(
        "h1",
        "gdelt",
        "fed",
        T + timedelta(minutes=1),
        0.6,
        pos=0.7,
        neg=0.1,
        neu=0.2,
        published_at=T - timedelta(hours=1),
    )
    out = _build([row])
    assert out["sent_1d_count"][0] == 0  # not knowable at T despite early publish


# --- 2. window boundaries ---------------------------------------------------- #


def test_event_at_exact_window_edge_excluded_but_present_in_wider_window():
    row = _row("h1", "gdelt", "fed", T - timedelta(minutes=15), 0.6, pos=0.7, neg=0.1, neu=0.2)
    out = _build([row])
    assert out["sent_15m_count"][0] == 0  # exactly aged out of the 15m half-open window
    assert out["sent_60m_count"][0] == 1  # still inside the 60m window


def test_event_just_inside_window_included():
    row = _row("h1", "gdelt", "fed", T - timedelta(minutes=14), 0.6, pos=0.7, neg=0.1, neu=0.2)
    out = _build([row])
    assert out["sent_15m_count"][0] == 1


# --- 3. missing behavior ----------------------------------------------------- #


def test_no_prior_events_counts_zero_scores_null_flag_zero_row_preserved():
    out = _build([], times=[T])
    assert out.height == 1  # the label/target row is preserved
    assert out["sent_15m_count"][0] == 0
    assert out["sent_15m_degraded_count"][0] == 0
    assert out["sent_15m_mean_score"][0] is None
    assert out["sent_15m_sum_score"][0] is None
    assert out["sent_15m_max_abs_score"][0] is None
    assert out["sent_15m_latest_age_min"][0] is None
    assert out["sent_15m_has_any"][0] == 0
    assert out["sentiment_feature_version"][0] == CFG.aggregation.feature_version


def test_degraded_event_counted_separately_not_in_score_aggregates():
    rows = [
        _row("h1", "gdelt", "fed", T - timedelta(minutes=5), 0.6, pos=0.7, neg=0.1, neu=0.2),
        _row("h2", "gdelt", "fed", T - timedelta(minutes=5), None, degraded=True),
    ]
    out = _build(rows)
    assert out["sent_15m_count"][0] == 1  # only the non-degraded event
    assert out["sent_15m_degraded_count"][0] == 1
    assert abs(out["sent_15m_mean_score"][0] - 0.6) < 1e-9


# --- 4. dedup ---------------------------------------------------------------- #


def test_duplicate_hash_same_model_counted_once():
    e = _row("dup", "gdelt", "fed", T - timedelta(minutes=5), 0.6, pos=0.7, neg=0.1, neu=0.2)
    out = _build([e, dict(e)])
    assert out["sent_15m_count"][0] == 1


def test_same_hash_different_model_counted_once():
    a = _row(
        "dup", "gdelt", "fed", T - timedelta(minutes=5), 0.6, pos=0.7, neg=0.1, neu=0.2, model="m_a"
    )
    b = _row(
        "dup",
        "gdelt",
        "fed",
        T - timedelta(minutes=5),
        -0.2,
        pos=0.2,
        neg=0.4,
        neu=0.4,
        model="m_b",
    )
    out = _build([a, b])
    assert out["sent_15m_count"][0] == 1  # one headline, not double-counted across models


# --- 5. stable feature names ------------------------------------------------- #


def test_feature_names_deterministic_and_match_output_columns():
    names = sentiment_feature_names(CFG)
    assert names == sentiment_feature_names(CFG)
    out = _build([_row("h1", "gdelt", "fed", T, 0.6, pos=0.7, neg=0.1, neu=0.2)])
    assert out.columns == ["target_time", *names, "sentiment_feature_version"]


def test_unknown_topic_and_source_do_not_create_columns_but_count_globally():
    # "earnings"/"reuters" are not in the curated breakdown lists -> no own columns, but
    # the event still contributes to the global all-sources/all-topics aggregate.
    row = _row(
        "h1", "reuters", "earnings", T - timedelta(minutes=1), 0.3, pos=0.4, neg=0.1, neu=0.5
    )
    out = _build([row])
    assert not any("earnings" in c for c in out.columns)
    assert not any("reuters" in c for c in out.columns)
    assert out["sent_15m_count"][0] == 1  # counted in the global group


# --- 10. network guard ------------------------------------------------------- #


def test_aggregation_never_touches_network(monkeypatch):
    import urllib.request

    from options_system.sentiment import gdelt, sec_edgar

    def _boom(*_a, **_k):
        raise AssertionError("network access attempted during offline aggregation")

    monkeypatch.setattr(urllib.request, "urlopen", _boom)
    monkeypatch.setattr(gdelt, "fetch_artlist", _boom)
    monkeypatch.setattr(sec_edgar, "fetch_submissions", _boom)
    out = _build([_row("h1", "gdelt", "fed", T, 0.6, pos=0.7, neg=0.1, neu=0.2)])
    assert out.height == 1
