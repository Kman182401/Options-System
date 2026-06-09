"""Scoring: deterministic FakeScorer + the optional local-FinBERT path (no download)."""

from __future__ import annotations

from options_system.sentiment.schema import SentimentScore
from options_system.sentiment.scoring import FakeScorer, FinbertScorer


def test_fake_scorer_deterministic_and_normalised():
    s = FakeScorer()
    a = s.score_text("stocks surge on strong earnings beat")
    b = s.score_text("stocks surge on strong earnings beat")
    assert a.positive_score == b.positive_score
    assert a.negative_score == b.negative_score
    for sc in (a, b):
        assert isinstance(sc, SentimentScore)
        assert abs((sc.positive_score + sc.negative_score + sc.neutral_score) - 1.0) < 1e-9


def test_fake_scorer_sign():
    s = FakeScorer()
    assert s.score_text("record growth, strong rally, profit surge").sentiment_score > 0
    assert s.score_text("recession fears, selloff, crash, default").sentiment_score < 0
    assert abs(s.score_text("the central bank meeting is scheduled").sentiment_score) < 1e-9


def test_finbert_available_is_safe_bool_no_download():
    # Must return a bool and must NOT attempt any network download.
    result = FinbertScorer.available("ProsusAI/finbert")
    assert isinstance(result, bool)


def test_finbert_construct_is_cheap():
    # Constructing must not load or fetch anything (lazy).
    scorer = FinbertScorer("ProsusAI/finbert")
    assert scorer.name == "ProsusAI/finbert"
    assert scorer._pipeline is None
