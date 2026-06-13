"""Phase-19 sentiment A/B: opt-in sentiment block, leakage, row-set parity, fail-closed,
attribution/decision logic, and the gates tied to the mm1 config.

Pure/offline tests on the assembly + orchestration helpers — no lake, no network, no
models trained (the heavy CV is exercised by the Phase-14 suite; here we lock the NEW
Phase-19 surface).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from options_system.microstructure.bars import feature_names
from options_system.microstructure.config import MicrostructureConfig
from options_system.microstructure.model_config import MicroModelConfig
from options_system.microstructure_model.dataset import (
    _attach_features,
    _attach_sentiment,
    _finalize,
    _matrix_from_frame,
    _require_scored_sentiment,
)
from options_system.microstructure_model.evaluate import (
    VERDICT_EDGE,
    VERDICT_NONE,
    decide_verdict,
)
from options_system.microstructure_model.phase19_ab import (
    assert_identical_rows,
    attribute,
    decide,
)
from options_system.microstructure_model.phase19_config import Phase19Config
from options_system.sentiment.config import SentimentConfig
from options_system.sentiment.features import sentiment_feature_names

FEATS = feature_names(MicrostructureConfig.load())
SCFG = SentimentConfig.load()
SENT = sentiment_feature_names(SCFG)

# Two labels on 2026-02-03: one covered (t0 after the news), one before all news.
_T_COV = datetime(2026, 2, 3, 15, 30, tzinfo=UTC)
_T_NULL = datetime(2026, 2, 3, 12, 0, tzinfo=UTC)


def _scored_events() -> pl.DataFrame:
    """Three scored news events at 15:00/15:10/15:20 UTC on 2026-02-03."""
    rows = []
    for ch, topic, obs, score in (
        ("e1", "fed", "2026-02-03T15:00:00", 0.6),
        ("e2", "inflation", "2026-02-03T15:10:00", -0.4),
        ("e3", "fed", "2026-02-03T15:20:00", 0.2),
    ):
        rows.append(
            {
                "content_hash": ch,
                "source": "gdelt",
                "query_topic": topic,
                "published_at": obs,
                "observed_at": obs,
                "sentiment_feature_version": "s1",
                "positive_score": max(score, 0.0),
                "negative_score": max(-score, 0.0),
                "neutral_score": 0.2,
                "sentiment_score": score,
                "model_name": "fake-lexicon-v1",
                "model_version_or_hash": "v1",
                "scored_at": "2026-02-03T17:00:00",
            }
        )
    return pl.DataFrame(rows)


def _labels(t0s: list[datetime]) -> pl.DataFrame:
    n = len(t0s)
    return pl.DataFrame(
        {
            "t0": t0s,
            "t1": [t + timedelta(minutes=30) for t in t0s],
            "label": [1, -1, 0][:n] + [0] * max(0, n - 3),
            "ret_t1": [0.01 * (i + 1) for i in range(n)],
            "sample_weight": [1.0 + 0.1 * i for i in range(n)],
            "uniqueness_weight": [0.6 + 0.01 * i for i in range(n)],
        }
    ).with_columns(pl.col("t0").dt.cast_time_unit("us"), pl.col("t1").dt.cast_time_unit("us"))


def _bars(t0s: list[datetime]) -> pl.DataFrame:
    """One bar exactly at each label t0 (exact backward match), distinct feature values."""
    data: dict[str, object] = {"ts_event": list(t0s)}
    for j, f in enumerate(FEATS):
        data[f] = [float(i + 0.1 * j) for i in range(len(t0s))]
    return pl.DataFrame(data).with_columns(pl.col("ts_event").dt.cast_time_unit("us"))


def _build_matrix(*, with_sentiment: bool):
    """Compose the assembly helpers the way load_micro_matrix does, in-memory."""
    t0s = [_T_NULL, _T_COV]
    labels = _labels(t0s)
    bars = _bars(t0s)
    model_cols = [*FEATS, *SENT] if with_sentiment else list(FEATS)
    if with_sentiment:
        labels = _attach_sentiment(labels, _scored_events(), SCFG)
    joined = _attach_features(labels, bars, FEATS)  # row gate on FEATS only
    m, _drops = _finalize(joined, FEATS)
    keep = [*model_cols, "label", "ret_t1", "sample_weight", "uniqueness_weight", "t0", "t1"]
    frame = m.select(keep)
    mmv = "mm2" if with_sentiment else "mm1"
    return _matrix_from_frame(
        frame, "ES", model_cols, "m1", "ml1", mmv, with_sentiment=with_sentiment
    )


# --------------------------------------------------------------------------- #
# Feature block: baseline byte-identical, treatment adds exactly the sent_* cols
# --------------------------------------------------------------------------- #
def test_baseline_is_ofi_only_and_treatment_adds_only_sentiment():
    b = _build_matrix(with_sentiment=False)
    t = _build_matrix(with_sentiment=True)
    assert b.feature_cols == list(FEATS)
    assert t.feature_cols == [*FEATS, *SENT]
    assert t.X.shape[1] == len(FEATS) + len(SENT)
    # The OFI sub-block of the treatment arm is byte-identical to the baseline matrix.
    assert np.array_equal(t.X[:, : len(FEATS)], b.X, equal_nan=True)
    assert b.with_sentiment is False and t.with_sentiment is True
    assert b.micro_model_version == "mm1" and t.micro_model_version == "mm2"


def test_both_arms_identical_row_set():
    b = _build_matrix(with_sentiment=False)
    t = _build_matrix(with_sentiment=True)
    # Must not raise — same labels, same row gate ⇒ same rows.
    assert_identical_rows(b, t, "ES")
    assert b.n == t.n == 2
    assert np.array_equal(b.t0, t.t0)
    assert np.array_equal(b.y, t.y)


def test_sentiment_attach_has_no_outcome_leakage():
    """Dropping the label outcome / t1 / ret before attaching leaves sent_* identical."""
    labels = _labels([_T_NULL, _T_COV])
    full = _attach_sentiment(labels, _scored_events(), SCFG)
    stripped = _attach_sentiment(labels.drop(["label", "t1", "ret_t1"]), _scored_events(), SCFG)
    for c in SENT:
        assert full[c].to_list() == stripped[c].to_list(), f"{c} changed when outcome dropped"


def test_sentiment_nulls_are_kept_not_imputed():
    """A label before any news keeps NaN score-aggregates + 0 counts; the row survives."""
    t = _build_matrix(with_sentiment=True)
    cols = t.feature_cols
    count_col = next(c for c in SENT if c.endswith("_count"))
    mean_col = next(c for c in SENT if "mean" in c)
    ci, mi = cols.index(count_col), cols.index(mean_col)
    # Row 0 is the t0=12:00 label (before all news) — null/zero sentiment, NOT dropped.
    assert t.n == 2
    assert t.X[0, ci] == 0.0  # count is a real zero
    assert np.isnan(t.X[0, mi])  # score aggregate is NaN (no events), never imputed
    # Row 1 is covered (t0=15:30 after the news) — has events.
    assert t.X[1, ci] > 0.0


def test_matrix_build_is_deterministic():
    a = _build_matrix(with_sentiment=True)
    b = _build_matrix(with_sentiment=True)
    assert a.feature_cols == b.feature_cols
    assert np.array_equal(a.X, b.X, equal_nan=True)
    assert np.array_equal(a.t0, b.t0) and np.array_equal(a.y, b.y)


# --------------------------------------------------------------------------- #
# Fail-closed when there is no scored sentiment
# --------------------------------------------------------------------------- #
def test_treatment_fail_closed_on_empty_scores():
    empty = _scored_events().clear()
    with pytest.raises(ValueError, match="score_backfill"):
        _require_scored_sentiment(empty, "ES")
    # A non-empty scored frame does not raise.
    _require_scored_sentiment(_scored_events(), "ES")


# --------------------------------------------------------------------------- #
# Attribution + decision (frozen logic)
# --------------------------------------------------------------------------- #
def _summary(verdict: str) -> dict:
    return {"verdict": verdict}


def test_attribution_three_branches():
    edge, none = _summary(VERDICT_EDGE), _summary(VERDICT_NONE)
    assert attribute(none, edge)["attribution"] == "attributable_to_sentiment"
    assert attribute(edge, edge)["attribution"] == "not_cleanly_attributable"
    assert attribute(edge, none)["attribution"] == "null"
    assert attribute(none, none)["attribution"] == "null"


def _res(treatment_pass: bool) -> dict:
    return {"attribution": {"treatment_pass": treatment_pass, "baseline_pass": False}}


def test_decision_rule():
    both_null = decide({"ES": _res(False), "NQ": _res(False)})
    assert both_null["overall"] == "no_significant_edge" and both_null["candidates"] == []

    one = decide({"ES": _res(True), "NQ": _res(False)})
    assert one["overall"] == "sentiment_edge_candidate_fragile"
    assert one["fragile"] is True and one["candidates"] == ["ES"]

    both = decide({"ES": _res(True), "NQ": _res(True)})
    assert both["overall"] == "sentiment_edge_candidate" and both["fragile"] is False


# --------------------------------------------------------------------------- #
# Gates are the mm1 config thresholds, unchanged
# --------------------------------------------------------------------------- #
def test_verdict_gates_match_mm1_config():
    v = MicroModelConfig.load().verdict
    # An all-pass value just inside each config threshold (explicit kwargs — no dict
    # unpacking, so the mixed Optional/non-Optional signature stays type-clean).
    pbo_ok = v.max_pbo - 0.01
    dsr_ok = v.min_gross_dsr + 0.01
    mg_ok = 1e-6
    act_ok = v.min_action_rate + 0.01
    f1_ok = v.min_macro_f1 + 0.01
    cpcv_ok = 0.01

    def g(
        *,
        pbo: float | None = pbo_ok,
        gross_dsr: float | None = dsr_ok,
        mean_gross: float | None = mg_ok,
        action: float = act_ok,
        macro_f1: float = f1_ok,
        cpcv: float | None = cpcv_ok,
    ) -> tuple[str, dict[str, bool]]:
        return decide_verdict(
            pbo=pbo,
            gross_dsr=gross_dsr,
            mean_gross_return=mean_gross,
            action_rate_value=action,
            macro_f1=macro_f1,
            cpcv_median_gross_sharpe=cpcv,
            v=v,
        )

    verdict, checks = g()
    assert verdict == VERDICT_EDGE and all(checks.values())

    # Each gate flips exactly at its config threshold.
    verdict, checks = g(pbo=v.max_pbo)
    assert not checks["pbo_below_max"] and verdict == VERDICT_NONE
    verdict, checks = g(gross_dsr=v.min_gross_dsr)
    assert not checks["gross_dsr_above_min"] and verdict == VERDICT_NONE
    verdict, checks = g(mean_gross=0.0)
    assert not checks["positive_gross_return"] and verdict == VERDICT_NONE
    verdict, checks = g(action=v.min_action_rate - 1e-6)
    assert not checks["action_rate_above_min"] and verdict == VERDICT_NONE
    verdict, checks = g(macro_f1=v.min_macro_f1 - 1e-6)
    assert not checks["macro_f1_above_min"] and verdict == VERDICT_NONE
    verdict, checks = g(cpcv=0.0)
    assert not checks["cpcv_median_gross_sharpe_positive"] and verdict == VERDICT_NONE
    # A missing statistic fails its gate (absence of evidence ≠ edge).
    _, checks = g(gross_dsr=None)
    assert not checks["gross_dsr_above_min"]


# --------------------------------------------------------------------------- #
# Phase-19 config
# --------------------------------------------------------------------------- #
def test_phase19_config_loads_and_is_consistent():
    cfg = Phase19Config.load()
    assert cfg.baseline.with_sentiment is False and cfg.baseline.model_version == "mm1"
    assert cfg.treatment.with_sentiment is True and cfg.treatment.model_version == "mm2"
    assert cfg.window.start_dt() < cfg.window.end_dt()
    assert cfg.sentiment_feature_version == SCFG.aggregation.feature_version  # s2 pinned


def test_phase19_config_rejects_two_baselines():
    bad = {
        "phase19_version": "p19",
        "window": {"start": "2026-03-10", "end": "2026-06-06"},
        "symbols": ["ES"],
        "arms": [
            {"name": "a", "with_sentiment": False, "model_version": "mm1"},
            {"name": "b", "with_sentiment": False, "model_version": "mm1b"},
        ],
        "sentiment_feature_version": "s2",
        "mlflow_experiment": "x",
    }
    with pytest.raises(ValueError, match="exactly one baseline"):
        Phase19Config.model_validate(bad)
