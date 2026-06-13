"""Phase-20 meta-labeling: primary side + meta-label, gating, meta-skill, the binary
fold-local weighting, the six gates (five mm1 + meta-skill), leak-safety, determinism,
attribution/decision, and fail-closed sentiment.

Pure/offline tests on the meta primitives + a SMALL real binary-CV exercise (seed 7,
deterministic) — no lake, no network, no spend.
"""

from __future__ import annotations

import numpy as np
import pytest

from options_system.microstructure.bars import feature_names
from options_system.microstructure.config import MicrostructureConfig
from options_system.microstructure.model_config import MicroModelConfig
from options_system.microstructure_model.dataset import (
    MicroTrainingMatrix,
    _require_scored_sentiment,
)
from options_system.microstructure_model.evaluate import VERDICT_EDGE, VERDICT_NONE
from options_system.microstructure_model.lgbm import fold_local_class_weights
from options_system.microstructure_model.meta_labeling import (
    gated_position,
    meta_label,
    meta_set_mask,
    primary_side,
)
from options_system.microstructure_model.meta_lgbm import META_CLASSES
from options_system.microstructure_model.phase20_config import Phase20Config
from options_system.microstructure_model.phase20_meta import (
    MetaMatrix,
    attribute,
    build_meta_matrix,
    decide,
    decide_meta_verdict,
    evaluate_b0,
    evaluate_meta_arm,
    gated_gross_metrics,
    kfold_oos_proba,
    meta_skill_metrics,
    run_meta_search,
)
from options_system.validation.config import ValidationConfig

FEATS = feature_names(MicrostructureConfig.load())
MMCFG = MicroModelConfig.load()
VCFG = ValidationConfig.load()
P20 = Phase20Config.load()


# --------------------------------------------------------------------------- #
# Primary side rule
# --------------------------------------------------------------------------- #
def test_primary_side_sign_and_zero_exclusion():
    ofi = np.array([2.5, -0.1, 0.0, -7.0, 1e-9, np.nan, np.inf])
    side = primary_side(ofi)
    assert side.tolist() == [1, -1, 0, -1, 1, 0, 0]  # 0/NaN/inf -> no side
    mask = meta_set_mask(side)
    assert mask.tolist() == [True, True, False, True, True, False, False]


def test_primary_side_is_deterministic():
    ofi = np.array([0.3, -0.2, 0.0, 5.0])
    assert np.array_equal(primary_side(ofi), primary_side(ofi))


# --------------------------------------------------------------------------- #
# Meta-label across every (label, side) combination incl. the 0-timeout
# --------------------------------------------------------------------------- #
def test_meta_label_all_combinations():
    # side +1: correct only when label == +1; timeout(0) and wrong(-1) -> 0.
    assert meta_label(np.array([1, 0, -1]), np.array([1, 1, 1])).tolist() == [1, 0, 0]
    # side -1: correct only when label == -1; timeout(0) and wrong(+1) -> 0.
    assert meta_label(np.array([-1, 0, 1]), np.array([-1, -1, -1])).tolist() == [1, 0, 0]


def test_meta_label_reads_only_label_not_returns():
    """meta_label is a function of (label, side) ONLY — no return/t1 input exists."""
    label = np.array([1, -1, 0, 1])
    side = np.array([1, -1, 1, -1])
    assert meta_label(label, side).tolist() == [1, 1, 0, 0]


# --------------------------------------------------------------------------- #
# Gated position (0 when not acting)
# --------------------------------------------------------------------------- #
def test_gated_position_acts_only_above_tau():
    side = np.array([1, -1, 1, -1])
    p = np.array([0.9, 0.9, 0.4, 0.5])  # tau=0.5 strict >
    pos = gated_position(side, p, 0.5)
    assert pos.tolist() == [1.0, -1.0, 0.0, 0.0]  # row3 0.5 not > 0.5 -> flat


# --------------------------------------------------------------------------- #
# Meta-skill metrics
# --------------------------------------------------------------------------- #
def test_meta_skill_acted_beats_always_when_gate_is_selective():
    # 6 events; primary correct on 3 (meta_label=1). A selective gate that acts on the
    # 3 correct + 1 wrong -> acted_hit 3/4=0.75 > always 3/6=0.5.
    y = np.array([1, 1, 1, 0, 0, 0])
    proba = np.array([0.9, 0.9, 0.9, 0.9, 0.1, 0.1])  # acts on 3 correct + 1 wrong
    m = meta_skill_metrics(y, proba, 0.5)
    assert m["always_act_hit_rate"] == 0.5
    assert m["acted_hit_rate"] == 0.75
    assert m["acted_hit_beats_always"] is True
    assert m["n_acted"] == 4


def test_meta_skill_always_act_does_not_beat_itself():
    y = np.array([1, 1, 0, 0])
    proba = np.ones(4)  # act on everything (the B0 behaviour)
    m = meta_skill_metrics(y, proba, 0.5)
    assert m["acted_hit_rate"] == m["always_act_hit_rate"] == 0.5
    assert m["acted_hit_beats_always"] is False
    assert m["balanced_accuracy"] == 0.5  # all-act predictor


# --------------------------------------------------------------------------- #
# Gated gross metrics (no costs; 0 on flat rows)
# --------------------------------------------------------------------------- #
def test_gated_gross_metrics_flat_rows_contribute_zero():
    pos = np.array([1.0, 0.0, -1.0, 0.0])
    ret = np.array([0.01, 0.02, -0.03, 0.04])
    w = np.ones(4)
    out = gated_gross_metrics(pos, ret, w, np.array([0.1]))
    # gross = [0.01, 0, 0.03, 0]; mean over 4 = 0.04/4 = 0.01
    assert out["n_scored"] == 4
    assert out["mean_gross_return"] == pytest.approx(0.01, abs=1e-9)


# --------------------------------------------------------------------------- #
# Six gates: five from mm1 config, the meta-skill substitute for macro-F1
# --------------------------------------------------------------------------- #
def test_meta_verdict_gates_use_mm1_and_phase20_thresholds():
    v = MMCFG.verdict
    floor = P20.meta_skill_min_balanced_accuracy
    ok = dict(
        pbo=v.max_pbo - 0.01,
        gross_dsr=v.min_gross_dsr + 0.01,
        mean_gross=1e-6,
        action=v.min_action_rate + 0.01,
        cpcv=0.01,
        bal=floor + 0.01,
        acted=0.6,
        always=0.5,
    )

    def g(**over):
        a = {**ok, **over}
        return decide_meta_verdict(
            pbo=a["pbo"],
            gross_dsr=a["gross_dsr"],
            mean_gross_return=a["mean_gross"],
            action_rate_value=a["action"],
            cpcv_median_gross_sharpe=a["cpcv"],
            balanced_accuracy=a["bal"],
            acted_hit_rate=a["acted"],
            always_act_hit_rate=a["always"],
            v=v,
            meta_skill_min_balanced_accuracy=floor,
        )

    verdict, checks = g()
    assert verdict == VERDICT_EDGE and all(checks.values())

    # Each inherited gross gate flips exactly at its mm1 threshold.
    assert not g(pbo=v.max_pbo)[1]["pbo_below_max"]
    assert not g(gross_dsr=v.min_gross_dsr)[1]["gross_dsr_above_min"]
    assert not g(mean_gross=0.0)[1]["positive_gross_return"]
    assert not g(action=v.min_action_rate - 1e-6)[1]["action_rate_above_min"]
    assert not g(cpcv=0.0)[1]["cpcv_median_gross_sharpe_positive"]
    # Meta-skill needs BOTH balanced-acc >= floor AND acted > always.
    assert not g(bal=floor - 1e-6)[1]["meta_skill"]
    assert not g(acted=0.5, always=0.5)[1]["meta_skill"]  # not strictly greater
    # A missing statistic fails its gate.
    assert not g(pbo=None)[1]["pbo_below_max"]
    assert not g(gross_dsr=None)[1]["gross_dsr_above_min"]
    assert g(bal=None)[1]["meta_skill"] is False


# --------------------------------------------------------------------------- #
# Fold-local BINARY balanced weighting computed from y_train ONLY
# --------------------------------------------------------------------------- #
def test_fold_local_binary_weights_from_train_only():
    # Imbalanced binary fold: 8 zeros, 2 ones. Balanced weights equalise class mass.
    y_train = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
    w = np.ones(10)
    cw = fold_local_class_weights(y_train, w, META_CLASSES, use_sample_weight=True)
    # balanced: weight[c] = n / (n_classes * count[c]) -> 0:10/(2*8)=0.625, 1:10/(2*2)=2.5
    assert cw[0] == pytest.approx(0.625)
    assert cw[1] == pytest.approx(2.5)
    # A class absent from this fold gets weight 1.0 (irrelevant — no rows here).
    cw_one = fold_local_class_weights(np.zeros(5, dtype=int), np.ones(5), META_CLASSES)
    assert cw_one[1] == 1.0


# --------------------------------------------------------------------------- #
# build_meta_matrix: exclusion, abs(ofi_top) column, leak-safety
# --------------------------------------------------------------------------- #
def _micro_matrix(ofi_vals: list[float], labels: list[int]) -> MicroTrainingMatrix:
    n = len(ofi_vals)
    p = len(FEATS)
    X = np.zeros((n, p), dtype=float)
    ofi_idx = list(FEATS).index("ofi_top")
    X[:, ofi_idx] = ofi_vals
    t0 = np.array(
        [np.datetime64("2026-04-01T14:00:00") + np.timedelta64(i, "m") for i in range(n)],
        dtype="datetime64[us]",
    )
    t1 = t0 + np.timedelta64(30, "m")
    return MicroTrainingMatrix(
        symbol="ES",
        X=X,
        y=np.array(labels, dtype=int),
        t0=t0,
        t1=t1,
        ret_t1=np.array([0.01 * (i + 1) for i in range(n)], dtype=float),
        sample_weight=np.ones(n),
        uniqueness_weight=np.full(n, 0.5),
        feature_cols=list(FEATS),
        microstructure_feature_version="m1",
        micro_label_version="ml1",
        micro_model_version="mm2",
        with_sentiment=True,
    )


def test_build_meta_matrix_excludes_no_side_and_adds_abs_ofi():
    # ofi_top: +,-,0,+ -> the 0 row is excluded; n_excluded == 1.
    mtm = _micro_matrix([2.0, -3.0, 0.0, 4.0], [1, -1, 0, 1])
    mm = build_meta_matrix(mtm, P20)
    assert mm.n == 3 and mm.n_excluded == 1
    assert mm.feature_cols[-1] == "abs_ofi_top"
    assert mm.n_features == len(FEATS) + 1
    # side = sign(ofi_top) on the kept rows.
    assert mm.side.tolist() == [1, -1, 1]
    # meta_label: row0 label1==side1 ->1; row1 label-1==side-1 ->1; row3 label1==side1 ->1.
    assert mm.y.tolist() == [1, 1, 1]
    # abs_ofi_top column = |ofi_top|.
    ofi_idx = list(FEATS).index("ofi_top")
    assert np.allclose(mm.X[:, -1], np.abs(mm.X[:, ofi_idx]))


def test_build_meta_matrix_is_leak_safe_wrt_returns_and_t1():
    """Perturbing ret_t1 / t1 must not change X, side, or the meta-label (built from
    ofi_top@t0 and the resolved label only)."""
    base = _micro_matrix([1.0, -2.0, 3.0, -4.0], [1, -1, 0, 1])
    mm1 = build_meta_matrix(base, P20)
    perturbed = MicroTrainingMatrix(
        symbol=base.symbol,
        X=base.X,
        y=base.y,
        t0=base.t0,
        t1=base.t1 + np.timedelta64(999, "m"),  # mangle t1
        ret_t1=base.ret_t1 * -50.0,  # mangle returns
        sample_weight=base.sample_weight,
        uniqueness_weight=base.uniqueness_weight,
        feature_cols=base.feature_cols,
        microstructure_feature_version="m1",
        micro_label_version="ml1",
        micro_model_version="mm2",
        with_sentiment=True,
    )
    mm2 = build_meta_matrix(perturbed, P20)
    assert np.array_equal(mm1.X, mm2.X, equal_nan=True)
    assert np.array_equal(mm1.side, mm2.side)
    assert np.array_equal(mm1.y, mm2.y)


def test_build_meta_matrix_stops_on_unknown_primary_feature(monkeypatch):
    mtm = _micro_matrix([1.0, -1.0], [1, -1])
    bad = P20.model_copy(update={"primary_feature": "not_a_real_feature"})
    with pytest.raises(ValueError, match="not an m1 feature column"):
        build_meta_matrix(mtm, bad)


# --------------------------------------------------------------------------- #
# Small REAL binary-CV exercise: determinism + B0 fails meta-skill by construction
# --------------------------------------------------------------------------- #
def _synth_meta(n: int = 96, seed: int = 0) -> MetaMatrix:
    """A small, t0-sorted meta-set with a weak real signal (both classes present)."""
    rng = np.random.default_rng(seed)
    t0 = np.array(
        [np.datetime64("2026-04-01T14:00:00") + np.timedelta64(i, "m") for i in range(n)],
        dtype="datetime64[us]",
    )
    t1 = t0 + np.timedelta64(30, "m")
    side = np.where(np.arange(n) % 2 == 0, 1, -1).astype(int)
    f1 = rng.normal(size=n)
    # y weakly depends on f1 so LightGBM has something to (try to) learn.
    y = ((f1 + rng.normal(scale=0.5, size=n)) > 0).astype(int)
    ofi = side * np.abs(rng.normal(size=n) + 0.5)
    X = np.column_stack([ofi, f1, np.abs(ofi), rng.normal(size=n)])
    return MetaMatrix(
        symbol="ES",
        X=X,
        y=y,
        side=side,
        t0=t0,
        t1=t1,
        ret_t1=rng.normal(scale=0.01, size=n),
        sample_weight=np.ones(n),
        uniqueness_weight=np.full(n, 0.5),
        feature_cols=["ofi_top", "f1", "abs_ofi_top", "sent_x"],
        n_excluded=0,
        micro_model_version="p20-M-meta-s2",
        microstructure_feature_version="m1",
        micro_label_version="ml1",
    )


def test_kfold_oos_proba_is_deterministic():
    mm = _synth_meta()
    p1, s1 = kfold_oos_proba(mm, {}, MMCFG, VCFG.kfold.n_splits, VCFG.kfold.embargo_pct)
    p2, s2 = kfold_oos_proba(mm, {}, MMCFG, VCFG.kfold.n_splits, VCFG.kfold.embargo_pct)
    assert np.array_equal(s1, s2)
    assert np.allclose(p1, p2)  # seed 7 + deterministic LightGBM -> identical


def test_fit_meta_fold_weights_use_train_subset_only():
    """fit_meta_fold computes balanced weights from y[train_idx] only; a single-class
    train fold yields no balancing (cw all 1.0) — proven via the weighting primitive it
    calls, with the binary classes."""
    mm = _synth_meta(n=60)
    # Train indices whose labels are all one class -> degenerate, cw stays 1.0.
    train_idx = np.where(mm.y == 0)[0][:20]
    cw = fold_local_class_weights(
        mm.y[train_idx], mm.sample_weight[train_idx], META_CLASSES, use_sample_weight=True
    )
    assert set(cw) == {0, 1}
    assert cw[1] == 1.0  # positive class absent from this fold


def test_meta_arm_and_b0_evaluation_structure_and_b0_fails_meta_skill():
    mm = _synth_meta()
    search = run_meta_search(mm, MMCFG, VCFG, P20.decision_threshold)
    assert search.n_trials == MMCFG.search.n_trials  # 8 inherited configs
    assert search.pbo_matrix.shape == (mm.n, search.n_trials)

    m = evaluate_meta_arm(mm, search, MMCFG, VCFG, P20)
    b0 = evaluate_b0(mm, MMCFG, VCFG, P20)
    # Six gate checks present (5 inherited gross + meta_skill; no macro_f1).
    assert set(m["verdict_checks"]) == {
        "pbo_below_max",
        "gross_dsr_above_min",
        "positive_gross_return",
        "action_rate_above_min",
        "cpcv_median_gross_sharpe_positive",
        "meta_skill",
    }
    assert m["verdict"] in (VERDICT_EDGE, VERDICT_NONE)
    # B0 always acts -> meta-skill fails by construction; action rate is 1.0.
    assert b0["action_rate"] == 1.0
    assert b0["verdict_checks"]["meta_skill"] is False
    assert b0["verdict"] == VERDICT_NONE  # no search -> PBO None -> fails too


def test_meta_search_is_deterministic():
    mm = _synth_meta()
    a = run_meta_search(mm, MMCFG, VCFG, P20.decision_threshold)
    b = run_meta_search(mm, MMCFG, VCFG, P20.decision_threshold)
    assert a.selected_overrides == b.selected_overrides
    assert np.allclose(a.gross_sr, b.gross_sr)
    assert np.allclose(a.selected_proba, b.selected_proba)


# --------------------------------------------------------------------------- #
# Attribution + decision (frozen logic)
# --------------------------------------------------------------------------- #
def test_attribute_uses_m_arm_verdict():
    edge = {"verdict": VERDICT_EDGE}
    none = {"verdict": VERDICT_NONE}
    assert attribute(none, edge) == {"b0_pass": False, "m_pass": True}
    assert attribute(edge, none) == {"b0_pass": True, "m_pass": False}


def _res(m_pass: bool) -> dict:
    return {"attribution": {"m_pass": m_pass, "b0_pass": False}}


def test_decision_rule_three_branches():
    both_null = decide({"ES": _res(False), "NQ": _res(False)})
    assert both_null["overall"] == "no_significant_edge" and both_null["candidates"] == []

    one = decide({"ES": _res(True), "NQ": _res(False)})
    assert one["overall"] == "meta_labeling_edge_candidate_fragile"
    assert one["fragile"] is True and one["candidates"] == ["ES"]

    both = decide({"ES": _res(True), "NQ": _res(True)})
    assert both["overall"] == "meta_labeling_edge_candidate" and both["fragile"] is False


# --------------------------------------------------------------------------- #
# Config + fail-closed sentiment (the meta pipeline relies on the dataset guard)
# --------------------------------------------------------------------------- #
def test_phase20_config_loads_and_is_consistent():
    from options_system.sentiment.config import SentimentConfig

    cfg = Phase20Config.load()
    assert cfg.window.start == "2026-01-26" and cfg.window.end == "2026-06-06"
    assert cfg.primary_feature == "ofi_top"
    assert cfg.decision_threshold == 0.5
    assert cfg.meta_skill_min_balanced_accuracy == 0.52
    assert cfg.sentiment_feature_version == SentimentConfig.load().aggregation.feature_version
    # supported region falls within the full window.
    assert cfg.window.start_dt() <= cfg.supported_region_start_dt() <= cfg.window.end_dt()


def test_phase20_config_rejects_supported_region_outside_window():
    bad = {
        "phase20_version": "p20",
        "window": {"start": "2026-01-26", "end": "2026-06-06"},
        "supported_region_start": "2025-01-01",  # before the window
        "symbols": ["ES"],
        "primary_feature": "ofi_top",
        "decision_threshold": 0.5,
        "meta_skill_min_balanced_accuracy": 0.52,
        "sentiment_feature_version": "s2",
        "mlflow_experiment": "x",
    }
    with pytest.raises(ValueError, match="must fall within the window"):
        Phase20Config.model_validate(bad)


def test_meta_pipeline_fail_closed_on_empty_sentiment():
    """The full-window meta matrix is a with_sentiment matrix; an empty scored lake
    fails closed at the dataset guard the pipeline depends on."""
    import polars as pl

    with pytest.raises(ValueError, match="score_backfill"):
        _require_scored_sentiment(pl.DataFrame(), "ES")


# --------------------------------------------------------------------------- #
# Canonical-verdict symbol-set guard (no partial/subset run may publish a verdict)
# --------------------------------------------------------------------------- #
def test_run_phase20_refuses_to_save_partial_symbol_verdict():
    """A subset run (only ES) must not save a canonical verdict — the pre-registered
    decision rule is over the exact {ES, NQ} set. The guard fires before any lake access."""
    from options_system.microstructure_model.phase20_meta import run_phase20

    with pytest.raises(ValueError, match="pre-registered over"):
        run_phase20(symbols=["ES"], save=True, interpret=False, log_mlflow=False)


def test_run_phase20_rejects_unknown_symbol_before_running():
    """An unknown / extra symbol set never matches the pre-registered set, so a canonical
    run is refused up front (clear message, no wasted modeling)."""
    from options_system.microstructure_model.phase20_meta import run_phase20

    with pytest.raises(ValueError, match="pre-registered over"):
        run_phase20(symbols=["ES", "NQ", "MES"], save=True, interpret=False, log_mlflow=False)
