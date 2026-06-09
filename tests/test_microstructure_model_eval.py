"""Micro-model validation discipline + honest verdict:

* purge/embargo use ``t1`` (no random CV);
* degenerate models (all-flat, majority-timeout) cannot pass the verdict;
* the run CLI parses its flags;
* the evaluation summary carries the required schema.

Synthetic data only — no lake, no Databento, no network.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np

from options_system.microstructure.bars import feature_names
from options_system.microstructure.config import MicrostructureConfig
from options_system.microstructure.model_config import MicroModelConfig
from options_system.microstructure_model import run as run_mod
from options_system.microstructure_model.dataset import MicroTrainingMatrix
from options_system.microstructure_model.evaluate import (
    VERDICT_EDGE,
    VERDICT_NONE,
    classification_metrics,
    decide_verdict,
    evaluate_micro_model,
    signal_return_metrics,
)
from options_system.microstructure_model.tune import kfold_oos, run_micro_search
from options_system.validation._purge import embargo_bars_from_pct, train_indices
from options_system.validation.config import ValidationConfig
from options_system.validation.purged_kfold import PurgedKFold

FEATS = feature_names(MicrostructureConfig.load())
_T0 = datetime(2026, 2, 2, 14, 30, tzinfo=UTC)


def _synthetic_matrix(n: int = 252, seed: int = 7) -> MicroTrainingMatrix:
    """A 3-class micro matrix with a faint signal in feature 0 and overlapping labels."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, len(FEATS)))
    # ~78% timeout (0); the rest take the sign of feature 0 plus noise.
    base = np.where(X[:, 0] + 0.5 * rng.standard_normal(n) > 0, 1, -1)
    is_event = rng.random(n) < 0.22
    y = np.where(is_event, base, 0).astype(int)
    ret = (0.5 * y + rng.standard_normal(n)) * 1e-3  # gross return loosely follows y
    t0 = np.array(
        [np.datetime64(_T0.replace(tzinfo=None)) + np.timedelta64(i, "m") for i in range(n)]
    )
    t1 = t0 + np.timedelta64(30, "m")  # 30-min labels overlap ~30 neighbours
    w = np.ones(n) + 0.05 * rng.standard_normal(n)
    uniq = np.clip(0.6 + 0.05 * rng.standard_normal(n), 0.1, 1.0)
    return MicroTrainingMatrix(
        symbol="ES",
        X=X,
        y=y,
        t0=t0,
        t1=t1,
        ret_t1=ret.astype(float),
        sample_weight=w.astype(float),
        uniqueness_weight=uniq.astype(float),
        feature_cols=list(FEATS),
        microstructure_feature_version="m1",
        micro_label_version="ml1",
        micro_model_version="mm1",
    )


# --- purge/embargo use t1, not a random split ------------------------------- #
def test_purge_uses_t1_no_overlap_train_into_test():
    mtm = _synthetic_matrix()
    embargo = embargo_bars_from_pct(0.01, mtm.n)
    cv = PurgedKFold(6, mtm.t0, mtm.t1, embargo_bars=embargo)
    for train_idx, test_idx in cv.split():
        seg_start = mtm.t0[test_idx].min()
        seg_end = mtm.t1[test_idx].max()
        # no training label may still be resolving inside the test segment window
        overlap = (mtm.t0[train_idx] <= seg_end) & (mtm.t1[train_idx] >= seg_start)
        assert not overlap.any()


def test_kfold_scores_every_sample_exactly_once():
    # A clean partition into OOS folds (impossible with a random/bootstrap split).
    mtm = _synthetic_matrix()
    mmcfg, vcfg = MicroModelConfig.load(), ValidationConfig.load()
    pred, scored = kfold_oos(mtm, {}, mmcfg, vcfg.kfold.n_splits, vcfg.kfold.embargo_pct)
    assert scored.all()
    assert set(np.unique(pred).astype(int)).issubset({-1, 0, 1})


def test_train_indices_excludes_overlapping_labels():
    # Direct check of the shared purge primitive the micro model relies on.
    t0 = np.array([np.datetime64("2026-02-02T14:30") + np.timedelta64(i, "m") for i in range(10)])
    t1 = t0 + np.timedelta64(5, "m")
    test_idx = np.array([5, 6])
    train = train_indices(t0, t1, test_idx, n=10, embargo_bars=0)
    # rows within 5 min of the test segment overlap and must be purged
    assert 5 not in train and 6 not in train
    assert 4 not in train  # t1[4] reaches into the test segment -> purged


# --- degenerate-model defenses ---------------------------------------------- #
def _thresholds():
    return MicroModelConfig.load().verdict


def test_all_flat_predictions_fail_action_rate_gate():
    pred = np.zeros(100, dtype=int)
    ret = np.full(100, 1e-3)
    sig = signal_return_metrics(pred, ret, np.ones(100), gross_sr_trials=np.array([0.0]))
    assert sig["mean_gross_return"] == 0.0  # flat earns nothing gross
    verdict, checks = decide_verdict(
        pbo=0.1,
        gross_dsr=0.9,
        mean_gross_return=sig["mean_gross_return"],
        action_rate_value=0.0,
        macro_f1=0.30,
        cpcv_median_gross_sharpe=0.5,
        v=_thresholds(),
    )
    assert checks["action_rate_above_min"] is False
    assert verdict == VERDICT_NONE


def test_majority_timeout_classifier_not_an_edge_despite_high_accuracy():
    # 80% timeout; predict-all-0 scores HIGH weighted accuracy (~0.8) — yet it must
    # NOT pass the verdict. The defending gates are action_rate (0 trades) and the
    # gross-return gates (a flat book earns nothing), not raw accuracy.
    rng = np.random.default_rng(1)
    y = np.where(rng.random(500) < 0.8, 0, np.where(rng.random(500) < 0.5, 1, -1)).astype(int)
    pred = np.zeros_like(y)
    cls = classification_metrics(y, pred, np.ones_like(y, dtype=float))
    sig = signal_return_metrics(pred, np.full(y.size, 1e-3), np.ones(y.size), np.array([0.0]))
    assert cls["weighted_accuracy"] >= 0.70  # high — accuracy alone looks "good"
    assert cls["action_rate"] == 0.0  # but it never actually trades
    assert sig["mean_gross_return"] == 0.0  # and a flat book earns nothing gross
    verdict, checks = decide_verdict(
        pbo=0.1,
        gross_dsr=0.9,
        mean_gross_return=sig["mean_gross_return"],
        action_rate_value=cls["action_rate"],
        macro_f1=cls["macro_f1"],
        cpcv_median_gross_sharpe=0.5,
        v=_thresholds(),
    )
    assert verdict == VERDICT_NONE
    assert checks["action_rate_above_min"] is False
    assert checks["positive_gross_return"] is False


def test_all_gates_pass_yields_edge_candidate():
    # Sanity: the verdict CAN say "edge candidate" when every gate is satisfied.
    verdict, checks = decide_verdict(
        pbo=0.1,
        gross_dsr=0.9,
        mean_gross_return=1e-4,
        action_rate_value=0.25,
        macro_f1=0.40,
        cpcv_median_gross_sharpe=0.3,
        v=_thresholds(),
    )
    assert all(checks.values())
    assert verdict == VERDICT_EDGE


def test_missing_statistic_fails_its_gate():
    # A None DSR/PBO must fail (absence of evidence is not evidence of edge).
    _, checks = decide_verdict(
        pbo=None,
        gross_dsr=None,
        mean_gross_return=1e-4,
        action_rate_value=0.25,
        macro_f1=0.40,
        cpcv_median_gross_sharpe=None,
        v=_thresholds(),
    )
    assert checks["pbo_below_max"] is False
    assert checks["gross_dsr_above_min"] is False
    assert checks["cpcv_median_gross_sharpe_positive"] is False


# --- CLI parsing ------------------------------------------------------------ #
def test_cli_parses_symbols_window_and_flags(monkeypatch):
    captured: dict = {}

    def _fake_run(symbol, mmcfg, vcfg, **kw):  # noqa: ANN001
        captured[symbol] = kw
        return {  # minimal shape for _print_verdict
            "symbol": symbol,
            "verdict": VERDICT_NONE,
            "verdict_checks": {},
            "action_rate": 0.0,
            "pred_balance": {},
            "classification": {"macro_f1": 0.0},
            "signal_return": {"gross_sharpe": 0.0, "gross_dsr": None, "mean_gross_return": 0.0},
            "pbo": None,
            "cpcv": {"gross_sharpe": {"mean": None, "median": None, "min": None, "max": None}},
            "shap": {"available": False},
        }

    monkeypatch.setattr(run_mod, "run_micro_symbol", _fake_run)
    rc = run_mod.main(
        [
            "--symbols",
            "ES",
            "--start",
            "2026-01-26",
            "--end",
            "2026-06-06",
            "--no-mlflow",
            "--no-interpret",
            "--rebuild-cache",
        ]
    )
    assert rc == 0
    kw = captured["ES"]
    assert kw["log_mlflow"] is False and kw["interpret"] is False and kw["rebuild_cache"] is True
    assert kw["start"] == datetime(2026, 1, 26, tzinfo=UTC)
    assert kw["end"] == datetime(2026, 6, 6, tzinfo=UTC)


# --- summary schema (integration through the real harness) ------------------ #
def test_summary_has_required_schema():
    mtm = _synthetic_matrix()
    mmcfg, vcfg = MicroModelConfig.load(), ValidationConfig.load()
    search = run_micro_search(mtm, mmcfg, vcfg)
    summary = evaluate_micro_model(mtm, search, mmcfg, vcfg)
    required = {
        "micro_model_version",
        "microstructure_feature_version",
        "micro_label_version",
        "n_samples",
        "effective_n",
        "n_features",
        "class_balance",
        "pred_balance",
        "action_rate",
        "pbo",
        "cpcv",
        "verdict",
        "verdict_checks",
    }
    assert required.issubset(summary)
    assert summary["verdict"] in {VERDICT_EDGE, VERDICT_NONE}
    assert summary["n_trials"] == 8
    assert summary["n_features"] == len(FEATS)
    # JSON-serializable (matches what save_micro_run writes)
    import json

    json.dumps(summary, default=str)
