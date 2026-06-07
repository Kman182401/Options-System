"""Evaluation harness on a synthetic in-memory matrix (no lake, no real model).

Covers: weighted scoring genuinely uses the weights; the harness runs end-to-end
and reports the distribution + PBO + effective N; effective N = Σ uniqueness; and
the run is deterministic. Uses the lightweight baselines only.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np

from options_system.validation.config import ValidationConfig
from options_system.validation.evaluate import (
    Matrix,
    _effective_n,
    _weighted_accuracy,
    baseline_estimators,
    evaluate,
    read_run,
    save_run,
)


def test_weighted_accuracy_honours_weights():
    y = np.array([1, 1, 0, 0])
    pred = np.array([1, 1, 1, 1])  # correct on the two 1s, wrong on the two 0s
    # Equal weights → 50% accuracy.
    assert abs(_weighted_accuracy(y, pred, np.ones(4)) - 0.5) < 1e-12
    # Heavy weight on the correct rows → accuracy rises well above 0.5 (weights matter).
    w = np.array([10.0, 10.0, 1.0, 1.0])
    assert abs(_weighted_accuracy(y, pred, w) - 20.0 / 22.0) < 1e-12


def test_effective_n_is_sum_of_uniqueness():
    uniq = np.array([0.2, 0.5, 0.3, 1.0])
    assert _effective_n(uniq, np.array([0, 2, 3])) == 0.2 + 0.3 + 1.0


def _synthetic_matrix(n=400, horizon=20, seed=5):
    rng = np.random.default_rng(seed)
    t0_start = datetime(2024, 1, 2, tzinfo=UTC)
    t0 = np.array(
        [
            np.datetime64((t0_start + timedelta(minutes=i)).replace(tzinfo=None), "us")
            for i in range(n)
        ]
    )
    t1 = t0 + np.timedelta64(horizon, "m")
    X = rng.normal(0.0, 1.0, (n, 4))  # pure-noise features ⇒ baselines should be unskilled
    y = rng.integers(-1, 2, n)  # labels in {-1,0,+1}
    ret = rng.normal(0.0, 0.01, n)
    weight = np.full(n, 1.0)
    uniqueness = np.full(n, 0.25)
    return Matrix("SYN", X, y, t0, t1, ret, weight, uniqueness, ["f0", "f1", "f2", "f3"])


def test_evaluate_runs_and_reports_distribution_pbo_and_effective_n():
    cfg = ValidationConfig.load()
    m = _synthetic_matrix()
    out = evaluate(baseline_estimators(seed=cfg.evaluation.seed), m, cfg)

    assert out["symbol"] == "SYN"
    assert out["validation_version"] == cfg.validation_version
    assert out["n_samples"] == m.n
    # effective N = Σ uniqueness = 400 * 0.25 = 100
    assert abs(out["effective_n_total"] - 100.0) < 1e-6
    # PBO present (two estimators) and a valid probability
    assert out["pbo"] is not None
    assert 0.0 <= out["pbo"]["pbo"] <= 1.0
    # CPCV distribution present for each estimator
    for name in ("dummy", "logistic"):
        assert out["cpcv"][name]["n_paths"] >= 1
        assert "sharpe_mean" in out["cpcv"][name]
        assert out["kfold"][name]["effective_n"] > 0


def test_evaluate_is_deterministic():
    cfg = ValidationConfig.load()
    m = _synthetic_matrix()
    a = evaluate(baseline_estimators(seed=cfg.evaluation.seed), m, cfg)
    b = evaluate(baseline_estimators(seed=cfg.evaluation.seed), m, cfg)
    assert a["kfold"] == b["kfold"]
    assert a["cpcv"] == b["cpcv"]
    assert a["pbo"] == b["pbo"]


def test_save_and_read_run_round_trip(tmp_path):
    cfg = ValidationConfig.load()
    out = evaluate(baseline_estimators(seed=cfg.evaluation.seed), _synthetic_matrix(), cfg)
    path = save_run(out, runs_dir=tmp_path)
    assert path.exists()
    again = read_run("SYN", runs_dir=tmp_path)
    assert again is not None
    assert again["symbol"] == "SYN"
    assert read_run("NOPE", runs_dir=tmp_path) is None
