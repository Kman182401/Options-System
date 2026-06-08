"""Honest evaluation: a noise model must yield 'no significant edge' (the teeth check).

If a directional model trained on features with no relationship to the target ever
showed an edge through this evaluator, the gate would be broken — so the core test
is that pure noise lands on the honest null, with every number finite and sourced
from the framework.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np

from options_system.models.config import ModelConfig
from options_system.models.dataset import TrainingMatrix
from options_system.models.evaluate_model import (
    VERDICT_NONE,
    evaluate_model,
    read_model_run,
    save_model_run,
)
from options_system.models.tune import run_search
from options_system.validation.config import ValidationConfig


def _fast_cfg() -> ModelConfig:
    c = ModelConfig.load()
    return c.model_copy(
        update={
            "lgbm": c.lgbm.model_copy(update={"n_estimators": 40}),
            "search": c.search.model_copy(update={"grid": {"reg_lambda": [5.0, 20.0]}}),
            "early_stopping": c.early_stopping.model_copy(update={"enabled": False}),
        }
    )


def _noise_tm(n: int = 540, horizon: int = 12, seed: int = 11) -> TrainingMatrix:
    rng = np.random.default_rng(seed)
    start = datetime(2024, 1, 2, tzinfo=UTC)
    t0 = np.array(
        [np.datetime64((start + timedelta(minutes=i)).replace(tzinfo=None), "us") for i in range(n)]
    )
    t1 = t0 + np.timedelta64(horizon, "m")
    X = rng.normal(size=(n, 6))  # features unrelated to the target
    ret = rng.normal(0.0, 0.01, n)
    y_dir = np.where(ret > 0, 1, -1)
    return TrainingMatrix(
        symbol="SYN",
        X=X,
        y=y_dir.copy(),
        y_dir=y_dir,
        t0=t0,
        t1=t1,
        ret=ret,
        weight=np.ones(n),
        uniqueness=np.full(n, 0.25),
        feature_cols=[f"f{i}" for i in range(6)],
        feature_version="v1",
        label_version="v1",
        timeout_handling="sign_return",
    )


def test_noise_model_has_no_significant_edge():
    tm = _noise_tm()
    cfg = _fast_cfg()
    vcfg = ValidationConfig.load()
    search = run_search(tm, cfg, vcfg)
    summary = evaluate_model(tm, search, cfg, vcfg)

    assert summary["verdict"] == VERDICT_NONE
    assert not all(summary["verdict_checks"].values())
    # every reported number is finite and framework-sourced
    p = summary["pooled_kfold"]
    for k in ("directional_accuracy", "excess_sharpe", "strategy_sharpe", "long_benchmark_sharpe"):
        assert np.isfinite(p[k])
    assert summary["pbo"] is not None
    assert summary["cpcv"]["n_paths"] >= 1
    assert summary["n_trials"] == cfg.search.n_trials


def test_save_and_read_model_run_round_trip(tmp_path):
    tm = _noise_tm()
    cfg = _fast_cfg()
    vcfg = ValidationConfig.load()
    summary = evaluate_model(tm, run_search(tm, cfg, vcfg), cfg, vcfg)
    path = save_model_run(summary, runs_dir=tmp_path)
    assert path.exists()
    again = read_model_run("SYN", runs_dir=tmp_path)
    assert again is not None and again["symbol"] == "SYN"
    assert read_model_run("NOPE", runs_dir=tmp_path) is None
