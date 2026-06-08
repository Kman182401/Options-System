"""In-CV search: every config is a counted trial, scored OOS without test-fold leakage."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np

from options_system.models.config import ModelConfig
from options_system.models.dataset import TrainingMatrix
from options_system.models.tune import run_search
from options_system.validation.config import ValidationConfig


def fast_cfg(n_configs: int = 2) -> ModelConfig:
    c = ModelConfig.load()
    grid = {"reg_lambda": [5.0, 20.0][:n_configs]}
    return c.model_copy(
        update={
            "lgbm": c.lgbm.model_copy(update={"n_estimators": 40}),
            "search": c.search.model_copy(update={"grid": grid}),
            "early_stopping": c.early_stopping.model_copy(update={"enabled": False}),
        }
    )


def synth_tm(n: int = 480, horizon: int = 12, seed: int = 4, signal: float = 0.0) -> TrainingMatrix:
    rng = np.random.default_rng(seed)
    start = datetime(2024, 1, 2, tzinfo=UTC)
    t0 = np.array(
        [np.datetime64((start + timedelta(minutes=i)).replace(tzinfo=None), "us") for i in range(n)]
    )
    t1 = t0 + np.timedelta64(horizon, "m")
    X = rng.normal(size=(n, 6))
    ret = rng.normal(0.0, 0.01, n) + signal * X[:, 0]
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


def test_search_counts_trials_and_scores_every_sample_once():
    tm = synth_tm()
    cfg = fast_cfg(2)
    res = run_search(tm, cfg, ValidationConfig.load())
    assert res.n_trials == 2 == len(res.configs)
    assert res.pbo_matrix.shape == (tm.n, 2)
    assert res.strategy_sr.shape == (2,) and res.excess_sr.shape == (2,)
    assert res.scored.all()  # purged K-fold scores every sample OOS exactly once
    assert 0 <= res.selected_index < 2
    assert set(np.unique(res.selected_pred)).issubset({-1, 1})


def test_search_selects_the_best_by_selection_metric():
    tm = synth_tm()
    cfg = fast_cfg(2)
    res = run_search(tm, cfg, ValidationConfig.load())
    accs = [c["directional_accuracy"] for c in res.per_config]
    # selection_metric defaults to directional_accuracy → winner is the argmax
    assert res.selected_index == int(np.argmax(accs))
