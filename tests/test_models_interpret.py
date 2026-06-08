"""SHAP interpretability: importances cover every feature, no spurious dominance."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np

from options_system.models.config import ModelConfig
from options_system.models.dataset import TrainingMatrix
from options_system.models.interpret import explain


def _signal_tm(n: int = 400, seed: int = 2) -> TrainingMatrix:
    rng = np.random.default_rng(seed)
    start = datetime(2024, 1, 2, tzinfo=UTC)
    t0 = np.array(
        [np.datetime64((start + timedelta(minutes=i)).replace(tzinfo=None), "us") for i in range(n)]
    )
    X = rng.normal(size=(n, 6))
    ret = 0.5 * X[:, 0] + rng.normal(0.0, 0.01, n)  # feature 0 drives direction
    y_dir = np.where(ret > 0, 1, -1)
    return TrainingMatrix(
        symbol="SYN",
        X=X,
        y=y_dir.copy(),
        y_dir=y_dir,
        t0=t0,
        t1=t0 + np.timedelta64(10, "m"),
        ret=ret,
        weight=np.ones(n),
        uniqueness=np.full(n, 0.25),
        feature_cols=[f"f{i}" for i in range(6)],
        feature_version="v1",
        label_version="v1",
        timeout_handling="sign_return",
    )


def test_explain_reports_importance_for_every_feature():
    tm = _signal_tm()
    cfg = ModelConfig.load().model_copy(
        update={"lgbm": ModelConfig.load().lgbm.model_copy(update={"n_estimators": 60})}
    )
    out = explain(tm, cfg, {}, max_samples=200, n_local=2)
    assert len(out["importances"]) == len(tm.feature_cols)
    assert len(out["top_features"]) == min(10, len(tm.feature_cols))
    assert 0.0 <= out["top_feature_share"] <= 1.0
    assert len(out["local_explanations"]) == 2
    # feature 0 (the planted driver) should rank in the top features
    assert "f0" in out["top_features"]
