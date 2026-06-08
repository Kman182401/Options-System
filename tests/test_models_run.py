"""Runner wiring: the configured target policy actually reaches the matrix.

Regression for the config-authority gap where ``run_symbol`` built the matrix with
the hard-coded default instead of ``mcfg.target.timeout_handling`` — so a
``drop`` config silently evaluated a ``sign_return`` matrix.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np

import options_system.models.run as run_mod
from options_system.models.config import ModelConfig
from options_system.models.dataset import TrainingMatrix
from options_system.validation.config import ValidationConfig


def _fast_cfg(handling: str) -> ModelConfig:
    c = ModelConfig.load()
    return c.model_copy(
        update={
            "target": c.target.model_copy(update={"timeout_handling": handling}),
            "lgbm": c.lgbm.model_copy(update={"n_estimators": 40}),
            "search": c.search.model_copy(update={"grid": {"reg_lambda": [5.0, 20.0]}}),
            "early_stopping": c.early_stopping.model_copy(update={"enabled": False}),
        }
    )


def _tm(handling: str, n: int = 420, seed: int = 7) -> TrainingMatrix:
    rng = np.random.default_rng(seed)
    start = datetime(2024, 1, 2, tzinfo=UTC)
    t0 = np.array(
        [np.datetime64((start + timedelta(minutes=i)).replace(tzinfo=None), "us") for i in range(n)]
    )
    X = rng.normal(size=(n, 6))
    ret = rng.normal(0.0, 0.01, n)
    y_dir = np.where(ret > 0, 1, -1)
    return TrainingMatrix(
        symbol="SYN",
        X=X,
        y=y_dir.copy(),
        y_dir=y_dir,
        t0=t0,
        t1=t0 + np.timedelta64(12, "m"),
        ret=ret,
        weight=np.ones(n),
        uniqueness=np.full(n, 0.25),
        feature_cols=[f"f{i}" for i in range(6)],
        feature_version="v1",
        label_version="v1",
        timeout_handling=handling,
    )


def test_run_symbol_forwards_configured_timeout_handling(monkeypatch):
    captured: dict = {}

    def fake_load(symbol, **kwargs):
        captured["timeout_handling"] = kwargs.get("timeout_handling")
        return _tm(kwargs.get("timeout_handling", "sign_return"))

    monkeypatch.setattr(run_mod, "load_training_matrix", fake_load)
    mcfg = _fast_cfg("drop")
    summary = run_mod.run_symbol(
        "SYN", mcfg, ValidationConfig.load(), log_mlflow=False, interpret=False, save=False
    )

    # the config's target policy must reach the matrix builder...
    assert captured["timeout_handling"] == "drop"
    # ...and be reflected in the published summary (not silently the default)
    assert summary["timeout_handling"] == "drop"
