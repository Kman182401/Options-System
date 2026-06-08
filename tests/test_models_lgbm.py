"""RegularizedLGBMDirectional: predicts a sign, honours weights, is deterministic."""

from __future__ import annotations

import numpy as np

from options_system.models.config import ModelConfig
from options_system.models.lgbm import build_estimator, fit_fold


def _planted(n=600, p=6, seed=1):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    y = np.where(X[:, 0] + 0.3 * rng.normal(size=n) > 0, 1, -1)
    return X, y


def test_predicts_sign_and_proba_shape():
    X, y = _planted()
    est = build_estimator(ModelConfig.load())
    est.fit(X, y, sample_weight=np.ones(len(y)))
    assert set(np.unique(est.predict(X))).issubset({-1, 1})
    assert list(est.classes_) == [-1, 1]
    proba = est.predict_proba(X)
    assert proba.shape == (len(y), 2)
    up = est.proba_up(X)
    assert up.min() >= 0.0 and up.max() <= 1.0


def test_deterministic_under_seed():
    X, y = _planted()
    a = build_estimator(ModelConfig.load())
    a.fit(X, y, sample_weight=np.ones(len(y)))
    b = build_estimator(ModelConfig.load())
    b.fit(X, y, sample_weight=np.ones(len(y)))
    assert np.array_equal(a.predict(X), b.predict(X))


def test_sample_weight_changes_fit():
    X, y = _planted()
    full = build_estimator(ModelConfig.load())
    full.fit(X, y, sample_weight=np.ones(len(y)))
    w = np.ones(len(y))
    w[y == 1] = 0.0  # zero-weight one class → different fit
    zeroed = build_estimator(ModelConfig.load())
    zeroed.fit(X, y, sample_weight=w)
    assert not np.array_equal(full.predict(X), zeroed.predict(X))


def test_fit_fold_early_stopping_runs_and_trims_rounds():
    X, y = _planted(n=800)
    n = len(y)
    t0 = np.arange(n).astype("datetime64[m]").astype("datetime64[us]")
    t1 = t0 + np.timedelta64(10, "m")
    train_idx = np.arange(0, 700)
    cfg = ModelConfig.load()
    est = build_estimator(cfg)  # early_stopping_rounds set from config
    fit_fold(
        est,
        X,
        y,
        np.ones(n),
        t0=t0,
        t1=t1,
        train_idx=train_idx,
        n=n,
        early_stopping=cfg.early_stopping,
        embargo_bars=0,
    )
    # early stopping engaged → best_iteration_ recorded and <= the configured cap
    assert est.best_iteration_ is not None
    assert est.best_iteration_ <= cfg.lgbm.n_estimators
