"""Regularized LightGBM **directional** classifier — the Phase-5 signal model.

A thin, sklearn-compatible wrapper around :class:`lightgbm.LGBMClassifier`,
pinned to a deliberately **heavily regularized** configuration. With only ~2.5-2.7k
effective samples against 45 features the design goal is a model that *cannot*
overfit — shallow trees, large leaves, L1+L2, sub-1 bagging/feature fractions —
not one that maximises in-sample accuracy.

The target is **direction** (``y ∈ {-1, +1}`` = down/up), so ``predict`` already
returns a trading sign and ``predict_proba`` a calibrated up/down probability (for
later thresholding — never thresholded here). Sample weights are honoured.
Determinism is pinned (fixed seeds, single-threaded, ``deterministic=True``) so a
run is byte-reproducible.

:func:`fit_fold` is the leak-safe fold fitter shared by the tuner and the
evaluator: when early stopping is enabled it carves a **purged** chronological
tail of the training fold as the early-stopping validation set, so neither the
test fold nor the inner-validation block can leak into the fit.
"""

from __future__ import annotations

import warnings
from contextlib import contextmanager
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from .config import EarlyStoppingCfg, ModelConfig

# Up = +1, down = -1. Fixed so predict() output maps straight to a trading sign.
_UP, _DOWN = 1, -1


@contextmanager
def _quiet_feature_names():
    """Silence sklearn's benign "X has no valid feature names" warning.

    LightGBM records column names ("Column_0"...) when fit on a bare ndarray, so
    sklearn then warns on every ndarray ``predict``. We pass numpy throughout by
    design; the warning is pure noise, so suppress just that one message.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="X does not have valid feature names")
        yield


class RegularizedLGBMDirectional(ClassifierMixin, BaseEstimator):
    """Heavily-regularized, deterministic LightGBM directional classifier.

    All hyperparameters are constructor args (sklearn convention: no logic in
    ``__init__``) so ``clone`` / ``get_params`` / ``set_params`` work and the
    estimator drops straight into the validation splitters.
    """

    def __init__(
        self,
        *,
        n_estimators: int = 600,
        learning_rate: float = 0.02,
        max_depth: int = 3,
        num_leaves: int = 7,
        min_child_samples: int = 200,
        subsample: float = 0.7,
        subsample_freq: int = 1,
        colsample_bytree: float = 0.6,
        reg_alpha: float = 1.0,
        reg_lambda: float = 5.0,
        min_split_gain: float = 0.0,
        max_bin: int = 127,
        early_stopping_rounds: int | None = None,
        random_state: int = 7,
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.num_leaves = num_leaves
        self.min_child_samples = min_child_samples
        self.subsample = subsample
        self.subsample_freq = subsample_freq
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.min_split_gain = min_split_gain
        self.max_bin = max_bin
        self.early_stopping_rounds = early_stopping_rounds
        self.random_state = random_state

    # -- internals ----------------------------------------------------------
    def _make(self) -> Any:
        import lightgbm as lgb

        return lgb.LGBMClassifier(
            objective="binary",
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            num_leaves=self.num_leaves,
            min_child_samples=self.min_child_samples,
            subsample=self.subsample,
            subsample_freq=self.subsample_freq,
            colsample_bytree=self.colsample_bytree,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            min_split_gain=self.min_split_gain,
            max_bin=self.max_bin,
            random_state=self.random_state,
            # Determinism: single-threaded, row-wise, fixed sub-sampling seeds.
            n_jobs=1,
            num_threads=1,
            deterministic=True,
            force_row_wise=True,
            bagging_seed=self.random_state,
            feature_fraction_seed=self.random_state,
            data_random_seed=self.random_state,
            verbose=-1,
        )

    # -- sklearn API --------------------------------------------------------
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
        eval_set: list[tuple[np.ndarray, np.ndarray]] | None = None,
        eval_sample_weight: list[np.ndarray] | None = None,
    ) -> RegularizedLGBMDirectional:
        """Fit the booster, honouring ``sample_weight``.

        If ``early_stopping_rounds`` is set **and** an ``eval_set`` is supplied, the
        number of boosting rounds is chosen by early stopping on that set; otherwise
        the full ``n_estimators`` are used.
        """
        import lightgbm as lgb

        self._model = self._make()
        fit_kw: dict[str, Any] = {"sample_weight": sample_weight}
        if self.early_stopping_rounds and eval_set:
            fit_kw["eval_set"] = eval_set
            if eval_sample_weight is not None:
                fit_kw["eval_sample_weight"] = eval_sample_weight
            fit_kw["callbacks"] = [
                lgb.early_stopping(self.early_stopping_rounds, verbose=False),
                lgb.log_evaluation(0),
            ]
        with _quiet_feature_names():
            self._model.fit(X, y, **fit_kw)
        self.classes_ = self._model.classes_
        self.feature_importances_ = self._model.feature_importances_
        self.best_iteration_ = getattr(self._model, "best_iteration_", None)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicted direction in ``{-1, +1}`` (already a trading sign)."""
        with _quiet_feature_names():
            return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Class probabilities, columns ordered by ``classes_`` (``[-1, +1]``)."""
        with _quiet_feature_names():
            return self._model.predict_proba(X)

    def proba_up(self, X: np.ndarray) -> np.ndarray:
        """P(up) — the probability of the ``+1`` class (for later thresholding)."""
        up_col = int(np.flatnonzero(self.classes_ == _UP)[0])
        return self.predict_proba(X)[:, up_col]


def build_estimator(
    cfg: ModelConfig, overrides: dict[str, Any] | None = None
) -> RegularizedLGBMDirectional:
    """Construct the estimator from a :class:`ModelConfig` (+ optional grid overrides)."""
    params = cfg.lgbm.as_params(overrides)
    es_rounds = cfg.early_stopping.rounds if cfg.early_stopping.enabled else None
    return RegularizedLGBMDirectional(
        **params, early_stopping_rounds=es_rounds, random_state=cfg.seed
    )


def fit_fold(
    est: RegularizedLGBMDirectional,
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    *,
    t0: np.ndarray,
    t1: np.ndarray,
    train_idx: np.ndarray,
    n: int,
    early_stopping: EarlyStoppingCfg | None,
    embargo_bars: int,
) -> RegularizedLGBMDirectional:
    """Fit ``est`` on ``train_idx`` with optional **purged** inner early stopping.

    When early stopping is on, the last ``inner_val_fraction`` of the (``t0``-sorted)
    training fold becomes the early-stopping validation block, and any training row
    whose ``[t0, t1]`` overlaps that block is purged (forward embargo applied) — so
    the inner-validation set cannot leak into the inner-training fit. Falls back to a
    plain full-``n_estimators`` fit when the fold is too small to split safely.
    """
    from ..validation._purge import train_indices

    train_idx = np.asarray(train_idx)
    if early_stopping and early_stopping.enabled and est.early_stopping_rounds:
        k = int(round(train_idx.size * early_stopping.inner_val_fraction))
        if 10 <= k < train_idx.size:
            inner_val = train_idx[-k:]
            inner_train = train_indices(
                t0, t1, inner_val, n, embargo_bars, candidates=train_idx[:-k]
            )
            if inner_train.size >= 50:
                est.fit(
                    X[inner_train],
                    y[inner_train],
                    sample_weight=w[inner_train],
                    eval_set=[(X[inner_val], y[inner_val])],
                    eval_sample_weight=[w[inner_val]],
                )
                return est
    est.fit(X[train_idx], y[train_idx], sample_weight=w[train_idx])
    return est
