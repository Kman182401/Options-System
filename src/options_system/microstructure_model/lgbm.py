"""Regularized 3-class LightGBM micro signal model + fold-local class weighting.

A thin, sklearn-compatible wrapper around :class:`lightgbm.LGBMClassifier` pinned to
a heavily-regularized **multiclass** config (objective ``multiclass``). The target is
the micro triple-barrier class ``y ∈ {-1, 0, +1}`` (lower / timeout / upper), so
``predict`` returns the class label directly — which IS the trading signal
(-1 short, 0 flat, +1 long). Determinism is pinned (fixed seeds, single-threaded,
``deterministic=True``).

**Fold-local class weighting.** The timeout class is ~78-80% of labels, so a model
left to its own devices predicts "0" everywhere. Class weights are therefore
computed INSIDE each training fold from ``y_train`` ONLY (never globally), then
MULTIPLIED into the persisted uniqueness/sample weights:

    effective_sample_weight_i = sample_weight_i * class_weight[y_i]

``num_class`` is deliberately NOT passed to LightGBM — sklearn's ``LGBMClassifier``
infers the class count from each fold's ``y``, so a CV fold that happens to be
missing one class never errors (the wrapper just won't predict the absent class for
that fold). :func:`fold_local_class_weights` mirrors the same robustness.
"""

from __future__ import annotations

import warnings
from contextlib import contextmanager
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from ..microstructure.model_config import ClassWeightingCfg, EarlyStoppingCfg, MicroModelConfig

# The 3-class signal alphabet, fixed so predict() output maps straight to a position.
CLASSES: tuple[int, int, int] = (-1, 0, 1)


@contextmanager
def _quiet_feature_names():
    """Silence sklearn's benign "X has no valid feature names" warning (we pass numpy)."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="X does not have valid feature names")
        yield


# --------------------------------------------------------------------------- #
# Fold-local class weighting (computed from y_train only, multiplied with weights)
# --------------------------------------------------------------------------- #
def fold_local_class_weights(
    y_train: np.ndarray,
    sample_weight_train: np.ndarray | None,
    classes: tuple[int, ...] = CLASSES,
    *,
    use_sample_weight: bool = True,
) -> dict[int, float]:
    """Balanced class weights from ``y_train`` ONLY (never the full dataset).

    Uses sklearn's ``compute_class_weight(class_weight="balanced", ...)`` over the
    classes actually PRESENT in ``y_train`` (sklearn raises if asked for an absent
    class), with the per-sample ``sample_weight`` folded into the balance when
    ``use_sample_weight`` is set — so a class's effective *mass*, not its raw count,
    is equalised: ``cw[c] · Σ sample_weight[y==c]`` is equal across present classes.
    Any class absent from this fold gets weight ``1.0`` (it has no rows here, so the
    value is irrelevant to the fit).
    """
    from sklearn.utils.class_weight import compute_class_weight

    y_train = np.asarray(y_train)
    present = np.array([c for c in classes if np.any(y_train == c)], dtype=int)
    cw: dict[int, float] = dict.fromkeys(classes, 1.0)
    if present.size < 2:
        return cw  # 0 or 1 class present → no balancing to do (degenerate fold)
    sw = (
        np.asarray(sample_weight_train, dtype=float)
        if (use_sample_weight and sample_weight_train is not None)
        else None
    )
    weights = compute_class_weight("balanced", classes=present, y=y_train, sample_weight=sw)
    for c, w in zip(present.tolist(), weights.tolist(), strict=True):
        cw[int(c)] = float(w)
    return cw


def effective_sample_weight(
    y: np.ndarray, base_weight: np.ndarray, class_weight: dict[int, float]
) -> np.ndarray:
    """Per-sample effective weight = ``base_weight * class_weight[y]`` (elementwise)."""
    y = np.asarray(y)
    base_weight = np.asarray(base_weight, dtype=float)
    mult = np.array([class_weight.get(int(c), 1.0) for c in y], dtype=float)
    return base_weight * mult


# --------------------------------------------------------------------------- #
# The estimator
# --------------------------------------------------------------------------- #
class RegularizedLGBMMulticlass(ClassifierMixin, BaseEstimator):
    """Heavily-regularized, deterministic 3-class LightGBM classifier.

    All hyperparameters are constructor args (sklearn convention) so ``clone`` /
    ``get_params`` / ``set_params`` work and the estimator drops straight into the
    validation splitters.
    """

    def __init__(
        self,
        *,
        n_estimators: int = 600,
        learning_rate: float = 0.02,
        max_depth: int = 3,
        num_leaves: int = 7,
        min_child_samples: int = 100,
        subsample: float = 0.7,
        subsample_freq: int = 1,
        colsample_bytree: float = 0.7,
        reg_alpha: float = 1.0,
        reg_lambda: float = 10.0,
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

    def _make(self) -> Any:
        import lightgbm as lgb

        # objective="multiclass" WITHOUT num_class: sklearn infers it from y, so a
        # fold missing one of {-1,0,+1} never raises a class-count mismatch.
        return lgb.LGBMClassifier(
            objective="multiclass",
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
            n_jobs=1,
            num_threads=1,
            deterministic=True,
            force_row_wise=True,
            bagging_seed=self.random_state,
            feature_fraction_seed=self.random_state,
            data_random_seed=self.random_state,
            verbose=-1,
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
        eval_set: list[tuple[np.ndarray, np.ndarray]] | None = None,
        eval_sample_weight: list[np.ndarray] | None = None,
    ) -> RegularizedLGBMMulticlass:
        """Fit the multiclass booster, honouring ``sample_weight`` (the fold-local
        effective weights). Early stopping is used only when both an
        ``early_stopping_rounds`` and an ``eval_set`` are supplied."""
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
        """Predicted class in ``{-1, 0, +1}`` — the trading signal (short/flat/long)."""
        with _quiet_feature_names():
            return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Class probabilities, columns ordered by ``classes_``."""
        with _quiet_feature_names():
            return self._model.predict_proba(X)


def build_micro_estimator(
    cfg: MicroModelConfig, overrides: dict[str, Any] | None = None
) -> RegularizedLGBMMulticlass:
    """Construct the estimator from a :class:`MicroModelConfig` (+ optional grid overrides)."""
    params = cfg.lgbm.as_params(overrides)
    es_rounds = cfg.early_stopping.rounds if cfg.early_stopping.enabled else None
    return RegularizedLGBMMulticlass(
        **params, early_stopping_rounds=es_rounds, random_state=cfg.seed
    )


def fit_micro_fold(
    est: RegularizedLGBMMulticlass,
    X: np.ndarray,
    y: np.ndarray,
    base_weight: np.ndarray,
    *,
    t0: np.ndarray,
    t1: np.ndarray,
    train_idx: np.ndarray,
    n: int,
    early_stopping: EarlyStoppingCfg | None,
    embargo_bars: int,
    class_weighting: ClassWeightingCfg,
    classes: tuple[int, ...] = CLASSES,
) -> RegularizedLGBMMulticlass:
    """Fit ``est`` on ``train_idx`` with FOLD-LOCAL class weights + purged early stopping.

    Class weights are computed from ``y[train_idx]`` ONLY (the outer training fold),
    then multiplied into ``base_weight`` to form the effective sample weights. When
    early stopping is on, the last ``inner_val_fraction`` of the (``t0``-sorted)
    training fold is the early-stopping block and any training row whose ``[t0, t1]``
    overlaps it is purged — so the inner-validation set cannot leak into the inner
    fit. The inner-train AND inner-val (eval) weights both use the SAME fold-local
    class mapping (never the global full-sample weights). Falls back to a plain fit
    when the fold is too small to split safely.
    """
    from ..validation._purge import train_indices

    train_idx = np.asarray(train_idx)
    if class_weighting.enabled:
        cw = fold_local_class_weights(
            y[train_idx],
            base_weight[train_idx],
            classes,
            use_sample_weight=class_weighting.use_sample_weight_in_balance,
        )
        eff_w = effective_sample_weight(y, base_weight, cw)
    else:
        eff_w = np.asarray(base_weight, dtype=float)

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
                    sample_weight=eff_w[inner_train],
                    eval_set=[(X[inner_val], y[inner_val])],
                    eval_sample_weight=[eff_w[inner_val]],
                )
                return est
    est.fit(X[train_idx], y[train_idx], sample_weight=eff_w[train_idx])
    return est
