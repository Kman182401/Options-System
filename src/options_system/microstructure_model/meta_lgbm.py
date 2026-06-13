"""Regularized BINARY LightGBM for the Phase-20 meta-model + fold-local weighting.

The Phase-20 meta-model is a *binary* gate: given the m1 order-flow block, the s2
sentiment block and the primary's ``|ofi_top|`` magnitude, it predicts
``P(meta_label = 1)`` — the probability that the fixed primary side rule
(``sign(ofi_top)`` at ``t0``) called the correct barrier side. This is a SEPARATE
estimator from the Phase-14 3-class :class:`RegularizedLGBMMulticlass` (which is left
byte-identical): the only changes are ``objective="binary"`` (``num_class`` not
applicable) and a ``predict_proba_pos`` convenience that returns ``P(class == 1)``.

Every other knob — the heavy regularization, the determinism pins, the early-stopping
plumbing — is inherited unchanged from ``config/micro_model.yaml`` (mm1) via
:func:`build_meta_estimator`. Fold-local **balanced** class weighting is reused
verbatim from :mod:`.lgbm` (``fold_local_class_weights`` / ``effective_sample_weight``),
now over the binary classes ``(0, 1)``, so the same audited weighting mechanism handles
the binary imbalance. Purge + embargo for the inner early-stopping block use the single
source of truth (``validation._purge.train_indices``).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from ..microstructure.model_config import ClassWeightingCfg, EarlyStoppingCfg, MicroModelConfig
from .lgbm import _quiet_feature_names, effective_sample_weight, fold_local_class_weights

# The binary meta alphabet: 0 = primary was wrong / timed out, 1 = primary was right.
META_CLASSES: tuple[int, int] = (0, 1)


class RegularizedLGBMBinary(ClassifierMixin, BaseEstimator):
    """Heavily-regularized, deterministic BINARY LightGBM classifier (meta-gate).

    Same constructor surface as :class:`.lgbm.RegularizedLGBMMulticlass` so it drops
    straight into the leak-safe fold machinery; the difference is ``objective="binary"``
    and :meth:`predict_proba_pos`.
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

        # objective="binary" (num_class not applicable); sklearn infers the 2 classes
        # from y so a degenerate single-class fold never raises.
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
    ) -> RegularizedLGBMBinary:
        """Fit the binary booster, honouring ``sample_weight`` (fold-local effective
        weights). Early stopping is used only when both ``early_stopping_rounds`` and an
        ``eval_set`` are supplied."""
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
        """Predicted class in ``{0, 1}`` (act / don't-act before thresholding)."""
        with _quiet_feature_names():
            return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Class probabilities, columns ordered by ``classes_``."""
        with _quiet_feature_names():
            return self._model.predict_proba(X)

    def predict_proba_pos(self, X: np.ndarray) -> np.ndarray:
        """``P(meta_label = 1)`` — the gate probability. 0 if the positive class is
        absent from a degenerate single-class training fold."""
        proba = self.predict_proba(X)
        classes = list(self.classes_)
        if 1 in classes:
            return np.asarray(proba[:, classes.index(1)], dtype=float)
        return np.zeros(X.shape[0], dtype=float)


def build_meta_estimator(
    cfg: MicroModelConfig, overrides: dict[str, Any] | None = None
) -> RegularizedLGBMBinary:
    """Construct the binary meta-estimator from the (inherited) mm1 config + grid overrides.

    Reuses ``cfg.lgbm.as_params`` (the exact mm1 regularization params + the 8-config
    grid overrides) and ``cfg.early_stopping`` / ``cfg.seed`` — nothing new is introduced.
    """
    params = cfg.lgbm.as_params(overrides)
    es_rounds = cfg.early_stopping.rounds if cfg.early_stopping.enabled else None
    return RegularizedLGBMBinary(**params, early_stopping_rounds=es_rounds, random_state=cfg.seed)


def fit_meta_fold(
    est: RegularizedLGBMBinary,
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
    classes: tuple[int, ...] = META_CLASSES,
) -> RegularizedLGBMBinary:
    """Fit ``est`` on ``train_idx`` with FOLD-LOCAL binary class weights + purged early
    stopping — the exact mechanism of :func:`.lgbm.fit_micro_fold`, over the binary
    classes ``(0, 1)``.

    Class weights are computed from ``y[train_idx]`` ONLY, then multiplied into
    ``base_weight``. When early stopping is on, the last ``inner_val_fraction`` of the
    (``t0``-sorted) training fold is the early-stopping block and any training row whose
    ``[t0, t1]`` overlaps it is purged, so the inner-validation set cannot leak into the
    inner fit. Inner-train AND inner-val weights both use the SAME fold-local mapping.
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
