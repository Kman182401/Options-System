"""The single fixed, regularized LightGBM regressor (arm T) + the QLIKE custom objective.

The contract fixes ONE heavily-regularized config a priori (no hyperparameter search — that removes
the selection-overfit failure mode that killed the Phase-19/20 near-misses). The model predicts the
forward **log-RV** target; its variance forecast is ``exp(prediction)``.

Training objective:

* **qlike** — a QLIKE-consistent custom objective on the variance scale. With ``d = y_true − pred``
  (both log-RV), ``QLIKE = exp(d) − d − 1`` ⇒ ``grad = ∂L/∂pred = 1 − exp(d)`` and
  ``hess = ∂²L/∂pred² = exp(d) > 0`` (convex). ``d`` is clipped to ``[−30, 30]`` so an early wild
  forecast cannot overflow ``exp``.
* **l2_log_rv** — the DISCLOSED fallback (squared error on log-RV) used only if the custom objective
  is infeasible; the choice is surfaced in the run output, never silent.

Early stopping uses a **purged** chronological tail of the training fold (overlap with the inner-val
block removed via the shared purge primitive), mirroring the micro model's leak-safe inner split.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .config import VolatilityConfig

_CLIP = 30.0


def qlike_objective(y_true: np.ndarray, raw_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """LightGBM custom objective for QLIKE on the variance scale (predicting log-RV).

    Returns ``(grad, hess)`` for ``L = exp(d) − d − 1`` with ``d = y_true − pred`` clipped to a
    safe range. ``grad = 1 − exp(d)``, ``hess = exp(d)``.
    """
    d = np.clip(np.asarray(y_true, dtype=float) - np.asarray(raw_pred, dtype=float), -_CLIP, _CLIP)
    e = np.exp(d)
    return 1.0 - e, e


def qlike_eval(y_true: np.ndarray, raw_pred: np.ndarray) -> tuple[str, float, bool]:
    """LightGBM custom eval metric: mean QLIKE (lower is better) for early stopping."""
    d = np.clip(np.asarray(y_true, dtype=float) - np.asarray(raw_pred, dtype=float), -_CLIP, _CLIP)
    loss = float(np.mean(np.exp(d) - d - 1.0))
    return "qlike", loss, False


class VolatilityLGBM:
    """Fixed regularized LightGBM regressor for forward log-RV (QLIKE or L2-log-RV objective)."""

    def __init__(self, *, params: dict[str, Any], objective_mode: str, seed: int) -> None:
        self.params = dict(params)
        self.objective_mode = objective_mode
        self.seed = seed
        self.early_stopping_rounds: int | None = None

    def _make(self) -> Any:
        import lightgbm as lgb

        objective: Any = qlike_objective if self.objective_mode == "qlike" else "regression"
        return lgb.LGBMRegressor(
            objective=objective,
            random_state=self.seed,
            n_jobs=1,
            num_threads=1,
            deterministic=True,
            force_row_wise=True,
            bagging_seed=self.seed,
            feature_fraction_seed=self.seed,
            data_random_seed=self.seed,
            verbose=-1,
            **self.params,
        )

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        eval_set: list[tuple[np.ndarray, np.ndarray]] | None = None,
    ) -> VolatilityLGBM:
        """Fit; use early stopping (QLIKE eval) only when an ``eval_set`` is supplied.

        The target is **centered on the train mean** before fitting and the offset added back at
        predict time. LightGBM does not boost-from-average for a custom objective, so it starts at
        raw 0 — far below true log-RV (≈ −9) where the QLIKE hessian vanishes, which would strand
        the model at a constant. Centering puts the optimizer's origin at the mean (QLIKE depends
        only on ``y − pred``, so it is invariant to this shift) and the gradients become
        well-scaled.
        """
        import warnings

        import lightgbm as lgb

        y = np.asarray(y, dtype=float)
        self._offset = float(np.mean(y))
        self._model = self._make()
        fit_kw: dict[str, Any] = {}
        if self.early_stopping_rounds and eval_set:
            fit_kw["eval_set"] = [
                (xv, np.asarray(yv, dtype=float) - self._offset) for xv, yv in eval_set
            ]
            fit_kw["eval_metric"] = qlike_eval
            fit_kw["callbacks"] = [
                lgb.early_stopping(self.early_stopping_rounds, verbose=False),
                lgb.log_evaluation(0),
            ]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="X does not have valid feature names")
            self._model.fit(x, y - self._offset, **fit_kw)
        self.feature_importances_ = self._model.feature_importances_
        self.best_iteration_ = getattr(self._model, "best_iteration_", None)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predicted forward log-RV (variance forecast = ``exp`` of this); offset added back."""
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="X does not have valid feature names")
            return np.asarray(self._model.predict(x), dtype=float) + self._offset


def build_vol_estimator(vcfg: VolatilityConfig) -> VolatilityLGBM:
    """Construct the fixed regressor from config (regularization params + objective + seed)."""
    est = VolatilityLGBM(
        params={**vcfg.lgbm.estimator_params(), "n_estimators": vcfg.lgbm.n_estimators},
        objective_mode=vcfg.lgbm.objective,
        seed=vcfg.seed,
    )
    est.early_stopping_rounds = vcfg.lgbm.early_stopping_rounds
    return est


def fit_vol_fold(
    est: VolatilityLGBM,
    x: np.ndarray,
    y: np.ndarray,
    *,
    t0: np.ndarray,
    t1: np.ndarray,
    train_idx: np.ndarray,
    n: int,
    inner_val_fraction: float,
    embargo_bars: int,
) -> VolatilityLGBM:
    """Fit on ``train_idx`` with a PURGED chronological-tail early-stopping block.

    The last ``inner_val_fraction`` of the (t0-sorted) training fold is the inner-validation block;
    any training row whose ``[t0, t1]`` overlaps it is purged (shared ``train_indices`` primitive),
    so the inner-val cannot leak into the inner fit. Falls back to a plain fit when the fold is too
    small to split safely.
    """
    from ..validation._purge import train_indices

    train_idx = np.asarray(train_idx)
    if est.early_stopping_rounds:
        k = int(round(train_idx.size * inner_val_fraction))
        if 10 <= k < train_idx.size:
            inner_val = train_idx[-k:]
            inner_train = train_indices(
                t0, t1, inner_val, n, embargo_bars, candidates=train_idx[:-k]
            )
            if inner_train.size >= 50:
                est.fit(x[inner_train], y[inner_train], eval_set=[(x[inner_val], y[inner_val])])
                return est
    est.fit(x[train_idx], y[train_idx])
    return est
