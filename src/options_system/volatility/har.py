"""HAR-RV benchmark (Corsi 2009) — the bar Phase 21 must beat.

A plain OLS regression of the forward log-RV target on the three causal HAR predictors (daily /
weekly / monthly trailing log-RV) with an intercept. This is *the* standard realized-volatility
benchmark; the whole verdict is whether the LightGBM treatment forecasts more accurately than this
(skill is measured **over and above HAR-RV**, the analog of "beat buy-and-hold" in the directional
phases). Pure numpy, fit per walk-forward training fold, deterministic.
"""

from __future__ import annotations

import numpy as np


def _design(x: np.ndarray) -> np.ndarray:
    """Prepend an intercept column to the (n, 3) HAR predictor matrix."""
    x = np.asarray(x, dtype=float)
    return np.column_stack([np.ones(x.shape[0]), x])


def fit_har(x_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    """OLS fit of ``y = b0 + b1·daily + b2·weekly + b3·monthly`` → coefficient vector (length 4)."""
    a = _design(x_train)
    coef, _resid, _rank, _sv = np.linalg.lstsq(a, np.asarray(y_train, dtype=float), rcond=None)
    return coef


def predict_har(coef: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Predict forward log-RV from HAR predictors and a fitted coefficient vector."""
    return _design(x) @ np.asarray(coef, dtype=float)
