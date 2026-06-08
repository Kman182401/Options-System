"""In-CV, regularization-focused hyperparameter search — trials counted for the DSR.

The search runs **inside** the purged K-fold so no configuration ever sees its own
test fold during selection. Each grid configuration is scored by its pooled
out-of-sample (purged-KFold) **weighted directional accuracy**; the best is
selected. Crucially, the *number of configurations tried* (``n_trials``) is
recorded and fed to the **Deflated Sharpe Ratio** downstream — every extra config
correctly raises the bar a result must clear, which is why the grid is kept tiny
(``config/models.yaml``).

This module produces everything the honest evaluation (``evaluate_model.py``) needs
to deflate properly: the per-config strategy/excess Sharpe estimates (the trial
distribution for the DSR) and the ``(samples × configs)`` return matrix (for PBO).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..common.logging import get_logger
from ..validation._purge import embargo_bars_from_pct
from ..validation.purged_kfold import PurgedKFold
from ..validation.stats import sharpe_ratio
from .config import ModelConfig
from .dataset import TrainingMatrix
from .lgbm import build_estimator, fit_fold

logger = get_logger(__name__)


@dataclass(frozen=True)
class SearchResult:
    """Outcome of the in-CV grid search (the inputs to honest, deflated evaluation)."""

    selected_overrides: dict
    selected_index: int
    n_trials: int
    selection_metric: str
    configs: list[dict]  # the override dict per trial, in deterministic order
    per_config: list[dict]  # pooled-OOS metrics per trial
    strategy_sr: np.ndarray  # (n_trials,) per-config pooled strategy Sharpe (DSR trials)
    excess_sr: np.ndarray  # (n_trials,) per-config pooled excess-over-long Sharpe (DSR trials)
    pbo_matrix: np.ndarray  # (n_samples, n_trials) per-sample strategy return (for PBO)
    selected_pred: np.ndarray  # (n_samples,) pooled OOS predicted sign for the winner
    scored: np.ndarray  # (n_samples,) bool — sample scored OOS (every sample once)


def _weighted_accuracy(y_true: np.ndarray, y_pred: np.ndarray, w: np.ndarray) -> float:
    sw = float(w.sum())
    if sw <= 0.0:
        return float("nan")
    return float((((y_true == y_pred).astype(float)) * w).sum() / sw)


def kfold_oos(
    tm: TrainingMatrix, overrides: dict, mcfg: ModelConfig, n_splits: int, embargo_pct: float
) -> tuple[np.ndarray, np.ndarray]:
    """Purged-KFold pooled OOS predictions for one config (every sample scored once).

    Returns ``(pred, scored)`` where ``pred`` is the predicted direction in
    ``{-1, +1}`` at each sample's out-of-sample fold (0 where unscored) and
    ``scored`` is the boolean mask of samples that received an OOS prediction.
    """
    n = tm.n
    embargo = embargo_bars_from_pct(embargo_pct, n)
    cv = PurgedKFold(n_splits, tm.t0, tm.t1, embargo_bars=embargo)
    pred = np.zeros(n, dtype=float)
    scored = np.zeros(n, dtype=bool)
    for train_idx, test_idx in cv.split():
        if train_idx.size == 0 or test_idx.size == 0:
            continue
        est = build_estimator(mcfg, overrides)
        fit_fold(
            est,
            tm.X,
            tm.y_dir,
            tm.weight,
            t0=tm.t0,
            t1=tm.t1,
            train_idx=train_idx,
            n=n,
            early_stopping=mcfg.early_stopping,
            embargo_bars=embargo,
        )
        pred[test_idx] = est.predict(tm.X[test_idx])
        scored[test_idx] = True
    return pred, scored


def run_search(tm: TrainingMatrix, mcfg: ModelConfig, vcfg) -> SearchResult:
    """Run the small grid through purged K-fold and select the best config (no leakage).

    ``vcfg`` is a :class:`options_system.validation.config.ValidationConfig` (the
    K-fold split/embargo settings). Selection uses pooled OOS weighted directional
    accuracy (or excess Sharpe, per ``search.selection_metric``).
    """
    configs = mcfg.search.configs()
    n_trials = len(configs)
    n = tm.n
    pbo_matrix = np.zeros((n, n_trials), dtype=float)
    strategy_sr = np.zeros(n_trials, dtype=float)
    excess_sr = np.zeros(n_trials, dtype=float)
    per_config: list[dict] = []
    preds: list[np.ndarray] = []
    scored_final = np.zeros(n, dtype=bool)

    for j, overrides in enumerate(configs):
        pred, scored = kfold_oos(tm, overrides, mcfg, vcfg.kfold.n_splits, vcfg.kfold.embargo_pct)
        scored_final = scored
        pos = pred.astype(float)  # already ±1 (or 0 where unscored)
        strat = pos * tm.ret
        pbo_matrix[:, j] = strat
        s = scored
        acc = _weighted_accuracy(tm.y_dir[s], pred[s].astype(int), tm.weight[s])
        sr = sharpe_ratio(strat[s])
        excess = (pos[s] - 1.0) * tm.ret[s]  # strategy minus perma-long benchmark
        ex_sr = sharpe_ratio(excess)
        strategy_sr[j] = sr
        excess_sr[j] = ex_sr
        preds.append(pred)
        per_config.append(
            {
                "overrides": overrides,
                "directional_accuracy": round(acc, 4),
                "strategy_sharpe": round(sr, 5),
                "excess_sharpe": round(ex_sr, 5),
                "mean_excess_return": round(
                    float((excess * tm.weight[s]).sum() / tm.weight[s].sum()), 8
                ),
            }
        )
        logger.info(
            f"  trial {j + 1}/{n_trials} {overrides} -> acc={acc:.4f} "
            f"strat_sr={sr:.4f} excess_sr={ex_sr:.4f}"
        )

    metric_key = (
        "directional_accuracy"
        if mcfg.search.selection_metric == "directional_accuracy"
        else "excess_sharpe"
    )
    scores = np.array([c[metric_key] for c in per_config], dtype=float)
    selected_index = int(np.argmax(scores))  # deterministic: first max wins

    return SearchResult(
        selected_overrides=configs[selected_index],
        selected_index=selected_index,
        n_trials=n_trials,
        selection_metric=mcfg.search.selection_metric,
        configs=configs,
        per_config=per_config,
        strategy_sr=strategy_sr,
        excess_sr=excess_sr,
        pbo_matrix=pbo_matrix,
        selected_pred=preds[selected_index],
        scored=scored_final,
    )
