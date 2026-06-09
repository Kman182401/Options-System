"""In-CV hyperparameter search for the micro model — trials counted for the DSR.

The search runs INSIDE the purged K-fold so no configuration ever sees its own test
fold during selection. Each grid configuration is scored by its pooled
out-of-sample **gross signal Sharpe** (``selection_metric``); the best is selected.
The number of configurations tried (``n_trials`` = 8) is recorded and fed to the
Deflated Sharpe Ratio downstream — every extra config raises the bar a result must
clear — and the ``(samples × configs)`` gross-return matrix feeds the PBO.

Fold-local class weighting is applied in every fit via :func:`fit_micro_fold`; this
module never touches the global class balance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..common.logging import get_logger
from ..microstructure.model_config import MicroModelConfig
from ..validation._purge import embargo_bars_from_pct
from ..validation.config import ValidationConfig
from ..validation.purged_kfold import PurgedKFold
from ..validation.stats import sharpe_ratio
from .dataset import MicroTrainingMatrix
from .lgbm import build_micro_estimator, fit_micro_fold

logger = get_logger(__name__)


@dataclass(frozen=True)
class MicroSearchResult:
    """Outcome of the in-CV grid search (the inputs to honest, deflated evaluation)."""

    selected_overrides: dict
    selected_index: int
    n_trials: int
    selection_metric: str
    configs: list[dict]  # the override dict per trial, in deterministic order
    per_config: list[dict]  # pooled-OOS metrics per trial
    gross_sr: np.ndarray  # (n_trials,) per-config pooled gross signal Sharpe (DSR trials)
    excess_sr: np.ndarray  # (n_trials,) per-config pooled excess-over-long Sharpe
    pbo_matrix: np.ndarray  # (n_samples, n_trials) per-sample gross signal return (for PBO)
    selected_pred: np.ndarray  # (n_samples,) pooled OOS predicted class for the winner
    scored: np.ndarray  # (n_samples,) bool — sample scored OOS (every sample once)


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Unweighted 3-class macro F1 over the fixed label set (robust to absent classes)."""
    from sklearn.metrics import f1_score

    return float(f1_score(y_true, y_pred, labels=[-1, 0, 1], average="macro", zero_division=0))


def kfold_oos(
    mtm: MicroTrainingMatrix,
    overrides: dict,
    mmcfg: MicroModelConfig,
    n_splits: int,
    embargo_pct: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Purged-KFold pooled OOS predicted CLASS for one config (every sample scored once).

    Returns ``(pred, scored)`` where ``pred`` is the predicted class in ``{-1, 0, +1}``
    at each sample's out-of-sample fold (0 where unscored) and ``scored`` is the
    boolean mask of samples that received an OOS prediction.
    """
    n = mtm.n
    embargo = embargo_bars_from_pct(embargo_pct, n)
    cv = PurgedKFold(n_splits, mtm.t0, mtm.t1, embargo_bars=embargo)
    pred = np.zeros(n, dtype=float)
    scored = np.zeros(n, dtype=bool)
    for train_idx, test_idx in cv.split():
        if train_idx.size == 0 or test_idx.size == 0:
            continue
        est = build_micro_estimator(mmcfg, overrides)
        fit_micro_fold(
            est,
            mtm.X,
            mtm.y,
            mtm.sample_weight,
            t0=mtm.t0,
            t1=mtm.t1,
            train_idx=train_idx,
            n=n,
            early_stopping=mmcfg.early_stopping,
            embargo_bars=embargo,
            class_weighting=mmcfg.class_weighting,
        )
        pred[test_idx] = est.predict(mtm.X[test_idx])
        scored[test_idx] = True
    return pred, scored


def run_micro_search(
    mtm: MicroTrainingMatrix, mmcfg: MicroModelConfig, vcfg: ValidationConfig
) -> MicroSearchResult:
    """Run the 8-config grid through purged K-fold and select the best (no leakage).

    Selection uses the pooled OOS metric named by ``search.selection_metric``
    (default ``gross_signal_sharpe``). All trials' gross/excess Sharpes and the
    per-sample gross-return matrix are returned for the deflated evaluation.
    """
    configs = mmcfg.search.configs()
    n_trials = len(configs)
    n = mtm.n
    pbo_matrix = np.zeros((n, n_trials), dtype=float)
    gross_sr = np.zeros(n_trials, dtype=float)
    excess_sr = np.zeros(n_trials, dtype=float)
    per_config: list[dict] = []
    preds: list[np.ndarray] = []
    scored_final = np.zeros(n, dtype=bool)

    for j, overrides in enumerate(configs):
        pred, scored = kfold_oos(mtm, overrides, mmcfg, vcfg.kfold.n_splits, vcfg.kfold.embargo_pct)
        scored_final = scored
        pos = pred.astype(float)  # predicted class IS the position (-1/0/+1)
        gross = pos * mtm.ret_t1
        pbo_matrix[:, j] = gross
        s = scored
        gsr = sharpe_ratio(gross[s])
        ex = (pos[s] - 1.0) * mtm.ret_t1[s]  # signal minus perma-long benchmark
        ex_sr = sharpe_ratio(ex)
        mf1 = _macro_f1(mtm.y[s], pred[s].astype(int))
        action = float(np.mean(pred[s] != 0)) if s.any() else 0.0
        gross_sr[j] = gsr
        excess_sr[j] = ex_sr
        preds.append(pred)
        per_config.append(
            {
                "overrides": overrides,
                "gross_signal_sharpe": round(gsr, 5),
                "excess_signal_sharpe": round(ex_sr, 5),
                "macro_f1": round(mf1, 4),
                "action_rate": round(action, 4),
                "mean_gross_return": round(
                    float((gross[s] * mtm.sample_weight[s]).sum() / mtm.sample_weight[s].sum()), 8
                ),
            }
        )
        logger.info(
            f"  trial {j + 1}/{n_trials} {overrides} -> gross_sr={gsr:.4f} "
            f"excess_sr={ex_sr:.4f} macroF1={mf1:.4f} action={action:.3f}"
        )

    metric_key = mmcfg.search.selection_metric  # validated ∈ known set
    scores = np.array([c[metric_key] for c in per_config], dtype=float)
    selected_index = int(np.argmax(scores))  # deterministic: first max wins

    return MicroSearchResult(
        selected_overrides=configs[selected_index],
        selected_index=selected_index,
        n_trials=n_trials,
        selection_metric=mmcfg.search.selection_metric,
        configs=configs,
        per_config=per_config,
        gross_sr=gross_sr,
        excess_sr=excess_sr,
        pbo_matrix=pbo_matrix,
        selected_pred=preds[selected_index],
        scored=scored_final,
    )
