"""Honest evaluation of the micro signal model — judged ONLY through the framework.

Every number comes from the Phase-4 leak-safe machinery (purged K-fold pooled pass,
Combinatorial Purged CV for the path distribution, and the overfitting statistics
PBO / PSR / DSR). It reports both **classification** quality (3-class: weighted +
balanced accuracy, macro F1, confusion matrix, class & prediction distributions,
action rate) and a **gross signal-return** proxy (``pred_class * ret_t1`` — NO costs,
NO slippage; this is not an executable backtest).

The output is one explicit **VERDICT** per symbol — *edge candidate* or *no
significant edge* — with the individual gate checks recorded so the call is
auditable, never massaged. "edge candidate" authorises ONLY the next phase (an
economic backtest with realistic costs); it does NOT authorise live trading.

The verdict/metric helpers are pure functions so the degenerate-model defenses
(all-flat predictions, majority-timeout classifier) are unit-tested directly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from config.settings import Settings

from ..common.logging import get_logger
from ..microstructure.model_config import MicroModelConfig, VerdictCfg
from ..validation import stats as st
from ..validation._purge import embargo_bars_from_pct
from ..validation.config import ValidationConfig
from ..validation.cpcv import CombinatorialPurgedCV
from .dataset import MicroTrainingMatrix
from .lgbm import build_micro_estimator, fit_micro_fold
from .tune import MicroSearchResult

logger = get_logger(__name__)

VERDICT_EDGE = "edge candidate"
VERDICT_NONE = "no significant edge"

_CLASSES = [-1, 0, 1]


# --------------------------------------------------------------------------- #
# Pure metric helpers (unit-tested directly)
# --------------------------------------------------------------------------- #
def action_rate(pred: np.ndarray) -> float:
    """Fraction of predictions that are non-flat (a trade signal, ``!= 0``)."""
    pred = np.asarray(pred)
    return float(np.mean(pred != 0)) if pred.size else 0.0


def _weighted_accuracy(y_true: np.ndarray, y_pred: np.ndarray, w: np.ndarray) -> float:
    sw = float(w.sum())
    if sw <= 0.0:
        return float("nan")
    return float((((y_true == y_pred).astype(float)) * w).sum() / sw)


def _class_distribution(arr: np.ndarray) -> dict[str, float]:
    """Fraction in each of {-1, 0, +1} (keys are strings for JSON)."""
    arr = np.asarray(arr)
    return {str(c): round(float(np.mean(arr == c)), 4) for c in _CLASSES}


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, w: np.ndarray) -> dict[str, Any]:
    """3-class classification metrics (weighted + balanced accuracy, macro F1, etc.)."""
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score

    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=_CLASSES)
    return {
        "weighted_accuracy": round(_weighted_accuracy(y_true, y_pred, w), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(
            float(f1_score(y_true, y_pred, labels=_CLASSES, average="macro", zero_division=0)), 4
        ),
        "confusion_matrix": {"labels": _CLASSES, "matrix": cm.astype(int).tolist()},
        "class_distribution_true": _class_distribution(y_true),
        "class_distribution_pred": _class_distribution(y_pred),
        "action_rate": round(action_rate(y_pred), 4),
    }


def signal_return_metrics(
    pred: np.ndarray,
    ret_t1: np.ndarray,
    w: np.ndarray,
    gross_sr_trials: np.ndarray,
) -> dict[str, Any]:
    """Gross signal-return proxy metrics (NO costs): Sharpe / PSR / DSR, excess-over-long.

    ``gross_signal_return = pred_class * ret_t1``; the long benchmark is ``1 * ret_t1``
    and ``excess = (pred_class - 1) * ret_t1``. DSR deflates by the trial distribution
    of per-config gross Sharpes (selection bias priced in).
    """
    pos = np.asarray(pred, dtype=float)
    ret = np.asarray(ret_t1, dtype=float)
    w = np.asarray(w, dtype=float)
    gross = pos * ret
    long_bench = ret
    excess = (pos - 1.0) * ret

    n_g, sr_g, sk_g, ku_g = st.return_moments(gross)
    psr_g = st.probabilistic_sharpe_ratio(sr_g, 0.0, n_g, sk_g, ku_g) if n_g >= 2 else None
    dsr_g = (
        st.deflated_sharpe_ratio(sr_g, np.asarray(gross_sr_trials, dtype=float), n_g, sk_g, ku_g)
        if n_g >= 2 and np.asarray(gross_sr_trials).size >= 1
        else None
    )
    n_e, sr_e, sk_e, ku_e = st.return_moments(excess)
    psr_e = st.probabilistic_sharpe_ratio(sr_e, 0.0, n_e, sk_e, ku_e) if n_e >= 2 else None
    sw = float(w.sum())
    mean_gross = float((gross * w).sum() / sw) if sw > 0 else float("nan")
    mean_excess = float((excess * w).sum() / sw) if sw > 0 else float("nan")
    return {
        "n_scored": int(pos.size),
        "gross_sharpe": round(sr_g, 5),
        "gross_psr": None if psr_g is None else round(psr_g, 5),
        "gross_dsr": None if dsr_g is None else round(dsr_g, 5),
        "mean_gross_return": round(mean_gross, 8),
        "excess_over_long_sharpe": round(sr_e, 5),
        "excess_over_long_psr": None if psr_e is None else round(psr_e, 5),
        "mean_excess_over_long_return": round(mean_excess, 8),
        "long_benchmark_sharpe": round(st.sharpe_ratio(long_bench), 5),
    }


def decide_verdict(
    *,
    pbo: float | None,
    gross_dsr: float | None,
    mean_gross_return: float | None,
    action_rate_value: float,
    macro_f1: float,
    cpcv_median_gross_sharpe: float | None,
    v: VerdictCfg,
) -> tuple[str, dict[str, bool]]:
    """Apply the fixed verdict gates. ALL must pass for an 'edge candidate'.

    A missing statistic (None) fails its gate — the absence of evidence is never
    counted as evidence of an edge.
    """
    checks = {
        "pbo_below_max": bool(pbo is not None and pbo < v.max_pbo),
        "gross_dsr_above_min": bool(gross_dsr is not None and gross_dsr > v.min_gross_dsr),
        "positive_gross_return": bool(
            (not v.require_positive_gross_return)
            or (mean_gross_return is not None and mean_gross_return > 0.0)
        ),
        "action_rate_above_min": bool(action_rate_value >= v.min_action_rate),
        "macro_f1_above_min": bool(macro_f1 >= v.min_macro_f1),
        "cpcv_median_gross_sharpe_positive": bool(
            (not v.require_cpcv_median_gross_sharpe_positive)
            or (cpcv_median_gross_sharpe is not None and cpcv_median_gross_sharpe > 0.0)
        ),
    }
    verdict = VERDICT_EDGE if all(checks.values()) else VERDICT_NONE
    return verdict, checks


# --------------------------------------------------------------------------- #
# CPCV out-of-sample path distribution (selected config)
# --------------------------------------------------------------------------- #
def _distribution_summary(values: list[float]) -> dict[str, float | None]:
    a = np.asarray(values, dtype=float)
    if a.size == 0:
        return {"mean": None, "median": None, "min": None, "max": None, "std": None}
    return {
        "mean": round(float(a.mean()), 5),
        "median": round(float(np.median(a)), 5),
        "min": round(float(a.min()), 5),
        "max": round(float(a.max()), 5),
        "std": None if a.size < 2 else round(float(a.std(ddof=1)), 5),
    }


def _cpcv_distribution(
    mtm: MicroTrainingMatrix, overrides: dict, mmcfg: MicroModelConfig, vcfg: ValidationConfig
) -> list[dict[str, float]]:
    """Per-OOS-path gross/excess Sharpe + macro F1 for the selected config via CPCV."""
    from sklearn.metrics import f1_score

    n = mtm.n
    embargo = embargo_bars_from_pct(vcfg.cpcv.embargo_pct, n)
    cv = CombinatorialPurgedCV(
        vcfg.cpcv.n_groups, vcfg.cpcv.test_groups, mtm.t0, mtm.t1, embargo_bars=embargo
    )
    n_paths, mapping = cv.assign_paths()
    buffers: list[dict[int, tuple[np.ndarray, np.ndarray]]] = [{} for _ in range(n_paths)]

    for split_index, (train_idx, _test) in enumerate(cv.split()):
        if train_idx.size == 0:
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
        for g in cv.combos[split_index]:
            g_idx = cv.groups[g]
            buffers[mapping[(split_index, g)]][g] = (g_idx, est.predict(mtm.X[g_idx]))

    paths: list[dict[str, float]] = []
    for path in buffers:
        if len(path) != cv.n_groups:
            continue
        order = sorted(path)
        idx = np.concatenate([path[g][0] for g in order])
        pred = np.concatenate([path[g][1] for g in order])
        pos = pred.astype(float)
        gross = pos * mtm.ret_t1[idx]
        excess = (pos - 1.0) * mtm.ret_t1[idx]
        paths.append(
            {
                "gross_sharpe": round(st.sharpe_ratio(gross), 5),
                "excess_over_long_sharpe": round(st.sharpe_ratio(excess), 5),
                "macro_f1": round(
                    float(
                        f1_score(
                            mtm.y[idx],
                            pred.astype(int),
                            labels=_CLASSES,
                            average="macro",
                            zero_division=0,
                        )
                    ),
                    4,
                ),
                "action_rate": round(action_rate(pred), 4),
                "effective_n": round(float(mtm.uniqueness_weight[idx].sum()), 2),
            }
        )
    return paths


# --------------------------------------------------------------------------- #
# Full evaluation
# --------------------------------------------------------------------------- #
def evaluate_micro_model(
    mtm: MicroTrainingMatrix,
    search: MicroSearchResult,
    mmcfg: MicroModelConfig,
    vcfg: ValidationConfig,
) -> dict[str, Any]:
    """Full honest evaluation of the selected config → a JSON-able summary + VERDICT."""
    s = search.scored
    pred = search.selected_pred
    y_s = mtm.y[s]
    pred_s = pred[s].astype(int)

    classification = classification_metrics(y_s, pred_s, mtm.sample_weight[s])
    signal = signal_return_metrics(pred_s, mtm.ret_t1[s], mtm.sample_weight[s], search.gross_sr)

    # --- PBO across the search configs (selection-overfit gate) ---
    pbo: dict[str, Any] | None = None
    if search.n_trials >= 2:
        n_part = min(10, (search.pbo_matrix.shape[0] // 2) * 2)
        if n_part >= 2:
            res = st.probability_of_backtest_overfitting(search.pbo_matrix, n_partitions=n_part)
            pbo = {
                "pbo": round(float(res["pbo"]), 4),
                "n_combinations": int(res["n_combinations"]),
                "n_partitions": n_part,
                "n_configs": search.n_trials,
            }

    # --- CPCV OOS path distribution (selected config) ---
    paths = _cpcv_distribution(mtm, search.selected_overrides, mmcfg, vcfg)
    cpcv = {
        "n_paths": len(paths),
        "gross_sharpe": _distribution_summary([p["gross_sharpe"] for p in paths]),
        "excess_over_long_sharpe": _distribution_summary(
            [p["excess_over_long_sharpe"] for p in paths]
        ),
        "macro_f1_mean": (
            None if not paths else round(float(np.mean([p["macro_f1"] for p in paths])), 4)
        ),
        "action_rate_mean": (
            None if not paths else round(float(np.mean([p["action_rate"] for p in paths])), 4)
        ),
        "effective_n_mean": (
            None if not paths else round(float(np.mean([p["effective_n"] for p in paths])), 2)
        ),
        "paths": paths,
    }

    # --- VERDICT (explicit, auditable, never massaged) ---
    cpcv_median = cpcv["gross_sharpe"]["median"]
    verdict, checks = decide_verdict(
        pbo=pbo["pbo"] if pbo else None,
        gross_dsr=signal["gross_dsr"],
        mean_gross_return=signal["mean_gross_return"],
        action_rate_value=classification["action_rate"],
        macro_f1=classification["macro_f1"],
        cpcv_median_gross_sharpe=cpcv_median,
        v=mmcfg.verdict,
    )

    return {
        "symbol": mtm.symbol,
        "verdict": verdict,
        "verdict_checks": checks,
        "verdict_thresholds": mmcfg.verdict.model_dump(mode="json"),
        "micro_model_version": mtm.micro_model_version,
        "microstructure_feature_version": mtm.microstructure_feature_version,
        "micro_label_version": mtm.micro_label_version,
        "validation_version": vcfg.validation_version,
        "target_mode": mmcfg.target.mode,
        "class_weighting": mmcfg.class_weighting.model_dump(mode="json"),
        "n_samples": mtm.n,
        "effective_n": round(mtm.effective_n, 2),
        "n_features": len(mtm.feature_cols),
        "feature_cols": list(mtm.feature_cols),
        "n_trials": search.n_trials,
        "selection_metric": search.selection_metric,
        "selected_overrides": search.selected_overrides,
        "search_configs": search.per_config,
        "class_balance": classification["class_distribution_true"],
        "pred_balance": classification["class_distribution_pred"],
        "action_rate": classification["action_rate"],
        "classification": classification,
        "signal_return": signal,
        "pbo": pbo,
        "cpcv": cpcv,
        "cost_disclaimer": "gross signal-return proxy only; not an executable backtest "
        "(no commissions or slippage)",
        "micro_model_config": mmcfg.to_dict(),
    }


# --------------------------------------------------------------------------- #
# Run persistence (consumed by observability/micro_model_health.py)
# --------------------------------------------------------------------------- #
def _runs_dir() -> Path:
    return Path(Settings().data_dir) / "micro_models" / "runs"


def save_micro_run(summary: dict[str, Any], *, runs_dir: Path | None = None) -> Path:
    """Persist one micro-model evaluation summary as JSON under ``data/micro_models/runs/``."""
    d = runs_dir or _runs_dir()
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{summary['symbol']}.json"
    with path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)
    return path


def read_micro_run(symbol: str, *, runs_dir: Path | None = None) -> dict[str, Any] | None:
    """Load a saved micro-model evaluation summary, or ``None`` if none exists."""
    path = (runs_dir or _runs_dir()) / f"{symbol}.json"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)
