"""Honest evaluation of the tuned signal model — judged ONLY through the framework.

Every number here comes from the Phase-4 leak-safe machinery (purged K-fold for the
pooled pass, Combinatorial Purged CV for the path distribution, and the overfitting
statistics PBO / PSR / DSR). This module adds nothing to the *leakage* logic — it
composes the existing splitters and stats — but it does two things the baseline
harness deliberately deferred:

* **Skill is reported over and above market beta.** The return metric is the
  *excess* of the model's position-weighted return over a **perma-long benchmark**
  (information-ratio style): ``excess = (position - 1) · ret``. A model that is
  effectively always-long has zero excess and so shows no edge, however much beta it
  rode. The Deflated Sharpe is computed on this excess series.
* **Tuning is paid for.** The number of search configurations (``n_trials``) deflates
  the Sharpe (DSR) and the configs feed the PBO — selection bias is priced in.

The output is one explicit **VERDICT** per symbol — *edge* or *no significant edge* —
with the individual gate checks recorded so the call is auditable, never massaged.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from config.settings import Settings

from ..common.logging import get_logger
from ..validation import stats as st
from ..validation._purge import embargo_bars_from_pct
from ..validation.config import ValidationConfig
from ..validation.cpcv import CombinatorialPurgedCV
from .config import ModelConfig
from .dataset import TrainingMatrix
from .lgbm import build_estimator, fit_fold
from .tune import SearchResult, _weighted_accuracy

logger = get_logger(__name__)

VERDICT_EDGE = "edge"
VERDICT_NONE = "no significant edge"


def _cpcv_distribution(
    tm: TrainingMatrix, overrides: dict, mcfg: ModelConfig, vcfg: ValidationConfig
) -> list[dict[str, float]]:
    """Per-OOS-path metrics for the selected config via Combinatorial Purged CV.

    Returns one dict per reconstructed path: excess-over-long Sharpe, raw strategy
    Sharpe, weighted directional accuracy and effective N. Incomplete paths (a split
    skipped) are dropped honestly.
    """
    n = tm.n
    embargo = embargo_bars_from_pct(vcfg.cpcv.embargo_pct, n)
    cv = CombinatorialPurgedCV(
        vcfg.cpcv.n_groups, vcfg.cpcv.test_groups, tm.t0, tm.t1, embargo_bars=embargo
    )
    n_paths, mapping = cv.assign_paths()
    buffers: list[dict[int, tuple[np.ndarray, np.ndarray]]] = [{} for _ in range(n_paths)]

    for split_index, (train_idx, _test) in enumerate(cv.split()):
        if train_idx.size == 0:
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
        for g in cv.combos[split_index]:
            g_idx = cv.groups[g]
            buffers[mapping[(split_index, g)]][g] = (g_idx, est.predict(tm.X[g_idx]))

    paths: list[dict[str, float]] = []
    for path in buffers:
        if len(path) != cv.n_groups:
            continue
        order = sorted(path)
        idx = np.concatenate([path[g][0] for g in order])
        pred = np.concatenate([path[g][1] for g in order])
        pos = pred.astype(float)
        strat = pos * tm.ret[idx]
        excess = (pos - 1.0) * tm.ret[idx]
        paths.append(
            {
                "excess_sharpe": round(st.sharpe_ratio(excess), 5),
                "strategy_sharpe": round(st.sharpe_ratio(strat), 5),
                "directional_accuracy": round(
                    _weighted_accuracy(tm.y_dir[idx], pred.astype(int), tm.weight[idx]), 4
                ),
                "effective_n": round(float(tm.uniqueness[idx].sum()), 2),
            }
        )
    return paths


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


def evaluate_model(
    tm: TrainingMatrix,
    search: SearchResult,
    mcfg: ModelConfig,
    vcfg: ValidationConfig,
) -> dict[str, Any]:
    """Full honest evaluation of the selected config → a JSON-able summary + VERDICT."""
    scored = search.scored
    pred = search.selected_pred
    pos = pred.astype(float)
    ret = tm.ret
    s = scored

    # --- pooled purged-KFold metrics (every sample scored OOS once) ---
    strat = (pos * ret)[s]
    excess = ((pos - 1.0) * ret)[s]
    long_bench = ret[s]
    directional_acc = _weighted_accuracy(tm.y_dir[s], pred[s].astype(int), tm.weight[s])

    n_strat, sr_strat, sk_strat, ku_strat = st.return_moments(strat)
    psr_strat = (
        st.probabilistic_sharpe_ratio(sr_strat, 0.0, n_strat, sk_strat, ku_strat)
        if n_strat >= 2
        else None
    )
    n_ex, sr_ex, sk_ex, ku_ex = st.return_moments(excess)
    psr_excess = (
        st.probabilistic_sharpe_ratio(sr_ex, 0.0, n_ex, sk_ex, ku_ex) if n_ex >= 2 else None
    )
    # Deflated Sharpe of the EXCESS series, deflated by the trial distribution of
    # excess Sharpes across all search configs (n_trials selection bias priced in).
    dsr_excess = (
        st.deflated_sharpe_ratio(sr_ex, search.excess_sr, n_ex, sk_ex, ku_ex)
        if n_ex >= 2 and search.excess_sr.size >= 1
        else None
    )
    sw = float(tm.weight[s].sum())
    mean_excess = float((excess * tm.weight[s]).sum() / sw) if sw > 0 else float("nan")
    long_sr = st.sharpe_ratio(long_bench)

    pooled = {
        "n_scored": int(s.sum()),
        "directional_accuracy": round(directional_acc, 4),
        "strategy_sharpe": round(sr_strat, 5),
        "strategy_psr": None if psr_strat is None else round(psr_strat, 5),
        "excess_sharpe": round(sr_ex, 5),
        "excess_psr": None if psr_excess is None else round(psr_excess, 5),
        "excess_dsr": None if dsr_excess is None else round(dsr_excess, 5),
        "mean_excess_return": round(mean_excess, 8),
        "long_benchmark_sharpe": round(long_sr, 5),
        "n_trials": search.n_trials,
    }

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

    # --- CPCV out-of-sample path distribution (selected config) ---
    paths = _cpcv_distribution(tm, search.selected_overrides, mcfg, vcfg)
    cpcv = {
        "n_paths": len(paths),
        "excess_sharpe": _distribution_summary([p["excess_sharpe"] for p in paths]),
        "strategy_sharpe": _distribution_summary([p["strategy_sharpe"] for p in paths]),
        "directional_accuracy_mean": (
            None
            if not paths
            else round(float(np.mean([p["directional_accuracy"] for p in paths])), 4)
        ),
        "effective_n_mean": (
            None if not paths else round(float(np.mean([p["effective_n"] for p in paths])), 2)
        ),
        "paths": paths,
    }

    # --- VERDICT (explicit, auditable, never massaged) ---
    v = mcfg.verdict
    checks = {
        "directional_accuracy_beats_chance": bool(directional_acc > v.min_directional_accuracy),
        "pbo_below_max": bool(pbo is not None and pbo["pbo"] < v.max_pbo),
        "excess_dsr_above_min": bool(dsr_excess is not None and dsr_excess > v.min_excess_dsr),
        "positive_excess_over_beta": bool((not v.require_positive_excess) or mean_excess > 0.0),
    }
    verdict = VERDICT_EDGE if all(checks.values()) else VERDICT_NONE

    return {
        "symbol": tm.symbol,
        "verdict": verdict,
        "verdict_checks": checks,
        "verdict_thresholds": v.model_dump(mode="json"),
        "model_version": mcfg.model_version,
        "feature_version": tm.feature_version,
        "label_version": tm.label_version,
        "validation_version": vcfg.validation_version,
        "macro_feature_version": tm.macro_feature_version,
        "with_macro": tm.with_macro,
        "ta_feature_version": tm.ta_feature_version,
        "with_ta": tm.with_ta,
        "timeout_handling": tm.timeout_handling,
        "n_samples": tm.n,
        "n_features": len(tm.feature_cols),
        "n_macro_features": len(tm.macro_cols),
        "n_ta_features": len(tm.ta_cols),
        "effective_n_total": round(float(tm.uniqueness.sum()), 2),
        "n_trials": search.n_trials,
        "selection_metric": search.selection_metric,
        "selected_overrides": search.selected_overrides,
        "search_configs": search.per_config,
        "pooled_kfold": pooled,
        "pbo": pbo,
        "cpcv": cpcv,
        "model_config": mcfg.to_dict(),
    }


# --------------------------------------------------------------------------- #
# Run persistence (consumed by observability/model_health.py)
# --------------------------------------------------------------------------- #
def _runs_dir() -> Path:
    return Path(Settings().data_dir) / "models" / "runs"


def save_model_run(
    summary: dict[str, Any], *, runs_dir: Path | None = None, suffix: str = ""
) -> Path:
    """Persist one model-evaluation summary as JSON under ``data/models/runs/``.

    The canonical price+macro run is ``<symbol>.json`` (``suffix=""``); the
    price-only baseline is saved under ``<symbol>_price_only.json`` so the Phase-6
    comparison keeps both side by side (model_health reads ``<symbol>.json``).
    """
    d = runs_dir or _runs_dir()
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{summary['symbol']}{suffix}.json"
    with path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)
    return path


def read_model_run(
    symbol: str, *, runs_dir: Path | None = None, suffix: str = ""
) -> dict[str, Any] | None:
    """Load a saved model-evaluation summary, or ``None`` if none exists for ``symbol``."""
    path = (runs_dir or _runs_dir()) / f"{symbol}{suffix}.json"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)
