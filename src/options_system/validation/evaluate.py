"""Evaluation harness — judge an estimator through the leak-safe machinery.

This is where the splitters, the sample weights and the overfitting statistics
come together. Given one or more sklearn-compatible estimators and the leak-free
``(features@t0, y, t1, weight)`` matrix, it:

* runs **purged + embargoed K-fold** so every sample is scored out-of-sample
  exactly once → a pooled OOS return series per estimator, and a
  ``(periods × estimators)`` performance matrix for **PBO**;
* runs **Combinatorial Purged CV** → a *distribution* of OOS paths per estimator
  (mean/median/spread of the Sharpe across ``C(N-1,k-1)`` recombined paths);
* honours the Phase-3 uniqueness **sample weights** in every fit and weighted
  metric, and reports **effective sample size** (Σ average-uniqueness) per fold
  and per path — because ~11k overlapping labels carry only ~2.7k independent
  observations;
* computes **PSR** and **DSR** (deflating for the number of estimators tried).

It is model-agnostic: pass any estimator exposing ``fit``/``predict`` (and
optionally ``predict_proba``). The shipped baselines are a most-frequent dummy and
a standardised logistic regression — deliberately unskilled, so on real data they
land near chance. If a dummy ever looks profitable through this harness, something
leaks: stop and investigate before trusting anything downstream.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from config.settings import Settings

from ..common.logging import get_logger
from ..data.store import DuckStore
from ..features.compute import feature_names
from ..features.config import FeatureConfig
from ..labeling.build import labels_with_features
from . import stats as stats_mod
from ._purge import embargo_bars_from_pct
from .config import ValidationConfig
from .cpcv import CombinatorialPurgedCV
from .purged_kfold import PurgedKFold

logger = get_logger(__name__)

_WIDE_START = datetime(2000, 1, 1, tzinfo=UTC)
_WIDE_END = datetime(2100, 1, 1, tzinfo=UTC)


# --------------------------------------------------------------------------- #
# Aligned, leak-free design matrix
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Matrix:
    """A leak-free, ``t0``-sorted ``(features@t0, y, t1, weight)`` design matrix."""

    symbol: str
    X: np.ndarray  # (n, p) features as-of t0
    y: np.ndarray  # (n,) labels in {-1, 0, +1}
    t0: np.ndarray  # (n,) event time
    t1: np.ndarray  # (n,) barrier-resolution time (drives purge/embargo)
    ret: np.ndarray  # (n,) realized log-return to t1 (return proxy)
    weight: np.ndarray  # (n,) persisted sample weight (≈ mean 1.0)
    uniqueness: np.ndarray  # (n,) average uniqueness (effective-N building block)
    feature_cols: list[str]

    @property
    def n(self) -> int:
        return int(self.y.shape[0])


def load_matrix(
    symbol: str,
    *,
    start: datetime | None = None,
    end: datetime | None = None,
    store: DuckStore | None = None,
) -> Matrix:
    """Assemble the aligned matrix for ``symbol`` via the leak-free labels↔features join.

    Uses :func:`options_system.labeling.build.labels_with_features` (features attached
    strictly as-of ``t0``), drops warmup rows with any null feature/label/weight, and
    returns ``t0``-sorted numpy arrays ready for the splitters.
    """
    start = start or _WIDE_START
    end = end or _WIDE_END
    m = labels_with_features(symbol, start, end, store=store)
    if m.is_empty():
        raise ValueError(
            f"no labels/features for {symbol} in [{start:%Y-%m-%d}, {end:%Y-%m-%d}] — "
            "build labels first (python -m options_system.labeling.build)"
        )
    feat_cols = [c for c in feature_names(FeatureConfig.load()) if c in m.columns]
    if not feat_cols:
        raise ValueError(f"no feature columns present for {symbol}; build features first")
    required = ["label", "ret", "weight", "avg_uniqueness", "t1", *feat_cols]
    m = m.sort("t0").drop_nulls(subset=required)
    # Drop rows with any non-finite feature: degenerate windows (e.g. a z-score with
    # zero rolling std) can emit ±inf/NaN, which sklearn estimators reject. These are
    # warmup/degenerate edges, not signal — dropping them keeps the matrix finite.
    if feat_cols:
        m = m.filter(pl.all_horizontal(pl.col(c).is_finite() for c in feat_cols))
    if m.is_empty():
        raise ValueError(f"all rows for {symbol} dropped (null/non-finite) after feature attach")
    return Matrix(
        symbol=symbol,
        X=m.select(feat_cols).to_numpy(),
        y=m["label"].to_numpy().astype(int),
        t0=m["t0"].to_numpy(),
        t1=m["t1"].to_numpy(),
        ret=m["ret"].to_numpy().astype(float),
        weight=m["weight"].to_numpy().astype(float),
        uniqueness=m["avg_uniqueness"].to_numpy().astype(float),
        feature_cols=feat_cols,
    )


# --------------------------------------------------------------------------- #
# Baselines + estimator plumbing
# --------------------------------------------------------------------------- #
def baseline_estimators(seed: int = 0) -> dict[str, Any]:
    """The two deliberately-unskilled baselines: most-frequent dummy + logistic.

    The logistic baseline is wrapped in a standardiser **inside** a pipeline, so the
    scaler is fit on each training fold only (no test-distribution leakage).
    """
    from sklearn.dummy import DummyClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    return {
        "dummy": DummyClassifier(strategy="most_frequent"),
        "logistic": Pipeline(
            [
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(max_iter=500, C=0.5, random_state=seed)),
            ]
        ),
    }


def _fit(est: Any, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> Any:
    """Fit ``est`` honouring sample weights (routing to a pipeline's final step)."""
    if isinstance(est, Pipeline):
        final = est.steps[-1][0]
        est.fit(X, y, **{f"{final}__sample_weight": w})
    else:
        est.fit(X, y, sample_weight=w)
    return est


def _positions(pred: np.ndarray) -> np.ndarray:
    """Map predicted labels {-1,0,+1} to trading positions (the sign)."""
    return np.sign(pred).astype(float)


def _weighted_accuracy(y_true: np.ndarray, y_pred: np.ndarray, w: np.ndarray) -> float:
    sw = float(w.sum())
    if sw <= 0.0:
        return float("nan")
    return float((((y_true == y_pred).astype(float)) * w).sum() / sw)


def _auc(est: Any, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> float | None:
    """Macro one-vs-rest AUC, or ``None`` when undefined (single class present, etc.)."""
    from sklearn.metrics import roc_auc_score

    try:
        proba = est.predict_proba(X)
        if len(np.unique(y)) < 2:
            return None
        return float(
            roc_auc_score(
                y, proba, multi_class="ovr", average="macro", labels=est.classes_, sample_weight=w
            )
        )
    except Exception:  # noqa: BLE001 - AUC is genuinely undefined for degenerate folds
        return None


def _effective_n(uniqueness: np.ndarray, idx: np.ndarray) -> float:
    """Effective sample size of a fold/path = Σ average-uniqueness over its samples."""
    return float(uniqueness[idx].sum())


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def _kfold_pass(
    estimators: dict[str, Any], m: Matrix, cfg: ValidationConfig
) -> tuple[dict[str, dict], np.ndarray, list[str]]:
    """Purged-KFold: every sample OOS once → per-estimator pooled metrics + a PBO matrix.

    Returns ``(per_estimator, pbo_matrix, est_order)`` where ``pbo_matrix`` is
    ``(n_samples, n_estimators)`` of per-sample return proxies (position × ret).
    """
    embargo = embargo_bars_from_pct(cfg.kfold.embargo_pct, m.n)
    cv = PurgedKFold(cfg.kfold.n_splits, m.t0, m.t1, embargo_bars=embargo)
    est_order = list(estimators)
    pbo_matrix = np.zeros((m.n, len(est_order)), dtype=float)
    pooled_pred = {name: np.zeros(m.n, dtype=float) for name in est_order}
    scored = np.zeros(m.n, dtype=bool)
    fold_reports = cv.split_report()

    for train_idx, test_idx in cv.split():
        if train_idx.size == 0 or test_idx.size == 0:
            continue
        scored[test_idx] = True
        for name in est_order:
            est = _fit(clone(estimators[name]), m.X[train_idx], m.y[train_idx], m.weight[train_idx])
            pooled_pred[name][test_idx] = est.predict(m.X[test_idx])

    per_estimator: dict[str, dict] = {}
    for j, name in enumerate(est_order):
        pos = _positions(pooled_pred[name])
        ret_series = (pos * m.ret)[scored]
        pbo_matrix[:, j] = pos * m.ret
        n_obs, sr, skew, kurt = stats_mod.return_moments(ret_series)
        psr = (
            stats_mod.probabilistic_sharpe_ratio(sr, 0.0, n_obs, skew, kurt) if n_obs >= 2 else None
        )
        per_estimator[name] = {
            "n_scored": int(scored.sum()),
            "effective_n": round(_effective_n(m.uniqueness, np.flatnonzero(scored)), 2),
            "accuracy": round(
                _weighted_accuracy(
                    m.y[scored], pooled_pred[name][scored].astype(int), m.weight[scored]
                ),
                4,
            ),
            "sharpe": round(sr, 5),
            "psr": None if psr is None else round(psr, 5),
            "mean_weighted_return": round(
                float(
                    (pos[scored] * m.ret[scored] * m.weight[scored]).sum() / m.weight[scored].sum()
                ),
                8,
            ),
        }

    # Deflated Sharpe: the estimators are the trials selected among.
    sr_trials = np.array([per_estimator[name]["sharpe"] for name in est_order], dtype=float)
    for name in est_order:
        n_obs = per_estimator[name]["n_scored"]
        sr = per_estimator[name]["sharpe"]
        _, _, skew, kurt = stats_mod.return_moments((_positions(pooled_pred[name]) * m.ret)[scored])
        dsr = (
            stats_mod.deflated_sharpe_ratio(sr, sr_trials, n_obs, skew, kurt)
            if n_obs >= 2
            else None
        )
        per_estimator[name]["dsr"] = None if dsr is None else round(dsr, 5)
        per_estimator[name]["n_trials"] = len(est_order)

    for name in est_order:
        per_estimator[name]["fold_report"] = fold_reports
    return per_estimator, pbo_matrix, est_order


def _cpcv_pass(estimators: dict[str, Any], m: Matrix, cfg: ValidationConfig) -> dict[str, dict]:
    """Combinatorial Purged CV → per-estimator distribution of OOS-path Sharpes."""
    embargo = embargo_bars_from_pct(cfg.cpcv.embargo_pct, m.n)
    cv = CombinatorialPurgedCV(
        cfg.cpcv.n_groups, cfg.cpcv.test_groups, m.t0, m.t1, embargo_bars=embargo
    )
    n_paths, mapping = cv.assign_paths()
    # path_buffers[name][path][group] = (sample_idx, predicted_labels)
    path_buffers: dict[str, list[dict[int, tuple[np.ndarray, np.ndarray]]]] = {
        name: [{} for _ in range(n_paths)] for name in estimators
    }

    for split_index, (train_idx, _test_idx) in enumerate(cv.split()):
        if train_idx.size == 0:
            continue
        combo = cv.combos[split_index]
        for name in estimators:
            est = _fit(clone(estimators[name]), m.X[train_idx], m.y[train_idx], m.weight[train_idx])
            for g in combo:
                g_idx = cv.groups[g]
                path_j = mapping[(split_index, g)]
                path_buffers[name][path_j][g] = (g_idx, est.predict(m.X[g_idx]))

    out: dict[str, dict] = {}
    for name in estimators:
        path_sharpes: list[float] = []
        path_accs: list[float] = []
        path_eff_n: list[float] = []
        for path in path_buffers[name]:
            if len(path) != cv.n_groups:
                continue  # incomplete path (a split was skipped) — drop it honestly
            order = sorted(path)  # groups in ascending (time) order
            idx = np.concatenate([path[g][0] for g in order])
            pred = np.concatenate([path[g][1] for g in order])
            ret_series = _positions(pred) * m.ret[idx]
            path_sharpes.append(stats_mod.sharpe_ratio(ret_series))
            path_accs.append(_weighted_accuracy(m.y[idx], pred.astype(int), m.weight[idx]))
            path_eff_n.append(_effective_n(m.uniqueness, idx))
        sharpes = np.asarray(path_sharpes, dtype=float)
        out[name] = {
            "n_paths": int(sharpes.size),
            "sharpe_mean": None if sharpes.size == 0 else round(float(sharpes.mean()), 5),
            "sharpe_std": None if sharpes.size < 2 else round(float(sharpes.std(ddof=1)), 5),
            "sharpe_min": None if sharpes.size == 0 else round(float(sharpes.min()), 5),
            "sharpe_median": None if sharpes.size == 0 else round(float(np.median(sharpes)), 5),
            "sharpe_max": None if sharpes.size == 0 else round(float(sharpes.max()), 5),
            "accuracy_mean": (None if not path_accs else round(float(np.nanmean(path_accs)), 4)),
            "effective_n_mean": (None if not path_eff_n else round(float(np.mean(path_eff_n)), 2)),
        }
    return out


def evaluate(
    estimators: dict[str, Any],
    m: Matrix,
    cfg: ValidationConfig | None = None,
) -> dict[str, Any]:
    """Run the full leak-safe evaluation for ``m`` and return a JSON-able summary.

    Combines a purged-KFold pooled pass (per-estimator metrics + PBO across the
    estimators) with a CPCV path-distribution pass. Deterministic under the config
    seed.
    """
    cfg = cfg or ValidationConfig.load()
    if len(estimators) == 0:
        raise ValueError("evaluate needs at least one estimator")

    kfold, pbo_matrix, est_order = _kfold_pass(estimators, m, cfg)
    cpcv = _cpcv_pass(estimators, m, cfg)

    pbo: dict[str, Any] | None = None
    if len(est_order) >= 2:
        n_part = min(10, (pbo_matrix.shape[0] // 2) * 2)
        if n_part >= 2:
            res = stats_mod.probability_of_backtest_overfitting(pbo_matrix, n_partitions=n_part)
            pbo = {
                "pbo": round(float(res["pbo"]), 4),
                "n_combinations": int(res["n_combinations"]),
                "estimators": est_order,
                "n_partitions": n_part,
            }

    return {
        "symbol": m.symbol,
        "validation_version": cfg.validation_version,
        "n_samples": m.n,
        "n_features": len(m.feature_cols),
        "effective_n_total": round(float(m.uniqueness.sum()), 2),
        "config": cfg.to_dict(),
        "kfold": kfold,
        "cpcv": cpcv,
        "pbo": pbo,
    }


# --------------------------------------------------------------------------- #
# Run persistence (consumed by observability/validation_health.py)
# --------------------------------------------------------------------------- #
def _runs_dir() -> Path:
    return Path(Settings().data_dir) / "validation"


def save_run(summary: dict[str, Any], *, runs_dir: Path | None = None) -> Path:
    """Persist one evaluation summary as JSON under ``data/validation/<symbol>.json``."""
    d = runs_dir or _runs_dir()
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{summary['symbol']}.json"
    with path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)
    return path


def read_run(symbol: str, *, runs_dir: Path | None = None) -> dict[str, Any] | None:
    """Load a saved evaluation summary, or ``None`` if no run exists for ``symbol``."""
    path = (runs_dir or _runs_dir()) / f"{symbol}.json"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def main() -> None:
    """CLI: evaluate the baselines on each symbol and save the runs.

    ``uv run python -m options_system.validation.evaluate --symbols MES MNQ``
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the leak-safe validation harness (baselines)."
    )
    parser.add_argument("--symbols", nargs="+", default=Settings().record_symbols)
    args = parser.parse_args()

    cfg = ValidationConfig.load()
    for symbol in args.symbols:
        logger.info(f"evaluating {symbol} (validation_version={cfg.validation_version})")
        m = load_matrix(symbol)
        summary = evaluate(baseline_estimators(seed=cfg.evaluation.seed), m, cfg)
        path = save_run(summary)
        pbo = summary["pbo"]["pbo"] if summary["pbo"] else "n/a"
        logger.info(
            f"{symbol}: n={summary['n_samples']} eff_n={summary['effective_n_total']} PBO={pbo}"
        )
        for name, met in summary["kfold"].items():
            logger.info(
                f"  {name}: acc={met['accuracy']} sharpe={met['sharpe']} "
                f"PSR={met['psr']} DSR={met['dsr']}"
            )
        logger.info(f"  saved -> {path}")


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
