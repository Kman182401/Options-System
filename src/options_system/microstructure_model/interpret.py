"""Optional SHAP interpretability for the 3-class micro signal model.

The prime directive says a human must be able to *explain* what the model does. SHAP
attributes predictions to features so we can see which order-flow drivers matter and
sanity-check that none dominates in a way that smells like leakage. For the multiclass
model the per-class attributions are aggregated to a single global importance
(mean ``|SHAP|`` over samples AND classes).

This is **interpretation, not performance evidence** — those numbers come only from
the leak-safe CV in ``evaluate.py``. The model is refit on the FULL matrix here
purely for the global SHAP view (the standard, correct thing to do). It is fully
OPTIONAL: if ``shap`` (or the plot backend) is unavailable, :func:`explain` returns
``{"available": False, ...}`` and never raises — the run continues.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..common.logging import get_logger
from ..microstructure.model_config import MicroModelConfig
from .dataset import MicroTrainingMatrix
from .lgbm import build_micro_estimator, effective_sample_weight, fold_local_class_weights

logger = get_logger(__name__)

# Above this share of total mean|SHAP| a single feature is "dominating" — worth a
# leakage second-look (the features are causal + leak-tested, so this should be low).
_DOMINANCE_THRESHOLD = 0.5


def _mean_abs_shap(values: Any, n_features: int) -> np.ndarray:
    """Reduce SHAP output (list per class / 3-D / 2-D) to per-feature mean ``|SHAP|``.

    Multiclass TreeExplainer returns either a list ``[class0, class1, class2]`` of
    ``(n, p)`` arrays, a 3-D array ``(n, p, n_classes)``, or (binary fallback) a 2-D
    array. We average ``|SHAP|`` over samples AND classes to one importance per
    feature.
    """
    if isinstance(values, list):
        arr = np.stack([np.asarray(v) for v in values], axis=0)  # (n_classes, n, p)
        mean_abs = np.abs(arr).mean(axis=(0, 1))
    else:
        arr = np.asarray(values)
        if arr.ndim == 3:  # (n, p, n_classes)
            mean_abs = np.abs(arr).mean(axis=(0, 2))
        elif arr.ndim == 2:  # (n, p)
            mean_abs = np.abs(arr).mean(axis=0)
        else:
            raise ValueError(f"unexpected SHAP shape {arr.shape}")
    if mean_abs.shape[0] != n_features:
        raise ValueError(f"SHAP feature count {mean_abs.shape[0]} != {n_features}")
    return mean_abs


def explain(
    mtm: MicroTrainingMatrix,
    mmcfg: MicroModelConfig,
    overrides: dict | None = None,
    *,
    max_samples: int = 2000,
    out_png: Path | None = None,
    estimator: Any = None,
) -> dict[str, Any]:
    """Global SHAP importances for the selected config on ``mtm`` (interpretation only).

    Returns a JSON-able dict: ranked global importances (mean ``|SHAP|`` over samples
    and classes), the top feature's share (dominance/leakage smell check), and the
    plot path. NEVER raises: any failure (shap missing, plotting error) is caught and
    surfaced as ``{"available": False, "reason": ...}``.
    """
    try:
        import shap

        feat_cols = mtm.feature_cols
        p = len(feat_cols)
        est = estimator
        if est is None:
            est = build_micro_estimator(mmcfg, overrides or {})
            # In-sample explanation fit with GLOBAL class weights (explicitly not a
            # performance number — the leak-safe CV owns those).
            cw = fold_local_class_weights(
                mtm.y,
                mtm.sample_weight,
                use_sample_weight=mmcfg.class_weighting.use_sample_weight_in_balance,
            )
            est.fit(
                mtm.X, mtm.y, sample_weight=effective_sample_weight(mtm.y, mtm.sample_weight, cw)
            )

        rng = np.random.default_rng(mmcfg.seed)
        n = mtm.n
        sel = (
            np.arange(n)
            if n <= max_samples
            else np.sort(rng.choice(n, size=max_samples, replace=False))
        )
        X_bg = mtm.X[sel]
        explainer = shap.TreeExplainer(est._model)
        mean_abs = _mean_abs_shap(explainer.shap_values(X_bg), p)
        total = float(mean_abs.sum()) or 1.0
        order = np.argsort(mean_abs)[::-1]
        importances = [
            {
                "feature": feat_cols[i],
                "mean_abs_shap": round(float(mean_abs[i]), 6),
                "share": round(float(mean_abs[i] / total), 4),
            }
            for i in order
        ]
        top_share = importances[0]["share"] if importances else 0.0
        png_path = _summary_plot(explainer, X_bg, feat_cols, out_png) if out_png else None
        return {
            "available": True,
            "n_background": int(X_bg.shape[0]),
            "top_features": [d["feature"] for d in importances[:10]],
            "importances": importances,
            "top_feature_share": round(float(top_share), 4),
            "dominance_flag": bool(top_share > _DOMINANCE_THRESHOLD),
            "summary_plot": png_path,
        }
    except Exception as exc:  # noqa: BLE001 - interpretation is optional, never fail the run
        logger.warning(f"SHAP interpretation unavailable ({exc}); continuing without it")
        return {"available": False, "reason": str(exc)}


def _summary_plot(
    explainer: Any, X_bg: np.ndarray, feat_cols: list[str], out_png: Path
) -> str | None:
    """Write a SHAP summary (bar) plot to ``out_png`` (headless Agg backend), best-effort."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import shap

        out_png.parent.mkdir(parents=True, exist_ok=True)
        shap.summary_plot(
            explainer.shap_values(X_bg),
            X_bg,
            feature_names=feat_cols,
            show=False,
            max_display=15,
            plot_type="bar",
        )
        plt.tight_layout()
        plt.savefig(out_png, dpi=110, bbox_inches="tight")
        plt.close("all")
        return str(out_png)
    except Exception as exc:  # noqa: BLE001 - plotting is best-effort
        logger.warning(f"SHAP summary plot failed ({exc}); continuing without it")
        return None
