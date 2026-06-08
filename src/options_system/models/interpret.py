"""SHAP interpretability for the directional signal model.

Why it exists: the prime directive says a human must be able to *explain* what the
model does. SHAP attributes each prediction to its features, so we can see which
drivers matter and sanity-check that none of them smells like leakage (one feature
dominating, or an economically implausible driver topping the list, is a red flag).

The model is refit on the **full** matrix here purely for interpretation — these
attributions are an explanation, **not** a performance number (those come only from
the leak-safe CV in ``evaluate_model.py``). Refitting in-sample is the standard,
correct thing to do for a global SHAP view.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..common.logging import get_logger
from .config import ModelConfig
from .dataset import TrainingMatrix
from .lgbm import build_estimator

logger = get_logger(__name__)

# Above this share of total mean|SHAP| a single feature is "dominating" — worth a
# leakage second-look (the features are causal + leak-tested, so this should be low).
_DOMINANCE_THRESHOLD = 0.5


def _to_2d_shap(values: Any, n_features: int) -> np.ndarray:
    """Normalise SHAP output (list / 2-D / 3-D) to a 2-D ``(n_samples, n_features)`` array.

    For a binary LightGBM classifier different SHAP versions return either a list
    ``[class0, class1]``, a 3-D array ``(n, p, n_classes)``, or a single 2-D array
    (the ``+1``-class margin). We always reduce to the ``+1``-class attributions.
    """
    if isinstance(values, list):
        arr = np.asarray(values[-1])  # +1 class
    else:
        arr = np.asarray(values)
        if arr.ndim == 3:  # (n, p, n_classes)
            arr = arr[..., -1]
    if arr.ndim != 2 or arr.shape[1] != n_features:
        raise ValueError(f"unexpected SHAP shape {arr.shape}; expected (n, {n_features})")
    return arr


def explain(
    tm: TrainingMatrix,
    mcfg: ModelConfig,
    overrides: dict | None = None,
    *,
    max_samples: int = 2000,
    n_local: int = 3,
    out_png: Path | None = None,
    estimator: Any = None,
) -> dict[str, Any]:
    """Global + local SHAP attributions for the selected config on ``tm``.

    Returns a JSON-able dict: ranked global importances (mean ``|SHAP|``), the top
    feature's share (dominance/leakage smell check), and a few local explanations.
    Optionally writes a SHAP summary plot to ``out_png``. Pass a prefit ``estimator``
    (fit on the full matrix) to avoid refitting.
    """
    import shap

    feat_cols = tm.feature_cols
    p = len(feat_cols)
    est = estimator
    if est is None:
        est = build_estimator(mcfg, overrides or {})
        est.fit(tm.X, tm.y_dir, sample_weight=tm.weight)  # in-sample fit: explanation only

    rng = np.random.default_rng(mcfg.seed)
    n = tm.n
    sel = (
        np.arange(n)
        if n <= max_samples
        else np.sort(rng.choice(n, size=max_samples, replace=False))
    )
    X_bg = tm.X[sel]

    explainer = shap.TreeExplainer(est._model)
    shap_vals = _to_2d_shap(explainer.shap_values(X_bg), p)

    mean_abs = np.abs(shap_vals).mean(axis=0)
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

    # A few local explanations: per-sample top-3 signed contributions.
    local: list[dict[str, Any]] = []
    for r in range(min(n_local, X_bg.shape[0])):
        row = shap_vals[r]
        top_idx = np.argsort(np.abs(row))[::-1][:3]
        local.append(
            {
                "sample": int(sel[r]),
                "predicted": int(est.predict(X_bg[r : r + 1])[0]),
                "top_contributions": [
                    {"feature": feat_cols[i], "shap": round(float(row[i]), 6)} for i in top_idx
                ],
            }
        )

    png_path: str | None = None
    if out_png is not None:
        png_path = _summary_plot(shap_vals, X_bg, feat_cols, out_png)

    return {
        "n_background": int(X_bg.shape[0]),
        "top_features": [d["feature"] for d in importances[:10]],
        "importances": importances,
        "top_feature_share": round(float(top_share), 4),
        "dominance_flag": bool(top_share > _DOMINANCE_THRESHOLD),
        "local_explanations": local,
        "summary_plot": png_path,
    }


def _summary_plot(
    shap_vals: np.ndarray, X_bg: np.ndarray, feat_cols: list[str], out_png: Path
) -> str | None:
    """Write a SHAP beeswarm summary plot to ``out_png`` (headless Agg backend)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import shap

        out_png.parent.mkdir(parents=True, exist_ok=True)
        shap.summary_plot(shap_vals, X_bg, feature_names=feat_cols, show=False, max_display=15)
        plt.tight_layout()
        plt.savefig(out_png, dpi=110, bbox_inches="tight")
        plt.close("all")
        return str(out_png)
    except Exception as exc:  # noqa: BLE001 - plotting is best-effort, never fail the run
        logger.warning(f"SHAP summary plot failed ({exc}); continuing without it")
        return None
