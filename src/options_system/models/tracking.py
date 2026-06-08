"""Local MLflow tracking — a file store on disk, never the cloud.

One MLflow run per (symbol, model_version) evaluation. We log: every version stamp
(model / feature / label / validation), the trial count and selected config, the
pooled-OOS and CPCV metric distributions, the overfitting gate (PBO / PSR / DSR),
the effective sample size, the SHAP summary, and the fitted model artifact. The run
is tagged with the **verdict** so the model-health view can surface it at a glance.

The tracking URI is a ``file://`` store under ``data/mlruns`` (``data/`` is
gitignored), so nothing leaves the machine and there is no server to run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from config.settings import Settings

from ..common.logging import get_logger

logger = get_logger(__name__)

_EXPERIMENT = "signal-model"


def tracking_uri() -> str:
    """``file://`` URI of the local MLflow store under ``data/mlruns`` (created on demand)."""
    d = Settings().data_dir / "mlruns"
    d.mkdir(parents=True, exist_ok=True)
    return d.as_uri()


def _flatten_metrics(summary: dict[str, Any]) -> dict[str, float]:
    """Pull the numeric gate metrics out of an evaluation summary for MLflow."""
    out: dict[str, float] = {"effective_n_total": float(summary["effective_n_total"])}
    pooled = summary.get("pooled_kfold", {})
    for k in (
        "directional_accuracy",
        "strategy_sharpe",
        "strategy_psr",
        "excess_sharpe",
        "excess_psr",
        "excess_dsr",
        "mean_excess_return",
        "long_benchmark_sharpe",
    ):
        v = pooled.get(k)
        if v is not None:
            out[f"pooled_{k}"] = float(v)
    if summary.get("pbo"):
        out["pbo"] = float(summary["pbo"]["pbo"])
    cpcv = summary.get("cpcv", {})
    ex = cpcv.get("excess_sharpe", {})
    for k in ("mean", "median", "min", "max", "std"):
        if ex.get(k) is not None:
            out[f"cpcv_excess_sharpe_{k}"] = float(ex[k])
    if cpcv.get("directional_accuracy_mean") is not None:
        out["cpcv_directional_accuracy_mean"] = float(cpcv["directional_accuracy_mean"])
    return out


def log_run(
    summary: dict[str, Any],
    shap_summary: dict[str, Any] | None,
    *,
    estimator: Any = None,
    shap_png: Path | None = None,
    experiment: str = _EXPERIMENT,
) -> str | None:
    """Log one evaluation to the local MLflow file store; return the run id (or ``None``)."""
    try:
        # MLflow 3 puts the local file store in "maintenance mode" and refuses it
        # unless this opt-out is set. We deliberately keep a local file store (no
        # cloud, no server, no DB) — set the flag before MLflow initialises.
        import os

        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
        import mlflow
    except Exception as exc:  # noqa: BLE001 - tracking is optional, never fail the run
        logger.warning(f"mlflow unavailable ({exc}); skipping tracking")
        return None

    mlflow.set_tracking_uri(tracking_uri())
    mlflow.set_experiment(experiment)
    run_name = f"{summary['symbol']}-{summary['model_version']}"
    with mlflow.start_run(run_name=run_name) as run:
        params: dict[str, Any] = {
            "symbol": summary["symbol"],
            "model_version": summary["model_version"],
            "feature_version": summary["feature_version"],
            "label_version": summary["label_version"],
            "validation_version": summary["validation_version"],
            "timeout_handling": summary["timeout_handling"],
            "n_samples": summary["n_samples"],
            "n_features": summary["n_features"],
            "n_trials": summary["n_trials"],
            "selection_metric": summary["selection_metric"],
        }
        for k, v in summary.get("selected_overrides", {}).items():
            params[f"sel_{k}"] = v
        mlflow.log_params(params)
        mlflow.log_metrics(_flatten_metrics(summary))
        mlflow.set_tag("verdict", summary["verdict"])

        mlflow.log_dict(summary, "evaluation.json")
        if shap_summary is not None:
            mlflow.log_dict(shap_summary, "shap.json")
        if shap_png is not None and Path(shap_png).exists():
            mlflow.log_artifact(str(shap_png))
        if estimator is not None and getattr(estimator, "_model", None) is not None:
            try:
                import mlflow.lightgbm

                mlflow.lightgbm.log_model(estimator._model, name="model")
            except Exception as exc:  # noqa: BLE001 - artifact logging is best-effort
                logger.warning(f"model artifact log failed ({exc}); continuing")
        return run.info.run_id
