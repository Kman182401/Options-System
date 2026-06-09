"""Local MLflow tracking for the micro signal model — a file store on disk.

One MLflow run per (symbol, micro_model_version) evaluation, logged to the same
local ``file://`` store under ``data/mlruns`` (gitignored) as the daily model, but a
separate experiment (``micro-signal-model``). We log the version stamps, the trial
count + selected config, the gross/excess signal metrics, the overfitting gate
(PBO / DSR), the classification metrics, the effective sample size, the optional
SHAP summary and the fitted model artifact, and tag the run with the **verdict**.
Best-effort: if MLflow is unavailable the run continues untracked.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from config.settings import Settings

from ..common.logging import get_logger

logger = get_logger(__name__)

_EXPERIMENT = "micro-signal-model"


def tracking_uri() -> str:
    """``file://`` URI of the local MLflow store under ``data/mlruns`` (created on demand)."""
    d = Settings().data_dir / "mlruns"
    d.mkdir(parents=True, exist_ok=True)
    return d.as_uri()


def _flatten_metrics(summary: dict[str, Any]) -> dict[str, float]:
    """Pull the numeric gate metrics out of an evaluation summary for MLflow."""
    out: dict[str, float] = {
        "effective_n": float(summary["effective_n"]),
        "n_samples": float(summary["n_samples"]),
        "action_rate": float(summary["action_rate"]),
    }
    for k in ("weighted_accuracy", "balanced_accuracy", "macro_f1"):
        v = summary.get("classification", {}).get(k)
        if v is not None:
            out[k] = float(v)
    for k in (
        "gross_sharpe",
        "gross_psr",
        "gross_dsr",
        "mean_gross_return",
        "excess_over_long_sharpe",
        "long_benchmark_sharpe",
    ):
        v = summary.get("signal_return", {}).get(k)
        if v is not None:
            out[k] = float(v)
    if summary.get("pbo"):
        out["pbo"] = float(summary["pbo"]["pbo"])
    gs = summary.get("cpcv", {}).get("gross_sharpe", {})
    for k in ("mean", "median", "min", "max"):
        if gs.get(k) is not None:
            out[f"cpcv_gross_sharpe_{k}"] = float(gs[k])
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
        import os

        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
        import mlflow
    except Exception as exc:  # noqa: BLE001 - tracking is optional, never fail the run
        logger.warning(f"mlflow unavailable ({exc}); skipping tracking")
        return None

    mlflow.set_tracking_uri(tracking_uri())
    mlflow.set_experiment(experiment)
    run_name = f"{summary['symbol']}-{summary['micro_model_version']}"
    with mlflow.start_run(run_name=run_name) as run:
        params: dict[str, Any] = {
            "symbol": summary["symbol"],
            "micro_model_version": summary["micro_model_version"],
            "microstructure_feature_version": summary["microstructure_feature_version"],
            "micro_label_version": summary["micro_label_version"],
            "validation_version": summary["validation_version"],
            "target_mode": summary["target_mode"],
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
