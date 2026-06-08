"""MLflow tracking: a local file store under data/, one run id back, no cloud."""

from __future__ import annotations

from options_system.models.tracking import log_run, tracking_uri


def _summary():
    return {
        "symbol": "SYN",
        "verdict": "no significant edge",
        "model_version": "v1",
        "feature_version": "v1",
        "label_version": "v1",
        "validation_version": "v1",
        "timeout_handling": "sign_return",
        "n_samples": 1000,
        "n_features": 45,
        "n_trials": 8,
        "selection_metric": "directional_accuracy",
        "selected_overrides": {"max_depth": 3, "reg_lambda": 20.0},
        "effective_n_total": 250.0,
        "pooled_kfold": {"directional_accuracy": 0.51, "excess_sharpe": -0.02, "excess_dsr": 0.01},
        "pbo": {"pbo": 0.83},
        "cpcv": {"excess_sharpe": {"mean": -0.02}, "directional_accuracy_mean": 0.51},
    }


def test_log_run_creates_local_store_and_returns_run_id(tmp_path, monkeypatch):
    monkeypatch.setenv("OPTIONS_DATA_DIR", str(tmp_path))
    assert tracking_uri().startswith("file://")
    run_id = log_run(_summary(), None)
    assert isinstance(run_id, str) and run_id
    assert (tmp_path / "mlruns").exists()
