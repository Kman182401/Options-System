"""Model-health gatherer: reads saved runs, fully-keyed whether or not one exists."""

from __future__ import annotations

import json

from options_system.observability.model_health import gather_model_health


def _write_run(runs_dir, symbol="SYN", verdict="no significant edge"):
    summary = {
        "symbol": symbol,
        "verdict": verdict,
        "verdict_checks": {"pbo_below_max": False},
        "model_version": "v1",
        "n_samples": 1000,
        "n_features": 45,
        "effective_n_total": 250.0,
        "n_trials": 8,
        "selected_overrides": {"max_depth": 3},
        "pooled_kfold": {"directional_accuracy": 0.51, "excess_sharpe": -0.02},
        "pbo": {"pbo": 0.83},
        "cpcv": {"n_paths": 5, "excess_sharpe": {"mean": -0.02}},
        "shap": {"top_features": ["rv_390", "vwap_dist"]},
        "mlflow_run_id": "abc123",
    }
    (runs_dir / f"{symbol}.json").write_text(json.dumps(summary))


def test_gather_reads_saved_run(tmp_path):
    _write_run(tmp_path, "SYN")
    rows = gather_model_health(["SYN", "MISSING"], runs_dir=tmp_path)
    by = {r["symbol"]: r for r in rows}

    syn = by["SYN"]
    assert syn["has_run"] is True
    assert syn["verdict"] == "no significant edge"
    assert syn["pbo"] == 0.83
    assert syn["shap_top_features"] == ["rv_390", "vwap_dist"]
    assert syn["mlflow_run_id"] == "abc123"

    missing = by["MISSING"]
    assert missing["has_run"] is False
    assert missing["verdict"] is None
    assert missing["pbo"] is None
