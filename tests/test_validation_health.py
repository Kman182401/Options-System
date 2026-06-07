"""gather_validation_health reads saved runs and always returns fully-keyed dicts."""

from __future__ import annotations

import json

from options_system.observability.validation_health import gather_validation_health


def _write_run(runs_dir, symbol):
    summary = {
        "symbol": symbol,
        "validation_version": "v1",
        "n_samples": 1234,
        "n_features": 45,
        "effective_n_total": 290.5,
        "pbo": {"pbo": 0.47, "n_combinations": 252, "estimators": ["dummy", "logistic"]},
        "kfold": {
            "dummy": {
                "accuracy": 0.5,
                "sharpe": -0.1,
                "psr": 0.2,
                "dsr": 0.0,
                "effective_n": 290.5,
            },
            "logistic": {
                "accuracy": 0.51,
                "sharpe": 0.02,
                "psr": 0.55,
                "dsr": 0.1,
                "effective_n": 290.5,
            },
        },
        "cpcv": {
            "dummy": {
                "n_paths": 5,
                "sharpe_mean": -0.05,
                "sharpe_std": 0.1,
                "effective_n_mean": 58.0,
            },
            "logistic": {
                "n_paths": 5,
                "sharpe_mean": 0.0,
                "sharpe_std": 0.08,
                "effective_n_mean": 58.0,
            },
        },
    }
    (runs_dir / f"{symbol}.json").write_text(json.dumps(summary), encoding="utf-8")


def test_gather_returns_populated_dict_for_saved_run(tmp_path):
    _write_run(tmp_path, "MES")
    rows = gather_validation_health(["MES"], runs_dir=tmp_path)
    assert len(rows) == 1
    info = rows[0]
    assert info["has_run"] is True
    assert info["symbol"] == "MES"
    assert info["n_samples"] == 1234
    assert info["effective_n_total"] == 290.5
    assert info["pbo"] == 0.47
    assert set(info["estimators"]) == {"dummy", "logistic"}
    assert info["cpcv"]["logistic"]["n_paths"] == 5


def test_gather_returns_complete_default_dict_for_missing_run(tmp_path):
    rows = gather_validation_health(["EMPTY"], runs_dir=tmp_path)
    info = rows[0]
    # Every key present even with no run, so the view never KeyErrors.
    assert info["has_run"] is False
    for key in (
        "symbol",
        "validation_version",
        "n_samples",
        "effective_n_total",
        "pbo",
        "estimators",
        "kfold",
        "cpcv",
    ):
        assert key in info
    assert info["n_samples"] == 0
    assert info["pbo"] is None


def test_gather_mixed_symbols(tmp_path):
    _write_run(tmp_path, "MES")
    rows = gather_validation_health(["MES", "MNQ"], runs_dir=tmp_path)
    by_symbol = {r["symbol"]: r for r in rows}
    assert by_symbol["MES"]["has_run"] is True
    assert by_symbol["MNQ"]["has_run"] is False
