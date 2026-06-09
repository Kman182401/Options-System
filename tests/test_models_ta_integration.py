"""Phase 10: opt-in TA integration into the training matrix + runner.

Covers default parity (TA off changes nothing), the hard error when TA is requested
but missing, the backward as-of + inf-sanitised attach, cache separation, the
evaluate-summary metadata, the CLI flags, and model-health TA awareness. Synthetic
frames / temp dirs where practical; real-lake checks skip when the lake is absent.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from glob import glob

import numpy as np
import polars as pl
import pytest

import options_system.models.run as run_mod
from options_system.data.store import DuckStore
from options_system.features.build import partition_glob as feature_glob
from options_system.models.config import ModelConfig
from options_system.models.dataset import (
    TrainingMatrix,
    _attach_ta,
    _cache_path,
    load_training_matrix,
)
from options_system.observability.model_health import gather_model_health
from options_system.ta.build import partition_glob as ta_glob
from options_system.validation.config import ValidationConfig


def _m(minutes: int) -> timedelta:
    return timedelta(minutes=minutes)


# --------------------------------------------------------------------------- #
# 1. Default parity — TA off is the current behavior, no ta_ columns
# --------------------------------------------------------------------------- #
def test_default_omits_ta_columns():
    """A default (synthetic) TrainingMatrix carries no TA state."""
    tm = TrainingMatrix(
        symbol="SYN",
        X=np.zeros((3, 2)),
        y=np.array([1, -1, 1]),
        y_dir=np.array([1, -1, 1]),
        t0=np.array([0, 1, 2]),
        t1=np.array([1, 2, 3]),
        ret=np.zeros(3),
        weight=np.ones(3),
        uniqueness=np.ones(3),
        feature_cols=["f0", "f1"],
        feature_version="v1",
        label_version="v1",
        timeout_handling="sign_return",
    )
    assert tm.ta_cols == []
    assert tm.ta_feature_version is None
    assert tm.with_ta is False


def test_with_ta_false_equals_default_content():
    """load_training_matrix(with_ta=False) is byte-identical to the default call."""
    if not glob(feature_glob("MES")):
        pytest.skip("no feature lake present")
    s, e = datetime(2021, 1, 1, tzinfo=UTC), datetime(2021, 4, 1, tzinfo=UTC)
    default = load_training_matrix("MES", start=s, end=e, use_cache=False)
    explicit = load_training_matrix("MES", start=s, end=e, use_cache=False, with_ta=False)
    assert default.feature_cols == explicit.feature_cols
    assert not any(c.startswith("ta_") for c in default.feature_cols)
    assert default.ta_cols == [] and default.with_ta is False
    assert np.array_equal(default.X, explicit.X)
    assert default.n == explicit.n


# --------------------------------------------------------------------------- #
# 2. TA requested but missing → clear ValueError (never a silent fallback)
# --------------------------------------------------------------------------- #
def test_with_ta_missing_lake_raises():
    with pytest.raises(ValueError, match=r"ta\.build"):
        load_training_matrix("NOPE_NO_SUCH_SYMBOL", with_ta=True, use_cache=False)


# --------------------------------------------------------------------------- #
# 3. Attach behavior — backward as-of, inf-sanitised, no rows dropped
# --------------------------------------------------------------------------- #
def test_attach_ta_is_backward_asof_and_sanitises_inf(tmp_path, monkeypatch):
    monkeypatch.setenv("OPTIONS_DATA_DIR", str(tmp_path))
    base = datetime(2020, 1, 2, 15, 0, tzinfo=UTC)
    feat_ts = [base, base + _m(5), base + _m(10)]
    feats = pl.DataFrame(
        {
            "ts_event": feat_ts,
            "ts_ingest": [base] * 3,
            "ta_x": [10.0, float("inf"), 30.0],  # middle row is degenerate +inf
        }
    ).with_columns(pl.col("ts_event").cast(pl.Datetime("us", "UTC")))
    part = tmp_path / "ta_features" / "symbol=TST" / "date=2020-01-02"
    part.mkdir(parents=True)
    feats.write_parquet(part / "part.parquet")

    frame = pl.DataFrame(
        {
            "t0": [base - _m(1), base + _m(3), base + _m(6), base + _m(10)],
            "label": [1, 1, 1, 1],
        }
    ).with_columns(pl.col("t0").cast(pl.Datetime("us", "UTC")))

    store = DuckStore()
    try:
        out = _attach_ta(store, frame, "TST", ["ta_x"]).sort("t0")
    finally:
        store.close()

    vals = out["ta_x"].to_list()
    assert vals[0] is None  # before any feature → no leak from the future
    assert vals[1] == 10.0  # latest <= t0 is the base bar
    assert vals[2] is None  # latest <= t0 is the +inf bar → sanitised to null
    assert vals[3] == 30.0  # exact match, inclusive
    assert out.height == frame.height  # additive: no rows dropped


def test_ta_columns_appended_without_dropping_rows():
    """TA adds COLUMNS after price+macro at each t0, never drops rows (count unchanged)."""
    if not glob(feature_glob("MES")) or not glob(ta_glob("MES")):
        pytest.skip("need feature lake + TA lake (run ta.build)")
    s, e = datetime(2021, 1, 1, tzinfo=UTC), datetime(2021, 6, 1, tzinfo=UTC)
    base = load_training_matrix("MES", start=s, end=e, use_cache=False, with_ta=False)
    aug = load_training_matrix("MES", start=s, end=e, use_cache=False, with_ta=True)

    assert aug.n == base.n  # TA adds columns, never drops rows
    assert aug.ta_cols and aug.ta_feature_version == "v2"
    assert all(c.startswith("ta_") for c in aug.ta_cols)
    assert aug.X.shape[1] == base.X.shape[1] + len(aug.ta_cols)
    # the price(+macro) block is byte-identical (TA is purely additive, appended last)
    assert np.array_equal(aug.X[:, : base.X.shape[1]], base.X)
    ta_block = aug.X[:, base.X.shape[1] :]
    assert np.isfinite(ta_block).any()  # real values present for this in-history window
    assert not np.isinf(ta_block).any()  # no infinity ever reaches X


# --------------------------------------------------------------------------- #
# 4. Cache separation — no-TA and TA matrices never collide
# --------------------------------------------------------------------------- #
def test_cache_path_separates_ta_from_nota():
    no_ta = _cache_path("MES", "v1", "v1", "sign_return", "macro-v1", "nota")
    ta = _cache_path("MES", "v1", "v1", "sign_return", "macro-v1", "ta-v2")
    assert no_ta != ta
    assert "nota" in no_ta.name
    assert "ta-v2" in ta.name  # TA cache key includes the ta_feature_version


# --------------------------------------------------------------------------- #
# 5. Evaluate summary carries the TA metadata
# --------------------------------------------------------------------------- #
def _fast_cfg() -> ModelConfig:
    c = ModelConfig.load()
    return c.model_copy(
        update={
            "lgbm": c.lgbm.model_copy(update={"n_estimators": 40}),
            "search": c.search.model_copy(update={"grid": {"reg_lambda": [5.0, 20.0]}}),
            "early_stopping": c.early_stopping.model_copy(update={"enabled": False}),
        }
    )


def _tm_with_ta(n: int = 420, seed: int = 7) -> TrainingMatrix:
    rng = np.random.default_rng(seed)
    start = datetime(2024, 1, 2, tzinfo=UTC)
    t0 = np.array(
        [np.datetime64((start + timedelta(minutes=i)).replace(tzinfo=None), "us") for i in range(n)]
    )
    X = rng.normal(size=(n, 6))
    ret = rng.normal(0.0, 0.01, n)
    y_dir = np.where(ret > 0, 1, -1)
    return TrainingMatrix(
        symbol="SYN",
        X=X,
        y=y_dir.copy(),
        y_dir=y_dir,
        t0=t0,
        t1=t0 + np.timedelta64(12, "m"),
        ret=ret,
        weight=np.ones(n),
        uniqueness=np.full(n, 0.25),
        feature_cols=["f0", "f1", "f2", "f3", "ta_a", "ta_b"],
        feature_version="v1",
        label_version="v1",
        timeout_handling="sign_return",
        ta_cols=["ta_a", "ta_b"],
        ta_feature_version="v2",
    )


def test_evaluate_summary_includes_ta_metadata(monkeypatch):
    monkeypatch.setattr(run_mod, "load_training_matrix", lambda symbol, **kw: _tm_with_ta())
    summary = run_mod.run_symbol(
        "SYN",
        _fast_cfg(),
        ValidationConfig.load(),
        with_ta=True,
        log_mlflow=False,
        interpret=False,
        save=False,
    )
    assert summary["with_ta"] is True
    assert summary["ta_feature_version"] == "v2"
    assert summary["n_ta_features"] == 2


# --------------------------------------------------------------------------- #
# 6. CLI — new flags parse + route correctly; existing flags unchanged
# --------------------------------------------------------------------------- #
def test_main_with_ta_routes_to_single_run(monkeypatch):
    seen: dict = {}

    def fake_run(symbol, mcfg, vcfg, **kw):
        seen.update(with_ta=kw.get("with_ta"), with_macro=kw.get("with_macro"))
        return {"_stub": True}

    monkeypatch.setattr(run_mod, "run_symbol", fake_run)
    monkeypatch.setattr(run_mod, "_print_verdict", lambda s: None)
    rc = run_mod.main(["--symbols", "SYN", "--with-ta", "--no-mlflow", "--no-interpret"])
    assert rc == 0
    assert seen["with_ta"] is True and seen["with_macro"] is True


def test_main_compare_ta_routes_to_ta_comparison(monkeypatch):
    seen: dict = {}

    def fake_cmp(symbol, mcfg, vcfg, **kw):
        seen["called"] = "ta"
        return {"_stub": True}

    monkeypatch.setattr(run_mod, "compare_ta_symbol", fake_cmp)
    monkeypatch.setattr(run_mod, "_print_ta_comparison", lambda c: None)
    rc = run_mod.main(["--symbols", "SYN", "--compare-ta", "--no-mlflow", "--no-interpret"])
    assert rc == 0 and seen["called"] == "ta"


def test_main_compare_still_runs_macro_comparison(monkeypatch):
    seen: dict = {}

    def fake_cmp(symbol, mcfg, vcfg, **kw):
        seen["called"] = "macro"
        return {"_stub": True}

    monkeypatch.setattr(run_mod, "compare_symbol", fake_cmp)
    monkeypatch.setattr(run_mod, "_print_comparison", lambda c: None)
    rc = run_mod.main(["--symbols", "SYN", "--compare", "--no-mlflow", "--no-interpret"])
    assert rc == 0 and seen["called"] == "macro"


def test_main_no_macro_remains_compatible(monkeypatch):
    seen: dict = {}

    def fake_run(symbol, mcfg, vcfg, **kw):
        seen.update(with_macro=kw.get("with_macro"), with_ta=kw.get("with_ta"))
        return {"_stub": True}

    monkeypatch.setattr(run_mod, "run_symbol", fake_run)
    monkeypatch.setattr(run_mod, "_print_verdict", lambda s: None)
    rc = run_mod.main(["--symbols", "SYN", "--no-macro", "--no-mlflow", "--no-interpret"])
    assert rc == 0
    assert seen["with_macro"] is False and seen["with_ta"] is False


# --------------------------------------------------------------------------- #
# 7. model-health is TA-aware and does not break on TA-less runs
# --------------------------------------------------------------------------- #
def test_model_health_exposes_ta_keys(tmp_path):
    info = gather_model_health(["ZZZ"], runs_dir=tmp_path)[0]
    assert info["with_ta"] is False
    assert info["ta_feature_version"] is None
    assert info["n_ta_features"] == 0
    assert info["has_ta_comparison"] is False
