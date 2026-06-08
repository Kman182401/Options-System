"""Training-matrix assembly: directional target, leak-free as-of attach, exact match.

The directional-target and as-of-join correctness tests are synthetic and always run.
The exact-match-to-Phase-4 check needs the real lake and skips when it is absent.
"""

from __future__ import annotations

from datetime import UTC, datetime
from glob import glob

import numpy as np
import polars as pl
import pytest

from options_system.data.store import DuckStore
from options_system.models.dataset import _asof_attach, derive_direction


def test_derive_direction_sign_return_folds_timeouts():
    label = np.array([1, -1, 0, 0])
    ret = np.array([0.1, -0.1, 0.2, -0.3])
    y_dir, keep = derive_direction(label, ret, "sign_return")
    assert list(y_dir) == [1, -1, 1, -1]  # timeouts take the sign of their return
    assert keep.all()  # nothing dropped


def test_derive_direction_drop_excludes_timeouts():
    label = np.array([1, -1, 0])
    ret = np.array([0.1, -0.1, 0.2])
    y_dir, keep = derive_direction(label, ret, "drop")
    assert list(keep) == [True, True, False]
    assert list(y_dir[keep]) == [1, -1]


def test_asof_attach_is_leak_free(tmp_path, monkeypatch):
    """The attached feature is the latest with ts_event <= t0 — never the future."""
    monkeypatch.setenv("OPTIONS_DATA_DIR", str(tmp_path))
    # three feature rows at 10:00, 10:05, 10:10
    base = datetime(2020, 1, 2, 15, 0, tzinfo=UTC)  # 10:00 ET-ish, value irrelevant
    feat_ts = [base, base + _m(5), base + _m(10)]
    feats = pl.DataFrame(
        {
            "ts_event": feat_ts,
            "ts_ingest": [base] * 3,
            "f0": [10.0, 20.0, 30.0],
        }
    ).with_columns(pl.col("ts_event").cast(pl.Datetime("us", "UTC")))
    part = tmp_path / "features" / "symbol=TST" / "date=2020-01-02"
    part.mkdir(parents=True)
    feats.write_parquet(part / "part.parquet")

    # labels: before-all (→ null), between (→ 10:00 value), exact (→ that bar's value)
    labels = pl.DataFrame({"t0": [base - _m(1), base + _m(3), base + _m(10)]}).with_columns(
        pl.col("t0").cast(pl.Datetime("us", "UTC"))
    )

    store = DuckStore()
    try:
        joined = _asof_attach(store, labels, "TST", ["f0"]).sort("t0")
    finally:
        store.close()
    vals = joined["f0"].to_list()
    assert vals[0] is None  # t0 before any feature → no leak from the future
    assert vals[1] == 10.0  # latest <= t0 is the 10:00 bar
    assert vals[2] == 30.0  # exact match picks that bar (<=, inclusive)


def test_matrix_matches_phase4_load_matrix_on_bounded_window():
    from options_system.features.build import partition_glob

    if not glob(partition_glob("MES")):
        pytest.skip("no feature lake present")
    from options_system.models.dataset import load_training_matrix
    from options_system.validation.evaluate import load_matrix

    s, e = datetime(2019, 5, 1, tzinfo=UTC), datetime(2019, 9, 1, tzinfo=UTC)
    ref = load_matrix("MES", start=s, end=e)
    new = load_training_matrix("MES", start=s, end=e, use_cache=False)
    assert new.n == ref.n
    assert np.array_equal(new.X, ref.X)
    assert np.array_equal(new.y, ref.y)
    assert np.array_equal(new.t0, ref.t0)
    assert np.array_equal(new.t1, ref.t1)
    assert np.array_equal(new.ret, ref.ret)
    assert np.array_equal(new.weight, ref.weight)
    assert np.array_equal(new.uniqueness, ref.uniqueness)
    assert new.feature_cols == ref.feature_cols


def _m(minutes: int):
    from datetime import timedelta

    return timedelta(minutes=minutes)
