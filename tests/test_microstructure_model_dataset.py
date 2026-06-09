"""Micro training-matrix assembly: as-of feature attach, leak-column exclusion,
inf rejection / NaN retention, and weight preservation.

Pure tests on the assembly helpers — no lake, no Databento, no network.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl

from options_system.microstructure.bars import feature_names
from options_system.microstructure.config import MicrostructureConfig
from options_system.microstructure_model.dataset import (
    _attach_features,
    _finalize,
    _matrix_from_frame,
)

FEATS = feature_names(MicrostructureConfig.load())
_T0 = datetime(2026, 2, 2, 14, 30, tzinfo=UTC)


def _bars_frame(n: int) -> pl.DataFrame:
    """n bars at 1-min spacing; each feature column holds the bar index (distinct)."""
    ts = [_T0 + timedelta(minutes=i) for i in range(n)]
    data: dict[str, object] = {"ts_event": ts}
    for f in FEATS:
        data[f] = [float(i) for i in range(n)]
    return pl.DataFrame(data).with_columns(pl.col("ts_event").dt.replace_time_zone("UTC"))


def _labels_frame(t0s: list[datetime]) -> pl.DataFrame:
    n = len(t0s)
    return pl.DataFrame(
        {
            "t0": t0s,
            "t1": [t + timedelta(minutes=30) for t in t0s],
            "label": [1, -1, 0][:n] + [0] * max(0, n - 3),
            "ret_t1": [0.01 * (i + 1) for i in range(n)],
            "sample_weight": [1.0 + 0.1 * i for i in range(n)],
            "uniqueness_weight": [0.6 + 0.01 * i for i in range(n)],
        }
    ).with_columns(
        pl.col("t0").dt.replace_time_zone("UTC"), pl.col("t1").dt.replace_time_zone("UTC")
    )


def test_attach_features_is_backward_asof_at_t0():
    bars = _bars_frame(5)
    # label A exactly on bar 2; label B 30s after bar 3 (backward -> bar 3).
    labels = _labels_frame([_T0 + timedelta(minutes=2), _T0 + timedelta(minutes=3, seconds=30)])
    joined = _attach_features(labels, bars, FEATS)
    # exact match -> bar index 2; backward match -> bar index 3
    assert joined["ofi_top"].to_list() == [2.0, 3.0]
    # every m1 feature attached
    for f in FEATS:
        assert f in joined.columns


def test_leak_columns_never_become_features():
    # ret_t1 / t1 / label are label columns, never in the feature set.
    for leak in ("ret_t1", "t1", "label", "barrier_touched", "sigma"):
        assert leak not in FEATS


def test_finalize_rejects_inf_keeps_nan_drops_null_label():
    n = 4
    base: dict[str, object] = {
        "t0": [_T0 + timedelta(minutes=i) for i in range(n)],
        "t1": [_T0 + timedelta(minutes=30 + i) for i in range(n)],
        "label": [1, -1, 0, 1],
        "ret_t1": [0.01, -0.02, 0.0, 0.03],
        "sample_weight": [1.0, 1.1, 1.2, 1.3],
        "uniqueness_weight": [0.6, 0.61, 0.62, 0.63],
    }
    for f in FEATS:
        base[f] = [0.0, 0.0, 0.0, 0.0]
    frame = pl.DataFrame(base).with_columns(
        pl.col("t0").dt.replace_time_zone("UTC"), pl.col("t1").dt.replace_time_zone("UTC")
    )
    # row 1: an inf feature -> dropped. row 2: a NaN feature -> KEPT. row 3: null label -> dropped.
    frame = frame.with_columns(
        pl.when(pl.int_range(pl.len()) == 1)
        .then(float("inf"))
        .when(pl.int_range(pl.len()) == 2)
        .then(float("nan"))
        .otherwise(pl.col("ofi_top"))
        .alias("ofi_top"),
        pl.when(pl.int_range(pl.len()) == 3).then(None).otherwise(pl.col("label")).alias("label"),
    )
    m, drops = _finalize(frame, FEATS)
    labels_kept = sorted(m["ret_t1"].to_list())
    assert drops["dropped_inf_feature"] == 1
    assert drops["dropped_null_label"] == 1
    # rows 0 (clean) and 2 (NaN feature kept) survive; rows 1 (inf) and 3 (null) gone
    assert labels_kept == [0.0, 0.01]
    assert m.height == 2


def test_matrix_preserves_weights_and_target():
    bars = _bars_frame(6)
    labels = _labels_frame([_T0 + timedelta(minutes=i) for i in range(3)])
    joined = _attach_features(labels, bars, FEATS)
    m, _ = _finalize(joined, FEATS)
    keep = [*FEATS, "label", "ret_t1", "sample_weight", "uniqueness_weight", "t0", "t1"]
    mtm = _matrix_from_frame(m.select(keep), "ES", FEATS, "m1", "ml1", "mm1")
    assert mtm.X.shape == (3, len(FEATS))
    assert set(mtm.y.tolist()).issubset({-1, 0, 1})
    assert np.allclose(mtm.sample_weight, [1.0, 1.1, 1.2])
    assert np.allclose(mtm.uniqueness_weight, [0.6, 0.61, 0.62])
    assert mtm.feature_cols == FEATS
    assert mtm.micro_model_version == "mm1"
    # effective N is the sum of uniqueness weights
    assert abs(mtm.effective_n - (0.6 + 0.61 + 0.62)) < 1e-9
