"""Label builder: idempotent writer, correct schema, leak-free features@t0 join (Task 5)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

import numpy as np
import polars as pl

from options_system.labeling.build import (
    _COLUMNS,
    _attach_weights,
    _write_symbol,
    labels_with_features,
    read_labels,
)
from options_system.labeling.config import LabelConfig
from options_system.labeling.triple_barrier import generate_labels

_WIDE_START = datetime(2000, 1, 1, tzinfo=UTC)
_WIDE_END = datetime(2100, 1, 1, tzinfo=UTC)


def _cfg(**over) -> LabelConfig:
    d = LabelConfig.load().to_dict()
    for k, v in over.items():
        d[k] = {**d[k], **v} if isinstance(v, dict) else v
    return LabelConfig.model_validate(d)


def _continuous(n: int = 1500, seed: int = 5) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.002, n)))
    t0 = datetime(2024, 1, 2, tzinfo=UTC)
    return pl.DataFrame(
        {
            "ts_event": [t0 + timedelta(minutes=i) for i in range(n)],
            "close": close.tolist(),
            "contract_id": ["MESH24"] * n,
            "session": ["RTH"] * n,
            "adj_factor": [1.0] * n,
        }
    )


def _labels_frame(cfg: LabelConfig) -> pl.DataFrame:
    cont = _continuous()
    labels = generate_labels(cont, cfg)
    labels = _attach_weights(labels, cont, cfg)
    return labels.with_columns(
        pl.lit("TEST").alias("symbol"),
        pl.lit(datetime.now(UTC)).cast(pl.Datetime("us", "UTC")).alias("ts_ingest"),
    ).select(_COLUMNS)


def test_writer_idempotent_and_schema(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("OPTIONS_DATA_DIR", str(tmp_path))
    cfg = _cfg(
        volatility={"ewm_span": 50, "min_samples": 50, "barrier_horizon_bars": 50},
        barriers={
            "pt_mult": 1.5,
            "sl_mult": 1.5,
            "max_hold_bars": 200,
            "vertical_label_sign": False,
        },
    )
    frame = _labels_frame(cfg)
    assert frame.height > 0
    # schema is exactly the persisted columns; version stamped; weights finite
    assert list(frame.columns) == list(_COLUMNS)
    assert set(frame["label_version"].to_list()) == {cfg.label_version}
    assert frame["weight"].is_finite().all()
    assert np.isclose(cast(float, frame["weight"].mean()), 1.0)

    n1 = _write_symbol(frame, "TEST")
    assert n1 == frame.height
    n2 = _write_symbol(frame, "TEST")  # re-run: idempotent on t0
    assert n2 == 0

    back = read_labels("TEST", _WIDE_START, _WIDE_END)
    assert back.height == frame.height
    assert (back["t1"] >= back["t0"]).all()


def test_features_at_t0_join_is_leak_free(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("OPTIONS_DATA_DIR", str(tmp_path))
    base = datetime(2024, 1, 2, 9, 0, tzinfo=UTC)
    ingest = datetime(2024, 1, 2, 23, 0, tzinfo=UTC)

    # two feature rows at 09:00 (ret_1=10) and 09:05 (ret_1=20)
    feats = pl.DataFrame(
        {
            "ts_event": [base, base + timedelta(minutes=5)],
            "ts_ingest": [ingest, ingest],
            "symbol": ["TEST", "TEST"],
            "ret_1": [10.0, 20.0],
            "feature_version": ["v1", "v1"],
            "degraded": [False, False],
        }
    )
    fdir = tmp_path / "features" / "symbol=TEST" / "date=2024-01-02"
    fdir.mkdir(parents=True)
    feats.write_parquet(fdir / "part-0.parquet")

    # labels at t0=09:04 (must see 09:00, NOT 09:05) and t0=09:06 (sees 09:05)
    t0s = [base + timedelta(minutes=4), base + timedelta(minutes=6)]
    labels = pl.DataFrame(
        {
            "t0": t0s,
            "t1": [t + timedelta(minutes=30) for t in t0s],
            "symbol": ["TEST", "TEST"],
            "ret": [0.001, -0.001],
            "label": pl.Series([1, -1], dtype=pl.Int8),
            "barrier": ["up", "dn"],
            "sigma": [0.01, 0.01],
            "n_bars": pl.Series([30, 30], dtype=pl.Int32),
            "contract_id": ["MESH24", "MESH24"],
            "roll_crossed": [False, False],
            "session": ["RTH", "RTH"],
            "degraded": [False, False],
            "avg_uniqueness": [1.0, 1.0],
            "weight": [1.0, 1.0],
            "side": pl.Series([None, None], dtype=pl.Int8),
            "meta_label": pl.Series([None, None], dtype=pl.Int8),
            "label_version": ["v1", "v1"],
            "ts_ingest": [ingest, ingest],
        }
    ).select(_COLUMNS)
    _write_symbol(labels, "TEST")

    joined = labels_with_features("TEST", _WIDE_START, _WIDE_END).sort("t0")
    assert joined.height == 2
    # leak-free: each t0 sees only the latest feature with ts_event <= t0
    assert joined["ret_1"].to_list() == [10.0, 20.0]


def test_read_labels_absent_symbol_empty(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("OPTIONS_DATA_DIR", str(tmp_path))
    assert read_labels("NOPE", _WIDE_START, _WIDE_END).is_empty()
