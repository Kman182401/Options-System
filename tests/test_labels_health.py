"""Labels-health view: class balance + counts over a written set (Task 6)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from options_system.data.store import DuckStore
from options_system.labeling.build import _COLUMNS, _attach_weights, _write_symbol
from options_system.labeling.config import LabelConfig
from options_system.labeling.triple_barrier import generate_labels
from options_system.observability.labels_health import gather_label_health


def _cfg(**over) -> LabelConfig:
    d = LabelConfig.load().to_dict()
    for k, v in over.items():
        d[k] = {**d[k], **v} if isinstance(v, dict) else v
    return LabelConfig.model_validate(d)


def _write_synthetic(symbol: str, seed: int) -> int:
    cfg = _cfg(
        volatility={"ewm_span": 50, "min_samples": 50, "barrier_horizon_bars": 50},
        barriers={
            "pt_mult": 1.5,
            "sl_mult": 1.5,
            "max_hold_bars": 200,
            "vertical_label_sign": False,
        },
    )
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.002, 1500)))
    t0 = datetime(2024, 1, 2, tzinfo=UTC)
    cont = pl.DataFrame(
        {
            "ts_event": [t0 + timedelta(minutes=i) for i in range(1500)],
            "close": close.tolist(),
            "contract_id": ["MESH24"] * 1500,
            "session": ["RTH"] * 1500,
        }
    )
    labels = _attach_weights(generate_labels(cont, cfg), cont, cfg).with_columns(
        pl.lit(symbol).alias("symbol"),
        pl.lit(datetime.now(UTC)).cast(pl.Datetime("us", "UTC")).alias("ts_ingest"),
    )
    return _write_symbol(labels.select(_COLUMNS), symbol)


def test_gather_label_health(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("OPTIONS_DATA_DIR", str(tmp_path))
    n = _write_synthetic("TEST", seed=5)
    assert n > 0

    store = DuckStore()
    try:
        health = gather_label_health(
            store, ["TEST", "EMPTY"], as_of=datetime(2030, 1, 1, tzinfo=UTC)
        )
    finally:
        store.close()

    test = next(h for h in health if h["symbol"] == "TEST")
    assert test["rows"] == n
    assert test["label_version"] == ["v1"]
    # class balance proportions sum to ~1 and cover the resolved classes
    assert abs(sum(test["class_balance"].values()) - 1.0) < 1e-6
    assert abs(sum(test["barrier_dist"].values()) - 1.0) < 1e-6
    assert 0.0 <= test["avg_uniqueness"] <= 1.0
    assert test["pct_roll_crossed"] == 0.0  # no rolls passed to synthetic build

    empty = next(h for h in health if h["symbol"] == "EMPTY")
    assert empty["rows"] == 0 and empty["class_balance"] == {}
