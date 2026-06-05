"""Volatility estimator + event sampler: causal σ, exact CUSUM, sane counts (Task 2)."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl

from options_system.labeling.config import LabelConfig
from options_system.labeling.events import (
    compute_sigma,
    cusum_events,
    grid_events,
    sample_events,
)


def _bars(closes: list[float]) -> pl.DataFrame:
    t0 = datetime(2024, 1, 2, tzinfo=UTC)
    return pl.DataFrame(
        {
            "ts_event": [t0 + timedelta(minutes=i) for i in range(len(closes))],
            "close": closes,
        }
    )


def _cfg(**over) -> LabelConfig:
    d = LabelConfig.load().to_dict()
    for k, v in over.items():
        d[k] = {**d[k], **v} if isinstance(v, dict) else v
    return LabelConfig.model_validate(d)


# --- CUSUM exactness -------------------------------------------------------- #


def test_cusum_fires_exactly_on_threshold_crossing():
    # constant +0.3 returns, threshold 1.0 -> cumsum hits 1.2 at i=3, resets;
    # 1.2 again at i=7, ... deterministic and hand-checkable.
    rets = np.full(12, 0.3)
    thresh = np.full(12, 1.0)
    idx = cusum_events(rets, thresh)
    assert idx.tolist() == [3, 7, 11]


def test_cusum_symmetric_negative_side():
    rets = np.array([-0.4, -0.4, -0.4, 0.0, 0.0])
    thresh = np.full(5, 1.0)
    # s_neg: -0.4, -0.8, -1.2 <= -1 -> event at i=2, reset
    assert cusum_events(rets, thresh).tolist() == [2]


def test_cusum_resets_after_each_event():
    rets = np.array([1.0, 1.0, 1.0, 1.0])
    thresh = np.full(4, 1.0)
    # each bar alone crosses -> event every bar
    assert cusum_events(rets, thresh).tolist() == [0, 1, 2, 3]


def test_cusum_warmup_nan_resets_accumulator():
    rets = np.array([0.6, math.nan, 0.6, 0.6])
    thresh = np.full(4, 1.0)
    # i0: s_pos=0.6 (<1). i1: nan -> reset to 0. i2: 0.6. i3: 1.2 -> event at 3.
    assert cusum_events(rets, thresh).tolist() == [3]


def test_grid_events_regular():
    assert grid_events(10, 3, 2).tolist() == [2, 5, 8]


# --- σ causality + scaling -------------------------------------------------- #


def test_sigma_is_causal_truncation_invariant():
    rng = np.random.default_rng(0)
    closes = (100 * np.exp(np.cumsum(rng.normal(0, 0.001, 600)))).tolist()
    cfg = _cfg(volatility={"ewm_span": 50, "min_samples": 50, "barrier_horizon_bars": 100})
    full = compute_sigma(_bars(closes), cfg)
    cut = 400
    trunc = compute_sigma(_bars(closes[:cut]), cfg)
    # σ at every bar < cut is identical whether or not future bars exist
    f = full["sigma"].to_numpy()[:cut]
    t = trunc["sigma"].to_numpy()
    both = np.isfinite(f) & np.isfinite(t)
    assert both.sum() > 0
    assert np.allclose(f[both], t[both], rtol=1e-12, atol=1e-15)


def test_sigma_warmup_is_null_then_defined():
    closes = (100 + np.arange(300) * 0.01).tolist()
    cfg = _cfg(volatility={"ewm_span": 50, "min_samples": 50, "barrier_horizon_bars": 100})
    out = compute_sigma(_bars(closes), cfg)
    s = out["sigma"].to_numpy()
    assert not np.isfinite(s[10])  # warmup
    assert np.isfinite(s[-1])  # defined later


def test_sigma_scales_with_horizon():
    rng = np.random.default_rng(1)
    closes = (100 * np.exp(np.cumsum(rng.normal(0, 0.001, 400)))).tolist()
    base = compute_sigma(
        _bars(closes),
        _cfg(volatility={"ewm_span": 50, "min_samples": 50, "barrier_horizon_bars": 100}),
    )
    quad = compute_sigma(
        _bars(closes),
        _cfg(volatility={"ewm_span": 50, "min_samples": 50, "barrier_horizon_bars": 400}),
    )
    b = base["sigma"].to_numpy()[-1]
    q = quad["sigma"].to_numpy()[-1]
    # 4x horizon -> sqrt(4)=2x sigma
    assert math.isclose(q, 2.0 * b, rel_tol=1e-12)


# --- event count is sane (far fewer than bars) ------------------------------ #


def test_event_count_far_fewer_than_bars():
    rng = np.random.default_rng(7)
    closes = (100 * np.exp(np.cumsum(rng.normal(0, 0.0008, 5000)))).tolist()
    cfg = _cfg(events={"method": "cusum", "cusum_mult": 1.0, "grid_step_bars": 60})
    out = compute_sigma(_bars(closes), cfg)
    idx = sample_events(out, cfg)
    assert 0 < idx.size < out.height // 10  # meaningfully sparse


def test_sample_events_grid_starts_after_warmup():
    closes = (100 + np.arange(500) * 0.01).tolist()
    cfg = _cfg(
        volatility={"ewm_span": 50, "min_samples": 50, "barrier_horizon_bars": 100},
        events={"method": "grid", "cusum_mult": 1.0, "grid_step_bars": 100},
    )
    out = compute_sigma(_bars(closes), cfg)
    idx = sample_events(out, cfg)
    assert idx.size > 0
    assert idx[0] >= 49  # no events during the σ warmup
