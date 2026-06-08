"""Short-horizon (micro) triple-barrier label tests — barrier logic, σ causality,
determinism, price-scale invariance.

The barrier-resolution unit tests drive ``_label_block`` directly with hand-built
arrays: ``sigma`` (and the event-sampling ``r``) are inputs to that function, so we
can place a single event and a controlled price path and assert the exact
first-touch outcome, the wall-clock vertical barrier, and the session-close hard
cap. Session-boundary + leakage teeth live in test_micro_labeling_leakage.py.

No network / no Databento / no credits — every fixture is synthetic.
"""

from __future__ import annotations

import math
from datetime import UTC, date, datetime, timedelta

import numpy as np
import polars as pl

from options_system.microstructure.config import MicrostructureConfig
from options_system.microstructure.label_config import MicroLabelConfig
from options_system.microstructure.labels import (
    _attach_block_sigma,
    _label_block,
    generate_micro_labels,
)

_S = 0.01  # constant σ used in the unit tests (1% over the 30-min horizon)
_LOG100 = math.log(100.0)
_START = datetime(2026, 5, 18, 14, 0, 0, tzinfo=UTC)  # 10:00 ET (EDT), inside RTH


def _cfg(**over) -> MicroLabelConfig:
    """Load defaults and shallow-merge nested overrides (one level), AFML-test style."""
    d = MicroLabelConfig.load().to_dict()
    for k, v in over.items():
        d[k] = {**d[k], **v} if isinstance(v, dict) else v
    return MicroLabelConfig.model_validate(d)


def _ts(n: int, *, dur: float = 10.0, start: datetime = _START) -> np.ndarray:
    """``n`` bar CLOSE timestamps spaced ``dur`` seconds apart (naive-UTC us)."""
    base = np.datetime64(start.replace(tzinfo=None), "us")
    return base + np.arange(1, n + 1) * np.timedelta64(int(round(dur * 1e6)), "us")


def _one_event(n: int) -> np.ndarray:
    """A returns array that makes the symmetric CUSUM fire exactly one event at bar 0."""
    r = np.zeros(n, dtype=np.float64)
    r[0] = 2.0 * _S  # exceeds threshold h = σ·cusum_mult (cusum_mult defaults to 1.0)
    return r


def _close_far() -> np.datetime64:
    """A session close well beyond the 30-min horizon (so guard-1 never excludes)."""
    return np.datetime64((_START + timedelta(seconds=10_000)).replace(tzinfo=None), "us")


# --- config ---------------------------------------------------------------- #


def test_config_loads_with_apriori_defaults():
    cfg = MicroLabelConfig.load()
    assert cfg.micro_label_version == "ml1"
    assert cfg.barriers.pt_mult == 1.5
    assert cfg.barriers.sl_mult == 1.5
    assert cfg.barriers.vertical_minutes == 30.0
    assert cfg.events.method == "cusum"


# --- barrier resolution (drive _label_block with controlled arrays) -------- #


def test_upper_barrier_first_touch():
    cfg = _cfg()
    n = 6
    ts = _ts(n)
    # cumulative log-return rises +0.006 per bar: crosses +1.5σ (=0.015) first at bar 3.
    logmid = _LOG100 + 0.006 * np.arange(n, dtype=np.float64)
    sigma = np.full(n, _S)
    rows = _label_block(
        ts,
        logmid,
        sigma,
        _one_event(n),
        contract_id="ESM26",
        session_date=date(2026, 5, 18),
        session_close=_close_far(),
        cfg=cfg,
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["label"] == 1
    assert row["barrier_touched"] == "upper"
    assert row["n_bars"] == 3  # touched at bar 3
    assert row["resolved_at_close"] is False
    assert math.isclose(row["ret_t1"], 0.018, rel_tol=1e-9)


def test_lower_barrier_first_touch():
    cfg = _cfg()
    n = 6
    ts = _ts(n)
    logmid = _LOG100 - 0.006 * np.arange(n, dtype=np.float64)
    sigma = np.full(n, _S)
    rows = _label_block(
        ts,
        logmid,
        sigma,
        _one_event(n),
        contract_id="ESM26",
        session_date=date(2026, 5, 18),
        session_close=_close_far(),
        cfg=cfg,
    )
    assert rows[0]["label"] == -1
    assert rows[0]["barrier_touched"] == "lower"
    assert rows[0]["n_bars"] == 3


def test_upper_precedence_when_both_would_touch():
    # Up at bar 2, down at bar 4 -> up wins (earlier first touch), mirroring daily.
    cfg = _cfg()
    n = 6
    ts = _ts(n)
    logmid = _LOG100 + np.array([0.0, 0.01, 0.02, 0.0, -0.02, -0.04])
    sigma = np.full(n, _S)
    rows = _label_block(
        ts,
        logmid,
        sigma,
        _one_event(n),
        contract_id="ESM26",
        session_date=date(2026, 5, 18),
        session_close=_close_far(),
        cfg=cfg,
    )
    assert rows[0]["label"] == 1 and rows[0]["barrier_touched"] == "upper"


def test_vertical_barrier_is_wall_clock_30min():
    # Flat path, bars span > 30 min -> resolves at the first bar past t0+30min.
    cfg = _cfg()
    n = 240  # 240 bars * 10s = 2400s > 1800s
    ts = _ts(n, dur=10.0)
    logmid = np.full(n, _LOG100)  # no move -> no horizontal touch
    sigma = np.full(n, _S)
    rows = _label_block(
        ts,
        logmid,
        sigma,
        _one_event(n),
        contract_id="ESM26",
        session_date=date(2026, 5, 18),
        session_close=_close_far(),
        cfg=cfg,
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["label"] == 0
    assert row["barrier_touched"] == "vertical"
    assert row["resolved_at_close"] is False
    # first bar with close >= t0 + 1800s : bar 0 closes at +10s, so bar index 180.
    assert row["n_bars"] == 180


def test_close_cap_when_no_bar_reaches_30min():
    # Bars span < 30 min but the 30-min mark is still <= session close: hard-cap at
    # the last in-session bar (barrier="close", resolved_at_close=True).
    cfg = _cfg()
    n = 60  # 60 bars * 10s = 600s < 1800s
    ts = _ts(n, dur=10.0)
    logmid = np.full(n, _LOG100)
    sigma = np.full(n, _S)
    rows = _label_block(
        ts,
        logmid,
        sigma,
        _one_event(n),
        contract_id="ESM26",
        session_date=date(2026, 5, 18),
        session_close=_close_far(),
        cfg=cfg,  # close at +10000s >= t0+1800s
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["barrier_touched"] == "close"
    assert row["resolved_at_close"] is True
    assert row["n_bars"] == n - 1  # resolved at the last bar


def test_no_read_past_t1():
    # Upper touched at bar 3. Perturbing any bar AFTER t1 must not change the label.
    cfg = _cfg()
    n = 240
    ts = _ts(n)
    sigma = np.full(n, _S)
    base = _LOG100 + np.concatenate(([0.0], np.full(n - 1, 0.006))).cumsum()
    base_rows = _label_block(
        ts,
        base,
        sigma,
        _one_event(n),
        contract_id="ESM26",
        session_date=date(2026, 5, 18),
        session_close=_close_far(),
        cfg=cfg,
    )
    t1_pos = base_rows[0]["n_bars"]  # bar index of t1 (p=0)
    perturbed = base.copy()
    perturbed[t1_pos + 1 :] += 5.0  # wild change strictly after t1
    pert_rows = _label_block(
        ts,
        perturbed,
        sigma,
        _one_event(n),
        contract_id="ESM26",
        session_date=date(2026, 5, 18),
        session_close=_close_far(),
        cfg=cfg,
    )
    assert pert_rows[0]["label"] == base_rows[0]["label"]
    assert pert_rows[0]["barrier_touched"] == base_rows[0]["barrier_touched"]
    assert pert_rows[0]["n_bars"] == base_rows[0]["n_bars"]
    assert math.isclose(pert_rows[0]["ret_t1"], base_rows[0]["ret_t1"], rel_tol=1e-12)


# --- σ causality ----------------------------------------------------------- #


def test_sigma_uses_only_past_bars():
    # σ at bar j must be unchanged by anything after bar j (causal EWMA, adjust=False).
    cfg = _cfg()
    n = 200
    rng = np.random.default_rng(7)
    mids = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.0005, n)))
    df = pl.DataFrame(
        {
            "mid_close": mids,
            "duration_s": np.full(n, 10.0),
        }
    )
    full = _attach_block_sigma(df, cfg)["_sigma"].to_numpy()
    j = 120
    df_pert = df.with_columns(
        pl.when(pl.int_range(pl.len()) > j)
        .then(pl.col("mid_close") * 3.0)
        .otherwise(pl.col("mid_close"))
        .alias("mid_close")
    )
    pert = _attach_block_sigma(df_pert, cfg)["_sigma"].to_numpy()
    head_full, head_pert = full[: j + 1], pert[: j + 1]
    finite = np.isfinite(head_full)
    assert np.allclose(head_full[finite], head_pert[finite], rtol=1e-12, atol=0.0)


# --- determinism + price-scale invariance ---------------------------------- #


def _rw_session(n: int, *, seed: int, scale: float = 1.0, start: datetime = _START) -> pl.DataFrame:
    """A single-session random-walk dollar-bar frame (deterministic by seed)."""
    rng = np.random.default_rng(seed)
    mids = 100.0 * scale * np.exp(np.cumsum(rng.normal(0.0, 0.0008, n)))
    base = start.replace(tzinfo=None)
    ts = [base + timedelta(seconds=10.0 * (i + 1)) for i in range(n)]
    return pl.DataFrame(
        {
            "ts_event": pl.Series(ts).dt.replace_time_zone("UTC"),
            "mid_close": mids,
            "duration_s": np.full(n, 10.0),
            "contract_id": ["ESM26"] * n,
            "bar_complete": [True] * n,
        }
    )


def test_deterministic_rerun():
    cfg = MicroLabelConfig.load()
    session = MicrostructureConfig.load().session
    bars = _rw_session(600, seed=11)
    a = generate_micro_labels(bars, cfg, session)
    b = generate_micro_labels(bars, cfg, session)
    assert a.equals(b)
    assert a.height > 0


def test_labels_invariant_to_global_price_scale():
    cfg = MicroLabelConfig.load()
    session = MicrostructureConfig.load().session
    a = generate_micro_labels(_rw_session(600, seed=11, scale=1.0), cfg, session)
    b = generate_micro_labels(_rw_session(600, seed=11, scale=7.3), cfg, session)
    assert a.height == b.height > 0
    assert a["label"].to_list() == b["label"].to_list()
    assert a["barrier_touched"].to_list() == b["barrier_touched"].to_list()
    assert a["n_bars"].to_list() == b["n_bars"].to_list()
    assert a["t1"].to_list() == b["t1"].to_list()
    assert np.allclose(a["ret_t1"].to_numpy(), b["ret_t1"].to_numpy(), rtol=1e-9, atol=1e-12)
    assert np.allclose(a["sigma"].to_numpy(), b["sigma"].to_numpy(), rtol=1e-9, atol=1e-12)
