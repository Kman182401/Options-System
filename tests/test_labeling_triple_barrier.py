"""Triple-barrier correctness: first-touch, t1 bounds, σ-scaling, rolls, back-adj (Task 3)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl

from options_system.labeling.config import LabelConfig
from options_system.labeling.triple_barrier import generate_labels, label_events


def _cfg(**over) -> LabelConfig:
    d = LabelConfig.load().to_dict()
    for k, v in over.items():
        d[k] = {**d[k], **v} if isinstance(v, dict) else v
    return LabelConfig.model_validate(d)


def _df(cr_from_p: list[float], *, sigma: float, contract: str = "MESH24") -> pl.DataFrame:
    """Bars whose cumulative log-return from bar 0 is ``cr_from_p``; constant σ column."""
    t0 = datetime(2024, 1, 2, tzinfo=UTC)
    closes = (100.0 * np.exp(np.asarray(cr_from_p))).tolist()
    n = len(closes)
    return pl.DataFrame(
        {
            "ts_event": [t0 + timedelta(minutes=i) for i in range(n)],
            "close": closes,
            "contract_id": [contract] * n,
            "session": ["RTH"] * n,
            "sigma": [sigma] * n,
        }
    )


# --- first-touch correctness ----------------------------------------------- #


def test_first_touch_upper():
    df = _df([0, 0.005, 0.010, 0.020, 0.020, 0.020, 0.020, 0.020, 0.020], sigma=0.01)
    cfg = _cfg(
        barriers={"pt_mult": 1.5, "sl_mult": 1.5, "max_hold_bars": 8, "vertical_label_sign": False}
    )
    out = label_events(df, np.array([0]), cfg)
    row = out.to_dicts()[0]
    assert row["label"] == 1 and row["barrier"] == "up"
    assert row["n_bars"] == 3
    assert row["t1"] == df["ts_event"][3]
    assert abs(row["ret"] - 0.020) < 1e-9


def test_first_touch_lower():
    df = _df([0, -0.004, -0.010, -0.018, -0.02, -0.02, -0.02, -0.02], sigma=0.01)
    cfg = _cfg(
        barriers={"pt_mult": 1.5, "sl_mult": 1.5, "max_hold_bars": 7, "vertical_label_sign": False}
    )
    out = label_events(df, np.array([0]), cfg)
    row = out.to_dicts()[0]
    assert row["label"] == -1 and row["barrier"] == "dn"
    assert row["n_bars"] == 3  # first bar with cr <= -0.015 is index 3 (-0.018)


def test_vertical_timeout_label_zero():
    df = _df([0, 0.002, 0.004, 0.003, 0.005, 0.004, 0.003, 0.002, 0.001], sigma=0.01)
    cfg = _cfg(
        barriers={"pt_mult": 1.5, "sl_mult": 1.5, "max_hold_bars": 8, "vertical_label_sign": False}
    )
    out = label_events(df, np.array([0]), cfg)
    row = out.to_dicts()[0]
    assert row["label"] == 0 and row["barrier"] == "time"
    assert row["n_bars"] == 8
    assert row["t1"] == df["ts_event"][8]


def test_vertical_label_sign_option():
    df = _df([0, 0.002, 0.004, 0.003, 0.005, 0.004, 0.003, 0.002, 0.006], sigma=0.01)
    cfg = _cfg(
        barriers={"pt_mult": 1.5, "sl_mult": 1.5, "max_hold_bars": 8, "vertical_label_sign": True}
    )
    row = label_events(df, np.array([0]), cfg).to_dicts()[0]
    assert row["barrier"] == "time" and row["label"] == 1  # sign(+0.006)


def test_upper_wins_when_hit_before_lower():
    # rises to upper at k=2, would hit lower later — first touch must win
    df = _df([0, 0.010, 0.020, -0.05, -0.05, -0.05], sigma=0.01)
    cfg = _cfg(
        barriers={"pt_mult": 1.5, "sl_mult": 1.5, "max_hold_bars": 5, "vertical_label_sign": False}
    )
    row = label_events(df, np.array([0]), cfg).to_dicts()[0]
    assert row["label"] == 1 and row["barrier"] == "up" and row["n_bars"] == 2


# --- t1 bounds invariant ---------------------------------------------------- #


def test_t1_bounds_hold_for_all_events():
    rng = np.random.default_rng(3)
    closes = (100 * np.exp(np.cumsum(rng.normal(0, 0.002, 1500)))).tolist()
    cfg = _cfg(
        volatility={"ewm_span": 50, "min_samples": 50, "barrier_horizon_bars": 100},
        barriers={
            "pt_mult": 1.5,
            "sl_mult": 1.5,
            "max_hold_bars": 120,
            "vertical_label_sign": False,
        },
    )
    t0 = datetime(2024, 1, 2, tzinfo=UTC)
    df = pl.DataFrame(
        {
            "ts_event": [t0 + timedelta(minutes=i) for i in range(len(closes))],
            "close": closes,
            "contract_id": ["MESH24"] * len(closes),
            "session": ["RTH"] * len(closes),
        }
    )
    out = generate_labels(df, cfg)
    assert out.height > 0
    assert (out["t1"] >= out["t0"]).all()
    assert (out["n_bars"] >= 1).all()
    assert (out["n_bars"] <= cfg.barriers.max_hold_bars).all()


# --- σ-scaling: wider σ => fewer price-barrier touches ---------------------- #


def test_barriers_scale_with_sigma():
    cr = [0, 0.004, 0.008, 0.012, 0.012, 0.012, 0.012, 0.012]
    base = _cfg(
        barriers={"pt_mult": 1.5, "sl_mult": 1.5, "max_hold_bars": 7, "vertical_label_sign": False}
    )
    tight = label_events(_df(cr, sigma=0.005), np.array([0]), base).to_dicts()[0]
    wide = label_events(_df(cr, sigma=0.02), np.array([0]), base).to_dicts()[0]
    # σ=0.005 -> up=0.0075, crossed (label +1). σ=0.02 -> up=0.03, never -> timeout 0.
    assert tight["label"] == 1 and tight["barrier"] == "up"
    assert wide["label"] == 0 and wide["barrier"] == "time"


# --- right-censoring: unresolved tail events are dropped -------------------- #


def test_right_censored_event_is_dropped():
    # flat path, event late, vertical window extends past data, no barrier -> drop
    df = _df([0.0] * 6, sigma=0.01)
    cfg = _cfg(
        barriers={"pt_mult": 1.5, "sl_mult": 1.5, "max_hold_bars": 50, "vertical_label_sign": False}
    )
    out = label_events(df, np.array([0]), cfg)
    assert out.is_empty()  # vbar=50 > n-1=5, no touch -> censored


def test_timeout_kept_when_window_fits():
    df = _df([0.0] * 12, sigma=0.01)
    cfg = _cfg(
        barriers={"pt_mult": 1.5, "sl_mult": 1.5, "max_hold_bars": 8, "vertical_label_sign": False}
    )
    out = label_events(df, np.array([0]), cfg)
    assert out.height == 1 and out["barrier"][0] == "time" and out["n_bars"][0] == 8


# --- roll crossing: flagged (adjust) and capped (close) --------------------- #


def test_roll_crossing_flagged_in_adjust_mode():
    df = _df([0.0] * 30, sigma=0.01)
    rolls = pl.DataFrame({"ts_event": [df["ts_event"][10]]})
    cfg = _cfg(
        barriers={
            "pt_mult": 1.5,
            "sl_mult": 1.5,
            "max_hold_bars": 20,
            "vertical_label_sign": False,
        },
        roll={"handling": "adjust"},
    )
    row = label_events(df, np.array([5]), cfg, rolls=rolls).to_dicts()[0]
    assert row["roll_crossed"] is True
    assert row["barrier"] == "time" and row["t1"] == df["ts_event"][25]  # walked through seam


def test_roll_crossing_caps_in_close_mode():
    df = _df([0.0] * 30, sigma=0.01)
    rolls = pl.DataFrame({"ts_event": [df["ts_event"][10]]})
    cfg = _cfg(
        barriers={
            "pt_mult": 1.5,
            "sl_mult": 1.5,
            "max_hold_bars": 20,
            "vertical_label_sign": False,
        },
        roll={"handling": "close"},
    )
    row = label_events(df, np.array([5]), cfg, rolls=rolls).to_dicts()[0]
    assert row["roll_crossed"] is True
    assert row["barrier"] == "roll" and row["t1"] == df["ts_event"][10] and row["n_bars"] == 5


# --- back-adjustment invariance (THE constraint) ---------------------------- #


def test_labels_invariant_to_global_price_scale():
    rng = np.random.default_rng(11)
    closes = 100 * np.exp(np.cumsum(rng.normal(0, 0.0015, 2000)))
    cfg = _cfg(
        volatility={"ewm_span": 50, "min_samples": 50, "barrier_horizon_bars": 100},
        barriers={
            "pt_mult": 1.5,
            "sl_mult": 1.5,
            "max_hold_bars": 200,
            "vertical_label_sign": False,
        },
        events={"method": "cusum", "cusum_mult": 1.0, "grid_step_bars": 60},
    )
    t0 = datetime(2024, 1, 2, tzinfo=UTC)

    def build(scale: float) -> pl.DataFrame:
        df = pl.DataFrame(
            {
                "ts_event": [t0 + timedelta(minutes=i) for i in range(len(closes))],
                "close": (closes * scale).tolist(),
                "contract_id": ["MESH24"] * len(closes),
                "session": ["RTH"] * len(closes),
            }
        )
        return generate_labels(df, cfg)

    a = build(1.0)
    b = build(7.3)  # a global back-adjustment-style rescale
    assert a.height == b.height > 0
    # discrete outcomes identical; continuous outcomes identical to float tol
    assert a["label"].to_list() == b["label"].to_list()
    assert a["barrier"].to_list() == b["barrier"].to_list()
    assert a["n_bars"].to_list() == b["n_bars"].to_list()
    assert a["t1"].to_list() == b["t1"].to_list()
    assert np.allclose(a["ret"].to_numpy(), b["ret"].to_numpy(), rtol=1e-9, atol=1e-12)
    assert np.allclose(a["sigma"].to_numpy(), b["sigma"].to_numpy(), rtol=1e-9, atol=1e-12)
