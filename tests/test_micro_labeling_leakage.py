"""Leakage + session-boundary teeth for the short-horizon (micro) labels.

Two guarantees, both with teeth:

1. NO READ PAST t1 — a label is a pure function of bars ``[t0, t1]``. We prove
   (a) truncating the bars at t1 reproduces the label exactly (point-in-time), and
   (b) the planted leak: a value that peeks ONE bar past t1 flips under a post-t1
   perturbation while our label does not — so the invariance check is non-trivial.

2. NO CROSSING THE RTH SESSION CLOSE — labels are built per ``(contract, ET-date)``
   block, so a label can physically never read the next session. We prove
   (a) every retained label's t1 stays in its own session and before the close,
   (b) events in the final 30 min are excluded, and (c) perturbing the NEXT
   session leaves this session's labels byte-identical.

No network / no Databento / no credits — every fixture is synthetic.
"""

from __future__ import annotations

import math
from datetime import UTC, date, datetime, timedelta

import numpy as np
import polars as pl

from options_system.microstructure.config import MicrostructureConfig
from options_system.microstructure.label_config import MicroLabelConfig
from options_system.microstructure.labels import _label_block, generate_micro_labels

_S = 0.01
_LOG100 = math.log(100.0)
_RES_COLS = ["label", "barrier_touched", "ret_t1", "sigma", "n_bars", "t1"]


def _one_event(n: int) -> np.ndarray:
    r = np.zeros(n, dtype=np.float64)
    r[0] = 2.0 * _S
    return r


def _ts(n: int, *, dur: float = 10.0, start: datetime) -> np.ndarray:
    base = np.datetime64(start.replace(tzinfo=None), "us")
    return base + np.arange(1, n + 1) * np.timedelta64(int(round(dur * 1e6)), "us")


# --- leakage: no read past t1 ---------------------------------------------- #


def test_planted_leak_past_t1_is_detectable():
    """Teeth: our label ignores bars after t1; a peek-one-past value would not."""
    cfg = MicroLabelConfig.load()
    n = 240
    start = datetime(2026, 5, 18, 14, 0, 0, tzinfo=UTC)
    ts = _ts(n, start=start)
    sigma = np.full(n, _S)
    close = np.datetime64((start + timedelta(seconds=10_000)).replace(tzinfo=None), "us")
    logmid = _LOG100 + np.concatenate(([0.0], np.full(n - 1, 0.006))).cumsum()  # upper at bar 3

    base = _label_block(
        ts,
        logmid,
        sigma,
        _one_event(n),
        contract_id="ESM26",
        session_date=date(2026, 5, 18),
        session_close=close,
        cfg=cfg,
    )[0]
    pos = base["n_bars"]  # t1 bar index (p == 0)
    leak_before = float(logmid[pos + 1] - logmid[0])  # a value that reads ONE bar past t1

    perturbed = logmid.copy()
    perturbed[pos + 1 :] += 5.0  # change everything strictly after t1
    pert = _label_block(
        ts,
        perturbed,
        sigma,
        _one_event(n),
        contract_id="ESM26",
        session_date=date(2026, 5, 18),
        session_close=close,
        cfg=cfg,
    )[0]
    leak_after = float(perturbed[pos + 1] - perturbed[0])

    # our label is unchanged ...
    assert pert["label"] == base["label"]
    assert math.isclose(pert["ret_t1"], base["ret_t1"], rel_tol=1e-12)
    assert pert["n_bars"] == base["n_bars"]
    # ... but a construction that peeked one bar past t1 WOULD have changed (teeth).
    assert not math.isclose(leak_after, leak_before, rel_tol=1e-6)


def test_truncation_at_t1_reproduces_label():
    """Point-in-time: truncating the bars at a label's t1 reproduces its resolution."""
    cfg = MicroLabelConfig.load()
    session = MicrostructureConfig.load().session
    bars = _rw_session(700, seed=3)
    full = generate_micro_labels(bars, cfg, session)
    assert full.height > 0
    target = full.row(0, named=True)
    trunc = bars.filter(pl.col("ts_event") <= target["t1"])
    re = generate_micro_labels(trunc, cfg, session).filter(pl.col("t0") == target["t0"])
    assert re.height == 1
    got = re.row(0, named=True)
    for c in _RES_COLS:
        assert got[c] == target[c], f"{c}: {got[c]} != {target[c]}"


def test_truncating_inside_window_changes_outcome():
    """Teeth complement: cutting bars BEFORE t1 changes/drops the label (the PIT
    check is sensitive — it is not vacuously satisfied)."""
    cfg = MicroLabelConfig.load()
    session = MicrostructureConfig.load().session
    bars = _rw_session(700, seed=3)
    full = generate_micro_labels(bars, cfg, session)
    target = full.row(0, named=True)
    bar_ts = bars.sort("ts_event")["ts_event"].to_numpy()
    t1_pos = int(np.searchsorted(bar_ts, np.datetime64(target["t1"].replace(tzinfo=None), "us")))
    cut = bars.filter(
        pl.col("ts_event") < pl.lit(bar_ts[t1_pos - 1]).cast(pl.Datetime("us", "UTC"))
    )
    re = generate_micro_labels(cut, cfg, session).filter(pl.col("t0") == target["t0"])
    changed = re.height == 0 or re.row(0, named=True)["t1"] != target["t1"]
    assert changed, "cutting inside the window left the outcome unchanged -> no teeth"


# --- session boundary ------------------------------------------------------ #


def test_final_30min_event_is_excluded():
    """An event whose t0+30min would fall after the session close is dropped."""
    cfg = MicroLabelConfig.load()
    n = 60
    start = datetime(2026, 5, 18, 19, 40, 0, tzinfo=UTC)  # 15:40 ET -> < 30 min to close
    ts = _ts(n, start=start)
    sigma = np.full(n, _S)
    logmid = np.full(n, _LOG100)
    # session close = 16:00 ET = 20:00 UTC; t0+30min = ~16:10 ET > close -> excluded.
    close = np.datetime64(datetime(2026, 5, 18, 20, 0, 0).replace(tzinfo=None), "us")
    rows = _label_block(
        ts,
        logmid,
        sigma,
        _one_event(n),
        contract_id="ESM26",
        session_date=date(2026, 5, 18),
        session_close=close,
        cfg=cfg,
    )
    assert rows == []


def test_no_label_crosses_session_close():
    cfg = MicroLabelConfig.load()
    session = MicrostructureConfig.load().session
    bars = _two_sessions()
    labels = generate_micro_labels(bars, cfg, session)
    assert labels.height > 0
    et = session.tz
    chk = labels.with_columns(
        pl.col("t0").dt.convert_time_zone(et).dt.date().alias("d0"),
        pl.col("t1").dt.convert_time_zone(et).dt.date().alias("d1"),
        (
            pl.col("t1").dt.convert_time_zone(et).dt.hour().cast(pl.Int32) * 60
            + pl.col("t1").dt.convert_time_zone(et).dt.minute().cast(pl.Int32)
        ).alias("t1_min"),
    )
    # every label resolves in its own session, strictly before 16:00 ET.
    assert (chk["d0"] == chk["d1"]).all()
    assert (chk["t1_min"] < session.rth_close_min).all()


def test_never_reads_next_session():
    """Perturbing the NEXT session leaves this session's labels byte-identical."""
    cfg = MicroLabelConfig.load()
    session = MicrostructureConfig.load().session
    bars = _two_sessions()
    base = generate_micro_labels(bars, cfg, session)
    s1 = base.filter(
        pl.col("t0").dt.convert_time_zone(session.tz).dt.date() == pl.lit(date(2026, 5, 18))
    )
    assert s1.height > 0
    # wildly perturb only the second session (2026-05-19).
    perturbed = bars.with_columns(
        pl.when(
            pl.col("ts_event").dt.convert_time_zone(session.tz).dt.date()
            == pl.lit(date(2026, 5, 19))
        )
        .then(pl.col("mid_close") * 2.0)
        .otherwise(pl.col("mid_close"))
        .alias("mid_close")
    )
    re = generate_micro_labels(perturbed, cfg, session)
    re_s1 = re.filter(
        pl.col("t0").dt.convert_time_zone(session.tz).dt.date() == pl.lit(date(2026, 5, 18))
    )
    assert re_s1.select(_RES_COLS).equals(s1.select(_RES_COLS))


# --- synthetic builders ---------------------------------------------------- #


def _rw_session(n: int, *, seed: int, start: datetime | None = None) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    mids = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.0008, n)))
    base = (start or datetime(2026, 5, 18, 14, 0, 0, tzinfo=UTC)).replace(tzinfo=None)
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


def _two_sessions() -> pl.DataFrame:
    # Session 1 late in the day (15:00-15:58 ET) so the final-30-min exclusion bites;
    # session 2 the next morning. Both within RTH (13:30-20:00 UTC during EDT).
    s1 = _rw_session(350, seed=21, start=datetime(2026, 5, 18, 19, 0, 0, tzinfo=UTC))
    s2 = _rw_session(350, seed=22, start=datetime(2026, 5, 19, 13, 35, 0, tzinfo=UTC))
    return pl.concat([s1, s2])
