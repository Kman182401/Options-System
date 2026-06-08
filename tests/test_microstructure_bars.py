"""Dollar-bar construction: thresholds, OHLCV, determinism, roll/session boundaries."""

from __future__ import annotations

from _micro_helpers import TEST_INST, comoving_stream, ev, ts
from options_system.microstructure.bars import assemble_features, build_dollar_bars
from options_system.microstructure.config import Instrument, MicrostructureConfig

CFG = MicrostructureConfig.load()
HI = Instrument(
    symbol="T",
    continuous_symbol="T.v.0",
    exec_symbol="t",
    multiplier=1.0,
    tick_size=0.25,
    dollar_threshold=1e9,
)


def test_bar_closes_at_dollar_threshold():
    # 6 trades of 100*20 = 2000 each; threshold 5000 -> close after the 3rd, 6th.
    events = [ev(ts(i * 0.001), trade=(100.0, 20.0, 1)) for i in range(6)]
    bars = build_dollar_bars(events, instrument=TEST_INST, session=CFG.session)
    assert len(bars) == 2
    assert all(b["bar_complete"] for b in bars)
    assert [b["n_trades"] for b in bars] == [3, 3]
    assert all(b["dollar_volume"] >= TEST_INST.dollar_threshold for b in bars)


def test_ohlcv_vwap_signed_volume():
    trades = [(100.0, 1.0, 1), (102.0, 2.0, -1), (99.0, 3.0, 1), (101.0, 4.0, -1)]
    events = [ev(ts(i * 0.001), trade=t) for i, t in enumerate(trades)]
    df = assemble_features(
        build_dollar_bars(events, instrument=HI, session=CFG.session), symbol="T", cfg=CFG
    )
    assert df.height == 1
    r = df.row(0, named=True)
    assert (r["open"], r["high"], r["low"], r["close"]) == (100.0, 102.0, 99.0, 101.0)
    assert r["volume"] == 10.0
    assert abs(r["vwap"] - 100.5) < 1e-9
    assert r["signed_vol"] == -2.0
    assert abs(r["trade_imbalance"] - (-0.2)) < 1e-9
    assert r["n_trades"] == 4
    assert r["bar_complete"] is False  # threshold not reached -> trailing partial bar


def test_duration_seconds():
    # Two sub-threshold trades (3000 each); the 2nd at t=2s crosses 5000 -> one bar.
    events = [ev(ts(0.0), trade=(100.0, 30.0, 1)), ev(ts(2.0), trade=(100.0, 30.0, 1))]
    bars = build_dollar_bars(events, instrument=TEST_INST, session=CFG.session)
    assert len(bars) == 1
    df = assemble_features(bars, symbol="T", cfg=CFG)
    assert abs(df.row(0, named=True)["duration_s"] - 2.0) < 1e-6


def test_determinism():
    s = comoving_stream(20)
    a = assemble_features(
        build_dollar_bars(s, instrument=TEST_INST, session=CFG.session), symbol="T", cfg=CFG
    )
    b = assemble_features(
        build_dollar_bars(s, instrument=TEST_INST, session=CFG.session), symbol="T", cfg=CFG
    )
    assert a.equals(b)


def test_roll_boundary_splits_bars_no_span():
    # 2 trades on contract 1 (4000 < 5000, no threshold close), then contract 2 appears.
    events = [
        ev(ts(0.000), iid=1, trade=(100.0, 20.0, 1)),
        ev(ts(0.001), iid=1, trade=(100.0, 20.0, 1)),
        ev(ts(0.002), iid=2, trade=(100.0, 20.0, 1)),
    ]
    df = assemble_features(
        build_dollar_bars(events, instrument=TEST_INST, session=CFG.session), symbol="T", cfg=CFG
    )
    assert df.height == 2
    assert df["contract_id"].to_list() == ["id1", "id2"]
    assert df["con_id"].to_list() == [1, 2]
    assert df["n_trades"].to_list() == [2, 1]
    # the contract-1 bar closed at the roll seam (boundary), contract-2 bar is partial
    assert df["bar_complete"].to_list() == [True, False]


def test_session_boundary_splits_bars():
    events = [
        ev(ts(0.0, day=0), iid=1, trade=(100.0, 20.0, 1)),
        ev(ts(0.001, day=0), iid=1, trade=(100.0, 20.0, 1)),
        ev(ts(0.0, day=1), iid=1, trade=(100.0, 20.0, 1)),
    ]
    df = assemble_features(
        build_dollar_bars(events, instrument=TEST_INST, session=CFG.session), symbol="T", cfg=CFG
    )
    assert df.height == 2
    dates = df["ts_event"].dt.date().to_list()
    assert dates[0] != dates[1]


def test_trade_less_stream_yields_no_bars():
    events = [ev(ts(i * 0.001), bid0=100.0 + i * 0.25) for i in range(5)]  # book moves, no trades
    bars = build_dollar_bars(events, instrument=TEST_INST, session=CFG.session)
    assert bars == []


def test_non_rth_events_are_dropped():
    # 03:00 UTC = 23:00 ET previous day -> outside RTH -> filtered out entirely.
    from _micro_helpers import BASE_NS

    overnight = BASE_NS - 11 * 3600 * 1_000_000_000  # 03:00 UTC same calendar day
    events = [ev(overnight, trade=(100.0, 60.0, 1)), ev(overnight + 1000, trade=(100.0, 60.0, 1))]
    assert build_dollar_bars(events, instrument=TEST_INST, session=CFG.session) == []
