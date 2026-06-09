"""TA engine correctness: spot-checks vs independent pandas references, determinism,
warmup/NaN, and spread exclusion (feature_version = v2)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import polars as pl
import pytest

from options_system.ta.compute import compute_ta, ta_feature_names
from options_system.ta.config import TaConfig

CFG = TaConfig.load()


def _synth(n: int = 300, *, contract: str = "MESM6", seed: int = 0) -> pl.DataFrame:
    """Deterministic single-contract RTH bar frame, all within one ET session date.

    Start 13:30 UTC = 09:30 ET, 1-minute bars. ``n`` defaults to 300 so the
    triple-EWM (TRIX) has a long converged tail for reference comparison.
    """
    rng = np.random.default_rng(seed)
    close = 5000.0 + rng.normal(0, 1.0, n).cumsum()
    open_ = np.empty(n)
    open_[0] = close[0]
    open_[1:] = close[:-1]
    wig = np.abs(rng.normal(0, 0.5, n)) + 0.25
    high = np.maximum(open_, close) + wig
    low = np.minimum(open_, close) - wig
    volume = (rng.integers(50, 500, n)).astype(float)
    t0 = datetime(2026, 6, 1, 13, 30, tzinfo=UTC)
    ts = [t0 + timedelta(minutes=i) for i in range(n)]
    return pl.DataFrame(
        {
            "ts_event": ts,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "session": ["RTH"] * n,
            "contract_id": [contract] * n,
            "adj_factor": [1.0] * n,
        }
    ).with_columns(pl.col("ts_event").dt.cast_time_unit("us"))


def _hlcv(df: pl.DataFrame):
    return (df[k].to_pandas() for k in ("high", "low", "close", "volume"))


# --- spot-checks vs independent (pandas / hand) references ------------------ #


def test_stoch_matches_reference():
    df = _synth()
    feats = compute_ta(df, CFG)
    h, low_, c, _ = _hlcv(df)
    hh, ll = h.rolling(14).max(), low_.rolling(14).min()
    k_ref = 100.0 * (c - ll) / (hh - ll)
    d_ref = k_ref.rolling(3).mean()
    np.testing.assert_allclose(
        feats["ta_stoch_k_14"].to_numpy()[60:], k_ref.to_numpy()[60:], rtol=1e-6, atol=1e-9
    )
    np.testing.assert_allclose(
        feats["ta_stoch_d_3"].to_numpy()[60:], d_ref.to_numpy()[60:], rtol=1e-6, atol=1e-9
    )


def test_cci_matches_reference():
    df = _synth()
    feats = compute_ta(df, CFG)
    h, low_, c, _ = _hlcv(df)
    tp = (h + low_ + c) / 3.0
    sma = tp.rolling(20).mean()
    mad = (tp - sma).abs().rolling(20).mean()
    ref = (tp - sma) / (0.015 * mad)
    np.testing.assert_allclose(
        feats["ta_cci_20"].to_numpy()[60:], ref.to_numpy()[60:], rtol=1e-6, atol=1e-9
    )


def test_mfi_matches_reference():
    df = _synth()
    feats = compute_ta(df, CFG)
    h, low_, c, v = _hlcv(df)
    tp = (h + low_ + c) / 3.0
    rmf = tp * v
    d = tp.diff()
    pos = rmf.where(d > 0, 0.0).rolling(14).sum()
    neg = rmf.where(d < 0, 0.0).rolling(14).sum()
    ref = 100.0 * pos / (pos + neg)
    np.testing.assert_allclose(
        feats["ta_mfi_14"].to_numpy()[60:], ref.to_numpy()[60:], rtol=1e-6, atol=1e-9
    )


def test_vortex_matches_reference():
    df = _synth()
    feats = compute_ta(df, CFG)
    h, low_, c, _ = _hlcv(df)
    pc = c.shift(1)
    tr = pd.concat([h - low_, (h - pc).abs(), (low_ - pc).abs()], axis=1).max(axis=1)
    trs = tr.rolling(14).sum()
    vip_ref = (h - low_.shift(1)).abs().rolling(14).sum() / trs
    vim_ref = (low_ - h.shift(1)).abs().rolling(14).sum() / trs
    np.testing.assert_allclose(
        feats["ta_vi_plus_14"].to_numpy()[60:], vip_ref.to_numpy()[60:], rtol=1e-6, atol=1e-9
    )
    np.testing.assert_allclose(
        feats["ta_vi_minus_14"].to_numpy()[60:], vim_ref.to_numpy()[60:], rtol=1e-6, atol=1e-9
    )


def test_trix_matches_reference():
    # triple-nested EWM: compare only the deep converged tail (seed/null effects washed out).
    df = _synth(n=300)
    feats = compute_ta(df, CFG)
    _, _, c, _ = _hlcv(df)
    lc = np.log(c)
    e1 = lc.ewm(span=15, adjust=False, min_periods=15).mean()
    e2 = e1.ewm(span=15, adjust=False, min_periods=15).mean()
    e3 = e2.ewm(span=15, adjust=False, min_periods=15).mean()
    ref = e3.diff()
    np.testing.assert_allclose(
        feats["ta_trix_15"].to_numpy()[200:], ref.to_numpy()[200:], rtol=1e-6, atol=1e-9
    )


# --- determinism ----------------------------------------------------------- #


def test_determinism_same_input_same_output():
    a = compute_ta(_synth(), CFG)
    b = compute_ta(_synth(), CFG)
    assert a.equals(b)


# --- spread exclusion ------------------------------------------------------ #


def test_spread_contract_is_rejected():
    df = _synth(contract="MESU0-MESZ0")
    with pytest.raises(ValueError, match="spread"):
        compute_ta(df, CFG)


# --- warmup / degraded ----------------------------------------------------- #


def test_warmup_rows_are_degraded_and_null():
    feats = compute_ta(_synth(300), CFG)
    mw = CFG.max_window()
    assert feats.head(mw)["degraded"].all(), "every warmup row must be degraded"
    # an early row has nulls (windows unfilled), not fabricated values
    assert feats["ta_cci_20"][:30].null_count() == 30
    # after the longest window, every feature is fully populated and rows aren't degraded
    tail = feats.slice(mw + 10, 5000)
    null_cols = {n: tail[n].null_count() for n in ta_feature_names(CFG) if tail[n].null_count() > 0}
    assert not null_cols, f"unexpected nulls after warmup: {null_cols}"
    assert not tail["degraded"].all()
