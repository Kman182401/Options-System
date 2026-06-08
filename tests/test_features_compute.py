"""Feature engine correctness: spot-checks vs independent references, determinism,
warmup/NaN, spread exclusion, and cross-asset causality (Tasks 2 & 5)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import cast

import numpy as np
import pandas as pd
import polars as pl
import pytest

from options_system.data.lake import Lake
from options_system.data.store import DuckStore
from options_system.features.compute import compute_features, feature_names
from options_system.features.config import FeatureConfig

CFG = FeatureConfig.load()


def _synth(n: int = 160, *, contract: str = "MESM6", seed: int = 0) -> pl.DataFrame:
    """Deterministic single-contract RTH bar frame, all within one ET session date.

    Start 13:30 UTC = 09:30 ET, 1-minute bars -> one session_date, adj_factor=1
    (continuous == raw) so references can be computed directly off close.
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


def _close(matched: pl.DataFrame) -> pd.Series:
    return matched["close"].to_pandas()


# --- spot-checks vs independent (pandas / hand) references ------------------ #


def test_rsi_matches_wilder_reference():
    df = _synth()
    feats = compute_features(df, CFG, symbol="MES", other=None)
    c = _close(df)
    # match the engine's seed: the first (undefined) diff becomes 0 gain/0 loss,
    # so the Wilder EWM seeds at index 0 exactly as polars does.
    delta = c.diff().fillna(0.0)
    ag = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    al = (-delta).clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rsi_ref = 100 - 100 / (1 + ag / al)
    got = feats["rsi_14"].to_numpy()
    ref = rsi_ref.to_numpy()
    # compare on the converged tail (seed effects washed out)
    np.testing.assert_allclose(got[60:], ref[60:], rtol=1e-6, atol=1e-6)


def test_atr_pct_matches_wilder_reference():
    df = _synth()
    feats = compute_features(df, CFG, symbol="MES", other=None)
    h, low_, c = df["high"].to_pandas(), df["low"].to_pandas(), df["close"].to_pandas()
    pc = c.shift(1)
    tr = pd.concat([h - low_, (h - pc).abs(), (low_ - pc).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    ref = (atr / c).to_numpy()
    got = feats["atr_pct_14"].to_numpy()
    np.testing.assert_allclose(got[60:], ref[60:], rtol=1e-6, atol=1e-9)


def test_vwap_dist_matches_manual_reference():
    df = _synth()
    feats = compute_features(df, CFG, symbol="MES", other=None)
    h, low_, c, v = (df[k].to_numpy() for k in ("high", "low", "close", "volume"))
    typical = (h + low_ + c) / 3.0
    vwap = np.cumsum(typical * v) / np.cumsum(v)  # single session -> straight cumsum
    ref = c / vwap - 1.0
    np.testing.assert_allclose(feats["vwap_dist"].to_numpy(), ref, rtol=1e-9, atol=1e-12)


# --- determinism ----------------------------------------------------------- #


def test_determinism_same_input_same_output():
    df = _synth()
    a = compute_features(df, CFG, symbol="MES", other=None)
    b = compute_features(df, CFG, symbol="MES", other=None)
    assert a.equals(b)


# --- spread exclusion ------------------------------------------------------ #


def test_spread_contract_is_rejected():
    df = _synth(contract="MESU0-MESZ0")
    with pytest.raises(ValueError, match="spread"):
        compute_features(df, CFG, symbol="MES", other=None)


# --- cross-asset causality (other's future never reaches a feature at t) ---- #


def test_cross_asset_uses_no_future_of_the_other_symbol():
    this = _synth(seed=1)
    other_full = _synth(seed=2)
    t = this["ts_event"][100]
    # truncate the OTHER symbol strictly after t, then blow up its tail
    other_trunc = other_full.filter(pl.col("ts_event") <= t)
    other_spiked = other_full.with_columns(
        pl.when(pl.col("ts_event") > t)
        .then(pl.col("close") * 5.0)
        .otherwise(pl.col("close"))
        .alias("close")
    )
    f_trunc = compute_features(this, CFG, symbol="MES", other=other_trunc)
    f_spiked = compute_features(this, CFG, symbol="MES", other=other_spiked)
    xa = [
        "xa_ret_spread",
        f"xa_ratio_z_{CFG.cross_asset.ratio_zscore_window}",
        f"xa_corr_{CFG.cross_asset.corr_window}",
    ]
    row_t = pl.col("ts_event") == t
    a = f_trunc.filter(row_t).select(xa).to_dicts()[0]
    b = f_spiked.filter(row_t).select(xa).to_dicts()[0]
    assert a == b, "cross-asset feature at t changed when only the other symbol's FUTURE changed"


# --- warmup / degraded (uses real lake data) ------------------------------- #


@pytest.fixture(scope="module")
def real_mes():
    if cast(pl.DataFrame, Lake().scan("bars_1m", "MES").collect()).is_empty():
        pytest.skip("lake not populated")
    store = DuckStore()
    df = store.get_bars(
        "MES", datetime(2000, 1, 1, tzinfo=UTC), datetime(2100, 1, 1, tzinfo=UTC), continuous=True
    )
    feats = compute_features(df, CFG, symbol="MES", other=None)
    store.close()
    return feats


def test_warmup_rows_are_degraded_and_null(real_mes):
    max_w = CFG.max_window()
    head = real_mes.head(max_w)
    assert head["degraded"].all(), "every warmup row must be degraded"
    # an early row has nulls (windows unfilled), not fabricated values
    assert real_mes["ret_60"][:60].null_count() == 60
    assert real_mes["rsi_14"][0] is None


def test_post_warmup_has_no_unexpected_nulls(real_mes):
    # after the longest window, the non-cross-asset features are fully populated
    tail = real_mes.slice(CFG.max_window() + 10, 5000)
    non_xa = [n for n in feature_names(CFG) if not n.startswith("xa_")]
    null_cols = {n: tail[n].null_count() for n in non_xa if tail[n].null_count() > 0}
    assert not null_cols, f"unexpected nulls after warmup: {null_cols}"
    assert not tail["degraded"].all()  # not everything past warmup is degraded
