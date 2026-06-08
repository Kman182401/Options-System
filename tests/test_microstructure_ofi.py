"""Pure order-flow math: known-answer OFI/micro-price/imbalance + the OFI↔Δmid fact."""

from __future__ import annotations

from _micro_helpers import TEST_INST, comoving_stream
from options_system.microstructure.bars import assemble_features, build_dollar_bars
from options_system.microstructure.config import MicrostructureConfig
from options_system.microstructure.ofi import (
    book_imbalance,
    level_ofi,
    micro_price,
    mid_price,
)


def _ofi(pb_p, qb_p, pa_p, qa_p, pb, qb, pa, qa):
    return level_ofi(pb_p, qb_p, pa_p, qa_p, pb, qb, pa, qa)


def test_level_ofi_bid_size_increase_same_price():
    # bid same price, size 5 -> 8 (+3 demand); ask unchanged.
    assert _ofi(100, 5, 101, 5, 100, 8, 101, 5) == 3.0


def test_level_ofi_bid_price_up_is_positive():
    # bid moves up: full new size counts as inflow.
    assert _ofi(100, 5, 101, 5, 100.25, 4, 101, 5) == 4.0


def test_level_ofi_bid_price_down_is_negative():
    assert _ofi(100, 5, 101, 5, 99.75, 9, 101, 5) == -5.0


def test_level_ofi_ask_up_is_bullish():
    # ask withdrawn upward -> supply removed -> OFI positive (+q_ask_prev).
    assert _ofi(100, 5, 101, 5, 100, 5, 101.25, 4) == 5.0


def test_level_ofi_ask_down_is_bearish():
    assert _ofi(100, 5, 101, 5, 100, 5, 100.75, 9) == -9.0


def test_level_ofi_absent_level_contributes_zero():
    nan = float("nan")
    assert _ofi(nan, 0, nan, 0, nan, 0, nan, 0) == 0.0
    # one side absent -> only the present side contributes
    assert _ofi(100, 5, nan, 0, 100, 8, nan, 0) == 3.0


def test_micro_price_leans_to_heavier_bid():
    # heavy bid (qb=10) vs empty ask queue -> micro near the ask price.
    assert micro_price(100.0, 101.0, 10.0, 0.0) == 101.0
    assert micro_price(100.0, 101.0, 0.0, 0.0) == 100.5  # fallback to mid
    assert mid_price(100.0, 101.0) == 100.5


def test_book_imbalance_bounds():
    assert book_imbalance((10.0,), (0.0,)) == 1.0
    assert book_imbalance((0.0,), (10.0,)) == -1.0
    assert book_imbalance((5.0,), (5.0,)) == 0.0
    assert book_imbalance((0.0,), (0.0,)) == 0.0


def test_ofi_and_mid_change_are_positively_correlated():
    """Contemporaneous OFI↔Δmid must be strongly positive (microstructure fact)."""
    cfg = MicrostructureConfig.load()
    bars = build_dollar_bars(comoving_stream(40), instrument=TEST_INST, session=cfg.session)
    df = assemble_features(bars, symbol="T", cfg=cfg)
    assert df.height >= 30
    import polars as pl

    corr = float(df.select(pl.corr("ofi_top", "dmid")).item())
    assert corr > 0.5, f"OFI↔Δmid corr should be strongly positive, got {corr:.3f}"
