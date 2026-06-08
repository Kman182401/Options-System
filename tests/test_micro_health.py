"""Microstructure health/QA gathering on synthetic bars (pure, no I/O)."""

from __future__ import annotations

from _micro_helpers import TEST_INST, comoving_stream
from options_system.microstructure.bars import assemble_features, build_dollar_bars
from options_system.microstructure.config import MicrostructureConfig
from options_system.observability.micro_health import gather_micro_health

CFG = MicrostructureConfig.load()


def _frame(n_bars: int):
    return assemble_features(
        build_dollar_bars(comoving_stream(n_bars), instrument=TEST_INST, session=CFG.session),
        symbol="T",
        cfg=CFG,
    )


def test_gather_reports_core_fields():
    df = _frame(40)
    [info] = gather_micro_health({"T": df}, CFG)
    assert info["symbol"] == "T"
    assert info["bars"] == df.height >= 30
    assert info["sessions"] >= 1
    assert info["bars_per_session_median"] > 0
    assert "ofi_vs_dmid_corr" in info
    # the synthetic stream is constructed so OFI and Δmid co-move
    assert info["ofi_vs_dmid_corr"] is not None and info["ofi_vs_dmid_corr"] > 0.5
    assert info["rolls_observed"] == 0  # single contract in the synthetic stream


def test_gather_clean_synthetic_has_no_nan_inf():
    df = _frame(20)
    [info] = gather_micro_health({"T": df}, CFG)
    # The only "bad" count allowed is the structural warmup null of the lag feature
    # on the first bar; no real feature should be NaN/inf on this clean stream.
    counts = info["nan_inf_counts"]
    assert set(counts).issubset({"ofi_top_lag1"})
    assert counts.get("ofi_top_lag1", 0) <= 1


def test_gather_empty_symbol():
    empty = assemble_features([], symbol="T", cfg=CFG)
    [info] = gather_micro_health({"T": empty}, CFG)
    assert info["bars"] == 0
