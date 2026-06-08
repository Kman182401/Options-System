"""THE leakage test for the microstructure layer — truncation-invariance + teeth.

A causal feature at bar t depends only on events with ts <= bar t's close, so
rebuilding from the event stream **truncated at bar k's close** must reproduce
bars 0..k bit-for-bit. A forward-looking feature does NOT survive truncation —
the final teeth assertion plants one (a feature that reads the NEXT bar) and
confirms it breaks invariance, proving the check can actually catch a leak.

Mirrors tests/test_features_leakage.py (the v1 price-feature teeth test).
"""

from __future__ import annotations

import math

import polars as pl

from _micro_helpers import TEST_INST, comoving_stream
from options_system.microstructure.bars import (
    assemble_features,
    build_dollar_bars,
    feature_names,
)
from options_system.microstructure.config import MicrostructureConfig

CFG = MicrostructureConfig.load()


def _match(a, b, tol=1e-9) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    if isinstance(a, float) and (math.isnan(a) or math.isnan(b)):
        return math.isnan(a) and math.isnan(b)
    return abs(a - b) <= tol * (1.0 + abs(b))


def test_truncation_invariance_every_feature():
    events = comoving_stream(40)
    full_raw = build_dollar_bars(events, instrument=TEST_INST, session=CFG.session)
    assert len(full_raw) >= 20
    k = len(full_raw) // 2
    t_close_ns = full_raw[k]["ts_close_ns"]

    pit_events = [e for e in events if e.ts_ns <= t_close_ns]
    pit_raw = build_dollar_bars(pit_events, instrument=TEST_INST, session=CFG.session)

    full = assemble_features(full_raw, symbol="T", cfg=CFG)
    pit = assemble_features(pit_raw, symbol="T", cfg=CFG)
    assert pit.height == k + 1, "truncation should reproduce exactly bars 0..k"

    names = feature_names(CFG)
    full_head = full.head(k + 1)
    for name in names:
        fa, pa = full_head[name].to_list(), pit[name].to_list()
        mism = [(i, fa[i], pa[i]) for i in range(k + 1) if not _match(fa[i], pa[i])]
        assert not mism, f"non-causal feature {name!r}: full vs pit mismatch {mism[:3]}"


def test_causality_open_before_close():
    full = assemble_features(
        build_dollar_bars(comoving_stream(10), instrument=TEST_INST, session=CFG.session),
        symbol="T",
        cfg=CFG,
    )
    assert (full["ts_open"] <= full["ts_event"]).all()


def test_leakage_check_has_teeth():
    """A forward-looking feature (reads the NEXT bar) must FAIL truncation-invariance."""
    events = comoving_stream(40)
    full_raw = build_dollar_bars(events, instrument=TEST_INST, session=CFG.session)
    k = len(full_raw) // 2
    t_close_ns = full_raw[k]["ts_close_ns"]
    pit_raw = build_dollar_bars(
        [e for e in events if e.ts_ns <= t_close_ns], instrument=TEST_INST, session=CFG.session
    )

    full = assemble_features(full_raw, symbol="T", cfg=CFG).with_columns(
        pl.col("ofi_top").shift(-1).alias("leaky_ofi")  # reads bar t+1 -> a leak
    )
    pit = assemble_features(pit_raw, symbol="T", cfg=CFG).with_columns(
        pl.col("ofi_top").shift(-1).alias("leaky_ofi")
    )
    full_leaky_k = full["leaky_ofi"].to_list()[k]
    pit_leaky_k = pit["leaky_ofi"].to_list()[k]  # last row of pit -> no t+1 -> null
    assert full_leaky_k is not None, "full stream should have bar k+1 to leak from"
    assert pit_leaky_k is None, "truncated stream has no future bar to leak from"
    assert not _match(full_leaky_k, pit_leaky_k), "teeth failed: leak not detected by truncation"
