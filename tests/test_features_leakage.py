"""THE critical test: truncation-invariance proves every feature is leak-free.

For a set of timestamps ``t`` we compute each feature two ways:

* **full** — build the continuous series from raw outright bars over the whole
  window ``[start, END]`` (END is AFTER a roll), then compute features.
* **point-in-time (pit)** — rebuild the continuous series from raw bars truncated
  at ``t`` (only rolls known as of ``t``), then compute features, take row ``t``.

A causal, back-adjustment-invariant feature satisfies ``full[t] == pit[t]``. A
forward-looking feature, or a price *level* on the continuous series (whose
back-adjustment secretly encodes the future roll between ``t`` and END), does NOT
— and the final "teeth" test confirms a level feature genuinely fails, so this
test can actually catch a leak.

The window is chosen so a roll (MES/MNQ 2024-03-10) sits between the cutoffs and
END, making the back-adjustment factor ``f_k != 1``.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime

import polars as pl
import pytest

from options_system.data.continuous import build_continuous
from options_system.data.lake import Lake
from options_system.features.compute import compute_features, feature_names
from options_system.features.config import FeatureConfig

WINDOW_START = datetime(2024, 1, 2, tzinfo=UTC)
WINDOW_END = datetime(2024, 3, 25, tzinfo=UTC)  # after the 2024-03-10 roll
CUTOFFS = [datetime(2024, 2, 26, 15, 0, tzinfo=UTC), datetime(2024, 3, 4, 15, 0, tzinfo=UTC)]
SYMBOLS = ("MES", "MNQ")

_lake = Lake()


def _raw_outright(symbol: str) -> pl.DataFrame:
    df = _lake.scan("bars_1m", symbol).collect()
    return df.filter(~pl.col("contract_id").str.contains("-"))


def _continuous(
    raw: pl.DataFrame, rolls_all: pl.DataFrame, lo: datetime, hi: datetime
) -> pl.DataFrame:
    """Point-in-time continuous: raw bars in [lo, hi] + only rolls known as of hi.

    Uses the persisted full-history roll set (filtered to ``<= hi``), exactly like
    production ``store.get_bars(continuous=True)`` — NOT detect_rolls on the slice
    (an empty roll set would make build_continuous return every overlapping
    contract). Truncating ``hi`` drops future rolls, which is the whole point.
    """
    sub = raw.filter((pl.col("ts_event") >= lo) & (pl.col("ts_event") <= hi))
    rolls = rolls_all.filter(pl.col("ts_event") <= hi)
    return build_continuous(sub, rolls, adjustment="ratio")


@pytest.fixture(scope="module")
def setup():
    cfg = FeatureConfig.load()
    raw = {s: _raw_outright(s) for s in SYMBOLS}
    rolls = {s: _lake.scan("roll_events", s).collect().sort("ts_event") for s in SYMBOLS}
    if any(raw[s].is_empty() or rolls[s].is_empty() for s in SYMBOLS):
        pytest.skip("lake/roll_events not populated (run backfill + continuous stitch first)")
    cont_full = {s: _continuous(raw[s], rolls[s], WINDOW_START, WINDOW_END) for s in SYMBOLS}
    full = {
        s: compute_features(cont_full[s], cfg, symbol=s, other=cont_full[_other(s)])
        for s in SYMBOLS
    }
    return cfg, raw, rolls, full


def _other(symbol: str) -> str:
    return SYMBOLS[1] if symbol == SYMBOLS[0] else SYMBOLS[0]


# 1e-4 relative = 0.01%. Most features match to ~1e-12; the exception is long
# Wilder-EWM chains (ADX) computed over back-adjusted prices ``close * f_k`` —
# ``(a*f_k - b*f_k)`` differs from ``(a-b)*f_k`` in the last bits, accumulating to
# ~1e-4. A genuine leak (a price LEVEL, or a forward window) moves the value by
# the roll factor (~1%) or more — far above this floor (the teeth test confirms).
_TOL = 1e-4


def _match(a, b, tol=_TOL) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    if isinstance(a, float) and (math.isnan(a) or math.isnan(b)):
        return math.isnan(a) and math.isnan(b)
    return abs(a - b) <= tol * (1.0 + abs(b))


@pytest.mark.parametrize("symbol", SYMBOLS)
def test_truncation_invariance_every_feature(setup, symbol):
    cfg, raw, rolls, full = setup
    names = feature_names(cfg)
    for t in CUTOFFS:
        pit_cont = {s: _continuous(raw[s], rolls[s], WINDOW_START, t) for s in SYMBOLS}
        pit = compute_features(pit_cont[symbol], cfg, symbol=symbol, other=pit_cont[_other(symbol)])
        pit_row = pit.filter(pl.col("ts_event") == t)
        full_row = full[symbol].filter(pl.col("ts_event") == t)
        assert pit_row.height == 1 and full_row.height == 1, f"missing row at {t}"
        pr, fr = pit_row.to_dicts()[0], full_row.to_dicts()[0]
        mismatches = {n: (fr[n], pr[n]) for n in names if not _match(pr[n], fr[n])}
        assert not mismatches, f"{symbol} @ {t}: leak (full,pit) in {mismatches}"


def test_leakage_test_has_teeth(setup):
    """A raw continuous-price LEVEL must FAIL truncation-invariance (sanity on the test)."""
    cfg, raw, rolls, full = setup
    t = CUTOFFS[0]
    pit_cont = _continuous(raw["MES"], rolls["MES"], WINDOW_START, t)
    full_cont = _continuous(raw["MES"], rolls["MES"], WINDOW_START, WINDOW_END)
    pit_level = pit_cont.filter(pl.col("ts_event") == t)["close"][0]
    full_level = full_cont.filter(pl.col("ts_event") == t)["close"][0]
    # back-adjustment rewrote the past level using the future roll -> they differ
    assert not _match(pit_level, full_level), "level unexpectedly invariant — no roll in window?"
    # and the raw recovered price (close/adj_factor) IS invariant (what we use in ratios)
    pit_raw = (
        pit_cont.filter(pl.col("ts_event") == t)["close"][0]
        / pit_cont.filter(pl.col("ts_event") == t)["adj_factor"][0]
    )
    full_raw = (
        full_cont.filter(pl.col("ts_event") == t)["close"][0]
        / full_cont.filter(pl.col("ts_event") == t)["adj_factor"][0]
    )
    assert _match(pit_raw, full_raw), "raw front price should be back-adjustment invariant"
