"""THE critical test: truncation-invariance proves every TA feature is leak-free.

For a set of timestamps ``t`` we compute each feature two ways:

* **full** — build the continuous series from raw outright bars over the whole
  window ``[start, END]`` (END is AFTER a roll), then compute TA features.
* **point-in-time (pit)** — rebuild the continuous series from raw bars truncated
  at ``t`` (only rolls known as of ``t``), then compute features, take row ``t``.

A causal, back-adjustment-invariant feature satisfies ``full[t] == pit[t]``. The
window is chosen so a roll (MES/MNQ 2024-03-10) sits between the cutoffs and END,
making the back-adjustment factor ``f_k != 1`` — every emitted TA feature is
degree-0 in price scale, so that factor cancels. The "teeth" test confirms a
deliberately forward-looking feature genuinely FAILS, so this test can catch a
real leak.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import cast

import polars as pl
import pytest

from options_system.data.continuous import build_continuous
from options_system.data.lake import Lake
from options_system.ta.compute import compute_ta, ta_feature_names
from options_system.ta.config import TaConfig

WINDOW_START = datetime(2024, 1, 2, tzinfo=UTC)
WINDOW_END = datetime(2024, 3, 25, tzinfo=UTC)  # after the 2024-03-10 roll
CUTOFFS = [datetime(2024, 2, 26, 15, 0, tzinfo=UTC), datetime(2024, 3, 4, 15, 0, tzinfo=UTC)]
SYMBOLS = ("MES", "MNQ")

_lake = Lake()


def _raw_outright(symbol: str) -> pl.DataFrame:
    df = cast("pl.DataFrame", _lake.scan("bars_1m", symbol).collect())
    return df.filter(~pl.col("contract_id").str.contains("-"))


def _continuous(
    raw: pl.DataFrame, rolls_all: pl.DataFrame, lo: datetime, hi: datetime
) -> pl.DataFrame:
    """Point-in-time continuous: raw bars in [lo, hi] + only rolls known as of hi."""
    sub = raw.filter((pl.col("ts_event") >= lo) & (pl.col("ts_event") <= hi))
    rolls = rolls_all.filter(pl.col("ts_event") <= hi)
    return build_continuous(sub, rolls, adjustment="ratio")


@pytest.fixture(scope="module")
def setup():
    cfg = TaConfig.load()
    raw = {s: _raw_outright(s) for s in SYMBOLS}
    rolls = {
        s: cast("pl.DataFrame", _lake.scan("roll_events", s).collect()).sort("ts_event")
        for s in SYMBOLS
    }
    if any(raw[s].is_empty() or rolls[s].is_empty() for s in SYMBOLS):
        pytest.skip("lake/roll_events not populated (run backfill + continuous stitch first)")
    cont_full = {s: _continuous(raw[s], rolls[s], WINDOW_START, WINDOW_END) for s in SYMBOLS}
    full = {s: compute_ta(cont_full[s], cfg) for s in SYMBOLS}
    return cfg, raw, rolls, full


# 1e-4 relative floor: most features match to ~1e-12; the only drift is float
# rounding in the EWM/rolling chains computed over back-adjusted prices. A genuine
# leak moves the value by the roll factor (~1%) — far above this (teeth confirms).
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
    names = ta_feature_names(cfg)
    for t in CUTOFFS:
        pit_cont = _continuous(raw[symbol], rolls[symbol], WINDOW_START, t)
        pit = compute_ta(pit_cont, cfg)
        pit_row = pit.filter(pl.col("ts_event") == t)
        full_row = full[symbol].filter(pl.col("ts_event") == t)
        assert pit_row.height == 1 and full_row.height == 1, f"missing row at {t}"
        pr, fr = pit_row.to_dicts()[0], full_row.to_dicts()[0]
        mismatches = {n: (fr[n], pr[n]) for n in names if not _match(pr[n], fr[n])}
        assert not mismatches, f"{symbol} @ {t}: leak (full,pit) in {mismatches}"


def test_leakage_test_has_teeth(setup):
    """A forward-looking feature (next-bar return) must FAIL truncation-invariance."""
    cfg, raw, rolls, full = setup
    t = CUTOFFS[0]
    pit_cont = _continuous(raw["MES"], rolls["MES"], WINDOW_START, t)
    full_cont = _continuous(raw["MES"], rolls["MES"], WINDOW_START, WINDOW_END)

    def _lead(cont: pl.DataFrame):
        leaked = cont.sort("ts_event").with_columns(
            (pl.col("close").shift(-1) / pl.col("close") - 1.0).alias("lead")
        )
        return leaked.filter(pl.col("ts_event") == t)["lead"][0]

    # pit truncates at t -> the next bar is unknown (null); full sees the real future
    assert not _match(_lead(pit_cont), _lead(full_cont)), "forward feature unexpectedly invariant"
