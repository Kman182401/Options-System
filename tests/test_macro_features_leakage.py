"""Leak-safety of macro features — the timing-vs-outcome rule, proven both ways.

The defining property of this layer: scheduled event *times* may be used ahead of
the event, but event *outcomes* may not be seen before they are released. These
tests prove (a) outcome features are strictly backward, (b) timing features use
the (known-ahead) schedule, and (c) the leak detector has teeth — a deliberately
forward (leaky) outcome look-up is caught by the very same invariance check that
the production path passes.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl

from options_system.features.macro_features import (
    _mins_to,
    _recent_change,
    compute_macro_features,
    macro_feature_names,
)
from options_system.macro.config import MacroConfig

_TS = pl.Datetime("us", "UTC")


def _events() -> pl.DataFrame:
    """Synthetic event table: monthly CPI + NFP, quarterly FOMC, with outcomes."""
    rows = []
    base = datetime(2022, 1, 12, 13, 30, tzinfo=UTC)  # a CPI-like 08:30 ET release
    for i in range(12):
        t = base + timedelta(days=30 * i)
        rows.append(
            {"event_time": t, "event_type": "cpi", "actual_pit": 280.0 + i, "prior": 279.0 + i}
        )
        rows.append(
            {
                "event_time": t + timedelta(days=3),
                "event_type": "nfp",
                "actual_pit": 150000.0 + 100 * i,
                "prior": 149900.0 + 100 * i,
            }
        )
    for i, day in enumerate(
        (
            datetime(2022, 1, 26, 19, 0, tzinfo=UTC),
            datetime(2022, 3, 16, 18, 0, tzinfo=UTC),
            datetime(2022, 5, 4, 18, 0, tzinfo=UTC),
        )
    ):
        rows.append(
            {
                "event_time": day,
                "event_type": "fomc",
                "actual_pit": 0.25 * (i + 2),
                "prior": 0.25 * (i + 1),
            }
        )
    return pl.DataFrame(rows).with_columns(pl.col("event_time").cast(_TS))


def _t0_grid() -> pl.Series:
    start = datetime(2022, 1, 1, tzinfo=UTC)
    return pl.Series("t0", [start + timedelta(hours=6 * k) for k in range(600)]).cast(_TS)


def test_columns_match_names():
    cfg = MacroConfig.load()
    out = compute_macro_features(_t0_grid(), _events(), cfg)
    assert out.columns == ["t0", *macro_feature_names(cfg)]


def test_outcome_features_are_strictly_backward():
    """Hiding the OUTCOMES of future events must not change outcome features at t0.

    We null actual/prior for every event released after a cut, keeping the rows
    (so the schedule is untouched), then assert the outcome columns are identical
    for all t0 <= cut. A backward look-up can only read events with
    event_time <= t0 <= cut, whose outcomes are untouched.
    """
    cfg = MacroConfig.load()
    events = _events()
    t0 = _t0_grid()
    cut = datetime(2022, 6, 1, tzinfo=UTC)

    hidden = events.with_columns(
        pl.when(pl.col("event_time") > pl.lit(cut).cast(_TS))
        .then(None)
        .otherwise(pl.col("actual_pit"))
        .alias("actual_pit"),
        pl.when(pl.col("event_time") > pl.lit(cut).cast(_TS))
        .then(None)
        .otherwise(pl.col("prior"))
        .alias("prior"),
    )

    full = compute_macro_features(t0, events, cfg)
    masked = compute_macro_features(t0, hidden, cfg)
    keep = (t0 <= cut).to_numpy()

    for col in [
        c for c in full.columns if c.startswith("macro_chg_") or c.startswith("macro_tendency_")
    ]:
        a = full[col].to_numpy()[keep]
        b = masked[col].to_numpy()[keep]
        assert np.array_equal(a, b, equal_nan=True), f"{col} changed when future outcomes hidden"


def test_timing_features_use_schedule_and_ignore_outcomes():
    """Timing features are available ahead of the event and never touch outcomes."""
    cfg = MacroConfig.load()
    events = _events()
    t0 = _t0_grid()

    # (a) minutes-to-next-FOMC just before a scheduled meeting is the right positive gap.
    just_before = pl.Series("t0", [datetime(2022, 3, 15, 18, 0, tzinfo=UTC)]).cast(_TS)
    one = compute_macro_features(just_before, events, cfg)
    # next FOMC is 2022-03-16 18:00 → exactly 24h = 1440 minutes ahead.
    assert abs(one["macro_mins_to_fomc"][0] - 1440.0) < 1e-6
    assert one["macro_mins_to_fomc"][0] > 0  # genuinely ahead of the event

    # (b) timing is invariant to outcomes: blanking every actual/prior leaves timing unchanged.
    no_outcomes = events.with_columns(
        pl.lit(None, dtype=pl.Float64).alias("actual_pit"),
        pl.lit(None, dtype=pl.Float64).alias("prior"),
    )
    full = compute_macro_features(t0, events, cfg)
    blanked = compute_macro_features(t0, no_outcomes, cfg)
    for col in [
        c
        for c in full.columns
        if "mins_to" in c or "mins_since" in c or c == "macro_events_next_24h"
    ]:
        assert np.array_equal(full[col].to_numpy(), blanked[col].to_numpy(), equal_nan=True), (
            f"timing col {col} depended on outcomes"
        )


def test_teeth_forward_outcome_leak_is_caught():
    """A deliberately FORWARD (leaky) outcome look-up fails the backward invariance check.

    This proves the test in :func:`test_outcome_features_are_strictly_backward` has
    teeth: the same hide-the-future-outcomes manipulation DOES change a forward
    look-up, so a real leak could not slip past it.
    """
    events = _events().filter(pl.col("event_type") == "fomc").sort("event_time")
    ev_ns = events["event_time"].dt.cast_time_unit("ns").dt.epoch("ns").to_numpy().astype("int64")
    actual = events["actual_pit"].to_numpy().astype("float64")
    prior = events["prior"].to_numpy().astype("float64")

    # A t0 strictly between the 2nd and 3rd FOMC events.
    t0 = pl.Series([datetime(2022, 4, 1, tzinfo=UTC)]).cast(_TS)
    t0_ns = t0.dt.cast_time_unit("ns").dt.epoch("ns").to_numpy().astype("int64")

    backward = _recent_change(t0_ns, ev_ns, actual, prior, forward=False)  # reads the 2nd event
    forward = _recent_change(
        t0_ns, ev_ns, actual, prior, forward=True
    )  # reads the 3rd (FUTURE) event

    # Hide the future (3rd) event's outcome.
    actual_hidden, prior_hidden = actual.copy(), prior.copy()
    actual_hidden[2] = np.nan
    prior_hidden[2] = np.nan
    backward_h = _recent_change(t0_ns, ev_ns, actual_hidden, prior_hidden, forward=False)
    forward_h = _recent_change(t0_ns, ev_ns, actual_hidden, prior_hidden, forward=True)

    # Backward is unaffected by hiding the future; forward is corrupted (leak caught).
    assert np.array_equal(backward, backward_h, equal_nan=True)
    assert not np.array_equal(forward, forward_h, equal_nan=True)
    assert np.isnan(forward_h[0]) and not np.isnan(backward[0])


def test_mins_to_empty_is_nan():
    out = _mins_to(np.array([0, 1, 2], dtype="int64"), np.empty(0, dtype="int64"))
    assert np.all(np.isnan(out))
