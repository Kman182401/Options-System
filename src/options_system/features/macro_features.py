"""Leak-safe macro / economic-event features, evaluated at arbitrary timestamps.

These features turn the ``macro_events`` table (``options_system.macro.ingest``)
into per-timestamp inputs for the signal model. They are computed **directly at
the label ``t0`` timestamps** (not stored on the bar grid): timing features are
closed-form from the release schedule, and outcome features are a backward
look-up — so this generalises unchanged to the live engine.

The point-in-time rule has two halves, and they are implemented by two different
look-ups — this is the whole leakage game:

* **Timing features (known ahead).** "Minutes to the next FOMC / CPI" looks
  **forward** at the release *schedule* (``event_time`` only — never an outcome).
  Release calendars are public in advance, so this is legitimately available
  before the event. Implemented with ``searchsorted`` over the sorted
  ``event_time`` array.
* **Outcome features (known only at release).** "Most-recent actual − prior" and
  rolling tendencies index **only into events with ``event_time <= t0``** (a
  strictly backward look-up), so a bar at ``t0`` can never see a value released
  after it. The leaky mirror (``forward=True`` in :func:`_recent_change`) exists
  *only* so the test suite can prove the leakage detector has teeth.

Macro context is instrument-independent, so the same features apply to MES and
MNQ at a given timestamp. Undefined values (before the first event of a type, or
no scheduled next event in the data) are ``NaN`` — LightGBM handles missing
values natively, so this never drops a row downstream.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from ..macro.config import MacroConfig

_TS = pl.Datetime("us", "UTC")
_NS_PER_MIN = 60_000_000_000  # nanoseconds per minute


def macro_feature_names(cfg: MacroConfig) -> list[str]:
    """Ordered macro feature column names implied by the config (deterministic)."""
    mf = cfg.features
    names: list[str] = []
    for t in mf.timing_types:
        names += [f"macro_mins_to_{t}", f"macro_mins_since_{t}"]
    names += [
        "macro_mins_to_event",
        "macro_mins_since_event",
        "macro_in_blackout",
        "macro_events_next_24h",
        "macro_is_fomc_day",
        "macro_is_nfp_day",
    ]
    names += [f"macro_chg_{t}" for t in mf.outcome_types]
    names += [f"macro_tendency_{t}" for t in mf.tendency_types]
    return names


# --------------------------------------------------------------------------- #
# Per-type building blocks (explicit, leak-safe look-ups over event_time arrays)
# --------------------------------------------------------------------------- #
def _epoch_ns(times: pl.Series) -> np.ndarray:
    """Datetime series → int64 nanoseconds since epoch (UTC)."""
    return times.dt.cast_time_unit("ns").dt.epoch("ns").to_numpy().astype("int64")


def _type_arrays(events: pl.DataFrame, etype: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sorted ``(event_time_ns, actual_pit, prior)`` arrays for one event type."""
    e = events.filter(pl.col("event_type") == etype).sort("event_time")
    if e.is_empty():
        z = np.empty(0, dtype="int64")
        f = np.empty(0, dtype="float64")
        return z, f, f
    return (
        _epoch_ns(e["event_time"]),
        e["actual_pit"].to_numpy().astype("float64"),
        e["prior"].to_numpy().astype("float64"),
    )


def _mins_since(t0_ns: np.ndarray, ev_ns: np.ndarray) -> np.ndarray:
    """Minutes since the last event with ``event_time <= t0`` (NaN if none)."""
    out = np.full(t0_ns.shape, np.nan)
    if ev_ns.size == 0:
        return out
    idx_prev = np.searchsorted(ev_ns, t0_ns, side="right") - 1  # last event_time <= t0
    ok = idx_prev >= 0
    out[ok] = (t0_ns[ok] - ev_ns[idx_prev[ok]]) / _NS_PER_MIN
    return out


def _mins_to(t0_ns: np.ndarray, ev_ns: np.ndarray) -> np.ndarray:
    """Minutes to the next event with ``event_time > t0`` (NaN if none scheduled).

    Looks **forward** at the schedule (``event_time`` only) — legitimate because
    release calendars are public in advance.
    """
    out = np.full(t0_ns.shape, np.nan)
    if ev_ns.size == 0:
        return out
    idx_next = np.searchsorted(ev_ns, t0_ns, side="right")  # first event_time > t0
    ok = idx_next < ev_ns.size
    out[ok] = (ev_ns[idx_next[ok]] - t0_ns[ok]) / _NS_PER_MIN
    return out


def _recent_change(
    t0_ns: np.ndarray,
    ev_ns: np.ndarray,
    actual: np.ndarray,
    prior: np.ndarray,
    *,
    forward: bool = False,
) -> np.ndarray:
    """``actual - prior`` of the most-recent release with ``event_time <= t0`` (NaN if none).

    ``forward=True`` is the deliberately-LEAKY mirror (it reads the *next* event's
    outcome, ``event_time > t0``). It exists only for the leakage "teeth" test —
    the production path is always ``forward=False`` (strictly backward).
    """
    out = np.full(t0_ns.shape, np.nan)
    if ev_ns.size == 0:
        return out
    if forward:
        idx = np.searchsorted(ev_ns, t0_ns, side="right")  # next release (FUTURE — leak)
        ok = idx < ev_ns.size
    else:
        idx = np.searchsorted(ev_ns, t0_ns, side="right") - 1  # last release <= t0
        ok = idx >= 0
    out[ok] = actual[idx[ok]] - prior[idx[ok]]
    return out


def _recent_tendency(
    t0_ns: np.ndarray, ev_ns: np.ndarray, actual: np.ndarray, prior: np.ndarray, window: int
) -> np.ndarray:
    """Trailing mean of the last ``window`` release changes, as of ``t0`` (backward)."""
    out = np.full(t0_ns.shape, np.nan)
    if ev_ns.size == 0:
        return out
    chg = actual - prior  # first release has NaN prior → NaN change
    # Trailing mean over the event sequence (ignoring the leading NaN change).
    tend = np.full(chg.shape, np.nan)
    for i in range(chg.size):
        lo = max(0, i - window + 1)
        w = chg[lo : i + 1]
        w = w[~np.isnan(w)]
        if w.size:
            tend[i] = w.mean()
    idx_prev = np.searchsorted(ev_ns, t0_ns, side="right") - 1
    ok = idx_prev >= 0
    out[ok] = tend[idx_prev[ok]]
    return out


# --------------------------------------------------------------------------- #
# Public: compute all macro features at a set of timestamps
# --------------------------------------------------------------------------- #
def compute_macro_features(t0: pl.Series, events: pl.DataFrame, cfg: MacroConfig) -> pl.DataFrame:
    """Macro features at each ``t0`` (UTC). Returns ``t0`` + the macro columns.

    Leak-free by construction: timing features read only ``event_time`` (the
    public schedule, forward-OK); outcome features read only events with
    ``event_time <= t0``. The output is keyed on ``t0`` so the caller can join it
    onto the label matrix. With no events the macro columns are all ``NaN``.
    """
    mf = cfg.features
    names = macro_feature_names(cfg)
    t0 = t0.cast(_TS)
    t0_ns = _epoch_ns(t0)
    cols: dict[str, np.ndarray] = {}

    # --- per-type timing (schedule, known ahead) ---
    for t in mf.timing_types:
        ev_ns, _, _ = _type_arrays(events, t)
        cols[f"macro_mins_to_{t}"] = _mins_to(t0_ns, ev_ns)
        cols[f"macro_mins_since_{t}"] = _mins_since(t0_ns, ev_ns)

    # --- aggregate high-impact timing (distinct event times across all hi types) ---
    hi = (
        events.filter(pl.col("event_type").is_in(cfg.high_impact_types()))
        .select("event_time")
        .unique()
        .sort("event_time")
    )
    hi_ns = _epoch_ns(hi["event_time"]) if not hi.is_empty() else np.empty(0, dtype="int64")
    mins_to_evt = _mins_to(t0_ns, hi_ns)
    cols["macro_mins_to_event"] = mins_to_evt
    cols["macro_mins_since_event"] = _mins_since(t0_ns, hi_ns)
    # In the pre-event blackout window: a high-impact event is <= lead minutes ahead.
    lead = float(mf.blackout_lead_minutes)
    cols["macro_in_blackout"] = np.where(
        np.isnan(mins_to_evt), 0.0, (mins_to_evt <= lead).astype("float64")
    )
    # Count of high-impact events scheduled in (t0, t0 + horizon].
    horizon_ns = mf.next_horizon_hours * 60 * _NS_PER_MIN
    left = np.searchsorted(hi_ns, t0_ns, side="right")
    right = np.searchsorted(hi_ns, t0_ns + horizon_ns, side="right")
    cols["macro_events_next_24h"] = (right - left).astype("float64")

    # --- is-FOMC-day / is-NFP-day (schedule; ET calendar-date membership) ---
    cols["macro_is_fomc_day"] = _is_event_day(t0, events, "fomc", cfg.timezone)
    cols["macro_is_nfp_day"] = _is_event_day(t0, events, "nfp", cfg.timezone)

    # --- per-type outcome change (backward only) ---
    for t in mf.outcome_types:
        ev_ns, actual, prior = _type_arrays(events, t)
        cols[f"macro_chg_{t}"] = _recent_change(t0_ns, ev_ns, actual, prior)

    # --- short rolling tendency (backward only) ---
    for t in mf.tendency_types:
        ev_ns, actual, prior = _type_arrays(events, t)
        cols[f"macro_tendency_{t}"] = _recent_tendency(
            t0_ns, ev_ns, actual, prior, mf.tendency_window
        )

    frame = pl.DataFrame({"t0": t0}).with_columns(
        [pl.Series(name, cols[name], dtype=pl.Float64) for name in names]
    )
    return frame


def _is_event_day(t0: pl.Series, events: pl.DataFrame, etype: str, tz: str) -> np.ndarray:
    """1.0 where ``t0``'s ET calendar date is a scheduled ``etype`` day (else 0.0).

    Schedule-based (uses ``event_time`` only), so it is legitimately known ahead.
    """
    e = events.filter(pl.col("event_type") == etype)
    if e.is_empty():
        return np.zeros(t0.len(), dtype="float64")
    days = set(e["event_time"].dt.convert_time_zone(tz).dt.date().to_list())
    t0_days = t0.dt.convert_time_zone(tz).dt.date().to_list()
    return np.array([1.0 if d in days else 0.0 for d in t0_days], dtype="float64")
