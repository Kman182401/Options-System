"""Point-in-time label joins for sentiment features (Phase 17 — offline scaffold).

Attach the aggregate sentiment features (:mod:`options_system.sentiment.features`) onto
label tables — the short-horizon micro labels (``data/micro_labels/``) and the daily
labels (``data/labels/``) — keyed on the label event time ``t0``. The join is purely
point-in-time: a label at ``t0`` only ever sees sentiment with ``observed_at <= t0``.

Leakage discipline (the whole reason this layer exists):

* The join key is ``t0`` (the event/decision time), never ``t1`` (the barrier
  resolution time) and never the realized return — those are outcomes, not inputs.
* Label outcome columns (``label``, ``ret``/``ret_t1``, ``barrier``…) flow through
  untouched as passengers; they are NEVER fed into the sentiment features.
* Every label row is preserved. "No sentiment" means no prior coverage (zero counts /
  null scores), it does not drop the sample.

Each helper returns ``(attached_frame, coverage)`` where ``coverage`` is a plain dict
of point-in-time coverage metadata (row counts, coverage rate, events used, time
ranges, feature version, windows) for the coverage report and observability.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

from options_system.sentiment.features import (
    attach_sentiment_asof,
    dedup_scored_events,
    normalize_scored_events,
    sentiment_feature_names,
)

if TYPE_CHECKING:
    from options_system.sentiment.config import SentimentConfig

_US_PER_MINUTE = 60_000_000


def _widest_window(cfg: SentimentConfig) -> tuple[str, int]:
    """The window with the most minutes — the horizon used for 'has any sentiment'."""
    name, minutes = max(cfg.aggregation.windows.items(), key=lambda kv: kv[1])
    return name, minutes


def _minmax_str(frame: pl.DataFrame, col: str) -> dict[str, str | None]:
    if frame.height == 0 or col not in frame.columns:
        return {"min": None, "max": None}
    sub = frame.select(col).drop_nulls()
    if sub.height == 0:
        return {"min": None, "max": None}
    return {"min": str(sub[col].min()), "max": str(sub[col].max())}


def _events_used(scored_non_degraded: pl.DataFrame, label_us: np.ndarray, wmax_min: int) -> int:
    """Count distinct events that land in at least one label's widest window.

    An event at ``observed_at = o`` is used by a label at ``t`` iff ``t - W < o <= t``,
    equivalently ``o <= t < o + W``. With sorted label times this is a pair of
    ``searchsorted`` lookups per event — no double counting across labels.
    """
    if scored_non_degraded.height == 0 or label_us.size == 0:
        return 0
    obs = scored_non_degraded["observed_at"].to_numpy().astype("datetime64[us]").astype(np.int64)
    w_us = wmax_min * _US_PER_MINUTE
    labels_sorted = np.sort(label_us)
    used = 0
    for o in obs.tolist():
        lo = int(np.searchsorted(labels_sorted, o, side="left"))  # first t >= o
        hi = int(np.searchsorted(labels_sorted, o + w_us, side="left"))  # first t >= o + W
        if hi > lo:
            used += 1
    return used


def _coverage(
    labels: pl.DataFrame,
    attached: pl.DataFrame,
    scored_norm: pl.DataFrame,
    cfg: SentimentConfig,
    *,
    time_col: str,
    duplicates: int,
) -> dict[str, Any]:
    """Build the point-in-time coverage metadata for an attached label frame."""
    agg = cfg.aggregation
    wname, wmin = _widest_window(cfg)
    widest_count_col = f"sent_{wname}_count"
    rows = labels.height
    non_degraded = scored_norm.filter(~pl.col("degraded"))

    rows_with_any = (
        int(attached.filter(pl.col(widest_count_col) > 0).height)
        if widest_count_col in attached.columns
        else 0
    )
    coverage_by_window = {
        w: (
            int(attached.filter(pl.col(f"sent_{w}_count") > 0).height)
            if f"sent_{w}_count" in attached.columns
            else 0
        )
        for w in agg.windows
    }

    label_us = (
        labels[time_col].to_numpy().astype("datetime64[us]").astype(np.int64)
        if rows
        else np.empty(0, dtype=np.int64)
    )

    return {
        "rows": rows,
        "rows_with_any_sentiment": rows_with_any,
        "coverage_rate": (rows_with_any / rows) if rows else 0.0,
        "events_used": _events_used(non_degraded, label_us, wmin),
        "scored_rows": int(scored_norm.height),
        "scored_non_degraded": int(non_degraded.height),
        "degraded_count": int(scored_norm.filter(pl.col("degraded")).height),
        "duplicate_count": int(duplicates),
        "observed_at": _minmax_str(non_degraded, "observed_at"),
        "label_time": _minmax_str(labels, time_col),
        "feature_version": agg.feature_version,
        "windows": list(agg.windows),
        "coverage_by_window": coverage_by_window,
    }


def _attach(
    labels: pl.DataFrame,
    scored_events: pl.DataFrame,
    cfg: SentimentConfig,
    *,
    time_col: str,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Shared core: normalize/dedup scored events, attach as-of, build coverage."""
    if time_col not in labels.columns:
        raise ValueError(f"label frame has no {time_col!r} column to join on")
    scored_norm = normalize_scored_events(scored_events)
    scored_norm, duplicates = dedup_scored_events(scored_norm)
    attached = attach_sentiment_asof(labels, scored_norm, cfg, time_col=time_col)
    coverage = _coverage(
        labels, attached, scored_norm, cfg, time_col=time_col, duplicates=duplicates
    )
    return attached, coverage


def attach_to_micro_labels(
    labels: pl.DataFrame,
    scored_events: pl.DataFrame,
    cfg: SentimentConfig,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Attach sentiment features to short-horizon micro labels on ``t0`` (point-in-time).

    Uses only ``t0``; preserves every label row; never reads ``t1`` / ``ret_t1`` / the
    label outcome. Returns ``(attached, coverage)``.
    """
    return _attach(labels, scored_events, cfg, time_col="t0")


def attach_to_daily_labels(
    labels: pl.DataFrame,
    scored_events: pl.DataFrame,
    cfg: SentimentConfig,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Attach sentiment features to daily labels on the label event time ``t0``.

    Uses only ``t0``; preserves every label row; never reads ``t1`` / ``ret`` / the label
    outcome. Returns ``(attached, coverage)``.
    """
    return _attach(labels, scored_events, cfg, time_col="t0")


def null_feature_count(attached: pl.DataFrame, cfg: SentimentConfig) -> int:
    """Total null cells across the emitted ``sent_*`` feature columns (a QA signal)."""
    cols = [c for c in sentiment_feature_names(cfg) if c in attached.columns]
    if not cols or attached.height == 0:
        return 0
    return int(attached.select(pl.sum_horizontal(pl.col(c).is_null().sum() for c in cols)).item())


def feature_columns_stable(attached: pl.DataFrame, cfg: SentimentConfig) -> bool:
    """True iff every declared feature column is present in the attached frame, in order."""
    names = sentiment_feature_names(cfg)
    present = [c for c in attached.columns if c in set(names)]
    return present == names
