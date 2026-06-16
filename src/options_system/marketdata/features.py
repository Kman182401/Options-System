"""Point-in-time cross-asset (x1) features from the daily market-state lake (System B).

Given the long-format daily lake (``series_id, obs_date, value, observed_at, ...``) and a
set of target times (a label's ``t0``), build causal cross-asset features. The single
leakage rule, identical in spirit to the sentiment aggregator: a feature for target time
``t`` may use, per series, only the **latest observation with ``observed_at <= t``** — and
``observed_at`` is the END of the observation day (see :mod:`lake`), so same-day EOD values
are never used to predict that same day. Everything is an as-of lookup against
already-known history; nothing reads the future.

Emitted columns (deterministic order):

* ``mkt_<label>_level``          — latest known level.
* ``mkt_<label>_chg_<h>d``       — level minus the level ``h`` observations earlier.
* ``mkt_<label>_z_<W>d``         — z-score of the level over the trailing ``W`` observations.
* ``mkt_curve_<long>_<short>``   — a term-structure spread (long level − short level).

A feature is null when there is insufficient known history for it (e.g. fewer than ``h``
prior observations); levels are null before a series' first observation. Null means
"not knowable yet", never zero.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

if TYPE_CHECKING:
    from options_system.marketdata.config import MarketDataConfig


def market_feature_names(cfg: MarketDataConfig) -> list[str]:
    """Deterministic, ordered list of emitted ``mkt_*`` feature column names."""
    names: list[str] = []
    w = cfg.features.zscore_window_days
    for s in cfg.series:
        names.append(f"mkt_{s.label}_level")
        for h in cfg.features.change_horizons_days:
            names.append(f"mkt_{s.label}_chg_{h}d")
        names.append(f"mkt_{s.label}_z_{w}d")
    for long_l, short_l in cfg.features.curve_pairs:
        names.append(f"mkt_curve_{long_l}_{short_l}")
    return names


class _SeriesArrays:
    """One series' history as sorted parallel arrays for as-of lookups."""

    __slots__ = ("obs_us", "value")

    def __init__(self, obs_us: np.ndarray, value: np.ndarray) -> None:
        self.obs_us = obs_us
        self.value = value

    def asof_index(self, t_us: int) -> int:
        """Index of the latest observation with ``observed_at <= t``; -1 if none."""
        return int(np.searchsorted(self.obs_us, t_us, side="right")) - 1


def _ensure_observed_utc(frame: pl.DataFrame) -> pl.DataFrame:
    dtype = frame.schema["observed_at"]
    if isinstance(dtype, pl.Datetime) and dtype.time_zone is None:
        return frame.with_columns(pl.col("observed_at").dt.replace_time_zone("UTC"))
    return frame


def _series_arrays(market_frame: pl.DataFrame, cfg: MarketDataConfig) -> dict[str, _SeriesArrays]:
    """Build per-label sorted (observed_at_us, value) arrays from the long lake frame."""
    frame = _ensure_observed_utc(market_frame)
    out: dict[str, _SeriesArrays] = {}
    for s in cfg.series:
        sub = frame.filter(pl.col("series_id") == s.id).sort("observed_at")
        if sub.height == 0:
            out[s.label] = _SeriesArrays(np.empty(0, np.int64), np.empty(0, np.float64))
            continue
        obs = sub["observed_at"].to_numpy().astype("datetime64[us]").astype(np.int64)
        val = sub["value"].to_numpy().astype(np.float64)
        out[s.label] = _SeriesArrays(obs, val)
    return out


def _target_times_us(target_times: pl.Series | Sequence[datetime] | np.ndarray) -> np.ndarray:
    s = target_times if isinstance(target_times, pl.Series) else pl.Series("t", list(target_times))
    s = s.cast(pl.Datetime("us")) if s.dtype == pl.Date else s
    dtype = s.dtype
    if isinstance(dtype, pl.Datetime):
        s = (
            s.dt.replace_time_zone("UTC")
            if dtype.time_zone is None
            else s.dt.convert_time_zone("UTC")
        )
    return s.to_numpy().astype("datetime64[us]").astype(np.int64)


def _level(arr: _SeriesArrays, idx: int) -> float | None:
    return float(arr.value[idx]) if idx >= 0 else None


def _change(arr: _SeriesArrays, idx: int, h: int) -> float | None:
    if idx < 0 or idx - h < 0:
        return None
    return float(arr.value[idx] - arr.value[idx - h])


def _zscore(arr: _SeriesArrays, idx: int, w: int) -> float | None:
    if idx < 0 or idx - w + 1 < 0:
        return None
    window = arr.value[idx - w + 1 : idx + 1]
    sd = float(window.std())
    if sd == 0.0:
        return 0.0
    return float((arr.value[idx] - float(window.mean())) / sd)


def build_market_features_for_times(
    market_frame: pl.DataFrame,
    target_times: pl.Series | Sequence[datetime] | np.ndarray,
    cfg: MarketDataConfig,
) -> pl.DataFrame:
    """Cross-asset features at each target time (point-in-time, no look-ahead).

    One row per target time (input order) with a ``target_time`` key, every column from
    :func:`market_feature_names`, and the ``marketdata_feature_version`` stamp.
    """
    names = market_feature_names(cfg)
    out_schema: dict[str, pl.DataType] = {"target_time": pl.Datetime("us", "UTC")}
    for n in names:
        out_schema[n] = pl.Float64()
    out_schema["marketdata_feature_version"] = pl.Utf8()

    t_us = _target_times_us(target_times)
    if t_us.size == 0:
        return pl.DataFrame(schema=out_schema)

    arrays = _series_arrays(market_frame, cfg)
    w = cfg.features.zscore_window_days
    horizons = cfg.features.change_horizons_days
    cols: dict[str, list[float | None]] = {n: [] for n in names}

    for ti in t_us.tolist():
        idx = {label: arr.asof_index(int(ti)) for label, arr in arrays.items()}
        for s in cfg.series:
            arr = arrays[s.label]
            i = idx[s.label]
            cols[f"mkt_{s.label}_level"].append(_level(arr, i))
            for h in horizons:
                cols[f"mkt_{s.label}_chg_{h}d"].append(_change(arr, i, h))
            cols[f"mkt_{s.label}_z_{w}d"].append(_zscore(arr, i, w))
        for long_l, short_l in cfg.features.curve_pairs:
            lvl_l = _level(arrays[long_l], idx[long_l])
            lvl_s = _level(arrays[short_l], idx[short_l])
            spread = lvl_l - lvl_s if (lvl_l is not None and lvl_s is not None) else None
            cols[f"mkt_curve_{long_l}_{short_l}"].append(spread)

    data: dict[str, object] = {"target_time": t_us.astype("datetime64[us]")}
    data.update(cols)
    data["marketdata_feature_version"] = [cfg.marketdata_feature_version] * int(t_us.size)
    return pl.DataFrame(data, schema=out_schema)


def attach_market_asof(
    labels_or_events: pl.DataFrame,
    market_frame: pl.DataFrame,
    cfg: MarketDataConfig,
    *,
    time_col: str = "t0",
) -> pl.DataFrame:
    """Attach the point-in-time ``mkt_*`` features onto ``labels_or_events`` by ``time_col``.

    Purely point-in-time: row ``r`` uses only series values with ``observed_at <=
    r[time_col]``. Every input row is preserved (missing context is null, not dropped);
    features align positionally to the input rows.
    """
    if time_col not in labels_or_events.columns:
        raise ValueError(f"attach_market_asof: time column {time_col!r} not in frame")
    feats = build_market_features_for_times(market_frame, labels_or_events[time_col], cfg)
    feats = feats.drop("target_time")
    return labels_or_events.hstack(feats)
