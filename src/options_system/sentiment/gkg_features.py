"""Point-in-time daily GKG tone aggregation (s3) — standalone, isolated from s2.

Turns the GKG scored lake (GDELT's per-article tone) into causal daily features at a set
of target times (a label's ``t0``). The single leakage rule is identical to the s2
sentiment aggregator: a feature for target time ``t`` uses only events whose
``observed_at`` (GDELT first-seen) falls in the half-open window ``(t - window, t]``.

This module is **deliberately self-contained** — it does NOT import the s2 aggregation
internals — so the Phase-19/20 ``s2`` feature path stays byte-for-byte unchanged. Columns
are prefixed ``gkgtone_`` (config ``aggregation.column_prefix``) so they never collide with
the ``sent_`` columns if both blocks are used together.

Per window, five fields:
* ``count``     — number of articles in the window (0 when empty).
* ``has_any``   — 1 if any article, else 0.
* ``mean_tone`` — mean GDELT tone in [-1, 1] (null when empty).
* ``tone_std``  — tone dispersion / disagreement (null when empty).
* ``pos_share`` — mean positive-word share (null when empty).

``count``/``has_any`` are 0 on an empty window; the averages are null (so a model can tell
"no news" from a real zero).
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

if TYPE_CHECKING:
    from options_system.sentiment.gkg_config import GkgConfig
    from options_system.sentiment.gkg_lake import GkgLake

_US_PER_MINUTE = 60_000_000
_FIELDS = ("count", "has_any", "mean_tone", "tone_std", "pos_share")
_COUNT_FIELDS = frozenset({"count", "has_any"})

_REQUIRED_COLS = ("content_hash", "observed_at", "sentiment_score", "positive_score")


def gkg_feature_names(cfg: GkgConfig) -> list[str]:
    """Deterministic, ordered list of emitted ``gkgtone_*`` feature names (window-major)."""
    p = cfg.aggregation.column_prefix
    return [f"{p}_{w}_{field}" for w in cfg.aggregation.windows for field in _FIELDS]


def _col_dtype(field: str) -> pl.DataType:
    if field == "has_any":
        return pl.Int8()
    if field == "count":
        return pl.Int32()
    return pl.Float64()


def _normalize(frame: pl.DataFrame) -> pl.DataFrame:
    missing = [c for c in _REQUIRED_COLS if c not in frame.columns]
    if missing:
        raise ValueError(f"gkg scored frame missing columns {missing}")
    dtype = frame.schema["observed_at"]
    if isinstance(dtype, pl.Datetime) and dtype.time_zone is None:
        frame = frame.with_columns(pl.col("observed_at").dt.replace_time_zone("UTC"))
    # Collapse re-emitted articles (same content_hash seen in >1 file) so tone is not
    # double-counted; keep the earliest observed copy (the first-seen instant).
    return frame.sort("observed_at").unique(
        subset=["content_hash"], keep="first", maintain_order=True
    )


def _arrays(frame: pl.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if frame.height == 0:
        z = np.empty(0, np.float64)
        return np.empty(0, np.int64), z, z.copy()
    f = frame.sort("observed_at")
    obs = f["observed_at"].to_numpy().astype("datetime64[us]").astype(np.int64)
    tone = f["sentiment_score"].to_numpy().astype(np.float64)
    pos = f["positive_score"].to_numpy().astype(np.float64)
    return obs, tone, pos


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


def _aggregate(
    obs: np.ndarray, tone: np.ndarray, pos: np.ndarray, t_us: int, delta_us: int
) -> dict[str, float | int | None]:
    """Aggregate the half-open window ``(t - delta, t]`` (side='right' = knowable at t)."""
    hi = int(np.searchsorted(obs, t_us, side="right"))
    lo = int(np.searchsorted(obs, t_us - delta_us, side="right"))
    n = hi - lo
    if n <= 0:
        return {"count": 0, "has_any": 0, "mean_tone": None, "tone_std": None, "pos_share": None}
    tw = tone[lo:hi]
    return {
        "count": n,
        "has_any": 1,
        "mean_tone": float(tw.mean()),
        "tone_std": float(tw.std()),
        "pos_share": float(pos[lo:hi].mean()),
    }


def build_gkg_features_for_times(
    scored_frame: pl.DataFrame,
    target_times: pl.Series | Sequence[datetime] | np.ndarray,
    cfg: GkgConfig,
) -> pl.DataFrame:
    """Aggregate GKG tone features at each target time (point-in-time, no leakage).

    One row per target time (input order) with a ``target_time`` key, every column from
    :func:`gkg_feature_names`, and a ``gkg_feature_version`` stamp.
    """
    names = gkg_feature_names(cfg)
    prefix = cfg.aggregation.column_prefix
    out_schema: dict[str, pl.DataType] = {"target_time": pl.Datetime("us", "UTC")}
    for w in cfg.aggregation.windows:
        for field in _FIELDS:
            out_schema[f"{prefix}_{w}_{field}"] = _col_dtype(field)
    out_schema["gkg_feature_version"] = pl.Utf8()

    t_us = _target_times_us(target_times)
    if t_us.size == 0:
        return pl.DataFrame(schema=out_schema)

    obs, tone, pos = _arrays(_normalize(scored_frame))
    cols: dict[str, list[float | int | None]] = {n: [] for n in names}
    windows = list(cfg.aggregation.windows.items())
    for ti in t_us.tolist():
        for wname, minutes in windows:
            agg = _aggregate(obs, tone, pos, int(ti), minutes * _US_PER_MINUTE)
            for field in _FIELDS:
                cols[f"{prefix}_{wname}_{field}"].append(agg[field])

    data: dict[str, object] = {"target_time": t_us.astype("datetime64[us]")}
    data.update(cols)
    data["gkg_feature_version"] = [cfg.aggregation.feature_version] * int(t_us.size)
    return pl.DataFrame(data, schema=out_schema)


def attach_gkg_asof(
    labels_or_events: pl.DataFrame,
    scored_frame: pl.DataFrame,
    cfg: GkgConfig,
    *,
    time_col: str = "t0",
) -> pl.DataFrame:
    """Attach point-in-time ``gkgtone_*`` features onto ``labels_or_events`` by ``time_col``.

    Purely point-in-time: row ``r`` uses only GKG events with ``observed_at <= r[time_col]``.
    Every input row is preserved; features align positionally.
    """
    if time_col not in labels_or_events.columns:
        raise ValueError(f"attach_gkg_asof: time column {time_col!r} not in frame")
    feats = build_gkg_features_for_times(scored_frame, labels_or_events[time_col], cfg)
    feats = feats.drop("target_time")
    return labels_or_events.hstack(feats)


def read_gkg_scores(lake: GkgLake | None = None) -> pl.DataFrame:
    """Read the GKG scored lake (offline — no network, no scoring)."""
    if lake is None:
        from options_system.sentiment.gkg_lake import GkgLake

        lake = GkgLake()
    return lake.read_scored()
