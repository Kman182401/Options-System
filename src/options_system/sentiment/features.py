"""Point-in-time sentiment feature aggregation (Phase 17 — fixture/offline scaffold).

Given scored sentiment events on disk (or an in-memory frame), turn them into causal,
versioned aggregate features attached at a set of *target times* (a label's ``t0``).
The single rule that prevents look-ahead leakage is the **point-in-time gate**: a
feature for target time ``t`` may only use events whose ``observed_at`` (the earliest
moment our system could have known the item — NOT ``published_at``, NOT ``ingested_at``)
falls in the half-open window ``(t - window, t]``.

Why ``observed_at`` and a half-open window:

* ``observed_at`` is the leakage-safe clock. ``published_at`` can be earlier than we
  could have known (we would be using information from before we had it); ``ingested_at``
  can be later (it reflects when we happened to store it, an implementation artefact).
  Only ``observed_at`` answers "could the live system have known this at ``t``?".
* The window is ``(t - window, t]``: an event exactly at ``t`` IS knowable at ``t`` and
  is included; an event exactly at ``t - window`` has just aged out and is excluded.
  This makes the boundaries deterministic and testable.

This module performs **no network access and no scoring** — it only reads/aggregates
already-scored rows. It does not train a model and emits no signal verdict.

Versioning: the emitted frames are stamped ``sentiment_feature_version`` = the
*aggregate* version (``cfg.aggregation.feature_version``, s2 — the first aggregate
version), which is a separate axis from the s1 raw/scored event schema.

Public API
----------
* :func:`sentiment_feature_names` — the deterministic, ordered feature column names.
* :func:`build_sentiment_features_for_times` — features for an explicit set of target times.
* :func:`attach_sentiment_asof` — attach those features onto a label/event frame by a time column.
* :func:`read_sentiment_scores` — read + normalize the scored lake (or a passed frame).
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from datetime import datetime
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import polars as pl

if TYPE_CHECKING:
    from options_system.sentiment.config import Aggregation, SentimentConfig
    from options_system.sentiment.lake import SentimentLake

_US_PER_MINUTE = 60_000_000

# Authoritative aggregate field -> stable, compact column token. Column names are built
# only from these tokens + sanitized config source/topic names, so they are stable and
# never come from arbitrary raw strings.
_FIELD_TOKEN: dict[str, str] = {
    "event_count": "count",
    "degraded_count": "degraded_count",
    "mean_sentiment_score": "mean_score",
    "sum_sentiment_score": "sum_score",
    "mean_positive_score": "mean_pos",
    "mean_negative_score": "mean_neg",
    "mean_neutral_score": "mean_neu",
    "max_abs_sentiment_score": "max_abs_score",
    "latest_observed_age_minutes": "latest_age_min",
}
_HAS_ANY_TOKEN = "has_any"

# Count-like fields default to 0 when a window is empty; every other (score) field is
# null when empty so a model can tell "no events" from a real zero.
_COUNT_FIELDS = frozenset({"event_count", "degraded_count"})

# Columns the aggregation needs on the scored frame (plus optional ``degraded``).
_REQUIRED_SCORED_COLS = (
    "content_hash",
    "source",
    "query_topic",
    "observed_at",
    "positive_score",
    "negative_score",
    "neutral_score",
    "sentiment_score",
    "model_name",
    "scored_at",
)


def _sanitize(name: str) -> str:
    """Lowercase, collapse non-alphanumerics to ``_`` — a safe, stable column token."""
    return re.sub(r"[^0-9a-z]+", "_", name.strip().lower()).strip("_")


def _col_dtype(field: str) -> pl.DataType:
    if field == _HAS_ANY_TOKEN:
        return pl.Int8()
    if field in _COUNT_FIELDS:
        return pl.Int32()
    return pl.Float64()


class _Spec(NamedTuple):
    """One emitted feature column: its name, which channel + window + field produce it."""

    col: str
    kind: str  # "all" | "all_flag" | "source" | "topic"
    key: str | None  # source/topic name for breakdowns, else None
    window: str
    minutes: int
    field: str  # an aggregate field name, or "has_any" for the flag


def _iter_specs(agg: Aggregation) -> Iterable[_Spec]:
    """Yield the feature columns in a deterministic order (window-major).

    For each window: the global group's full fields (+ optional ``has_any`` flag), then
    the per-source breakdowns, then the per-topic breakdowns — each over the reduced
    ``breakdown_fields``. The order here defines :func:`sentiment_feature_names`.
    """
    groups = set(agg.groups)
    for window, minutes in agg.windows.items():
        if "all_sources_all_topics" in groups:
            for field in agg.fields:
                yield _Spec(
                    f"sent_{window}_{_FIELD_TOKEN[field]}", "all", None, window, minutes, field
                )
            if agg.emit_has_any:
                yield _Spec(
                    f"sent_{window}_{_HAS_ANY_TOKEN}",
                    "all_flag",
                    None,
                    window,
                    minutes,
                    _HAS_ANY_TOKEN,
                )
        if "by_source" in groups:
            for src in agg.breakdown_sources:
                for field in agg.breakdown_fields:
                    yield _Spec(
                        f"sent_{window}_source_{_sanitize(src)}_{_FIELD_TOKEN[field]}",
                        "source",
                        src,
                        window,
                        minutes,
                        field,
                    )
        if "by_topic" in groups:
            for tp in agg.breakdown_topics:
                for field in agg.breakdown_fields:
                    yield _Spec(
                        f"sent_{window}_topic_{_sanitize(tp)}_{_FIELD_TOKEN[field]}",
                        "topic",
                        tp,
                        window,
                        minutes,
                        field,
                    )


def sentiment_feature_names(cfg: SentimentConfig) -> list[str]:
    """The deterministic, ordered list of emitted ``sent_*`` feature column names.

    Same config -> same ordered names. Excludes the ``target_time`` key and the
    ``sentiment_feature_version`` stamp (those are added around the features).
    """
    return [spec.col for spec in _iter_specs(cfg.aggregation)]


# --- normalization + reading ------------------------------------------------- #


def _ensure_dt_utc(frame: pl.DataFrame, col: str) -> pl.DataFrame:
    """Coerce ``col`` to ``Datetime('us','UTC')`` (parse strings / set tz as UTC)."""
    if col not in frame.columns:
        return frame
    dtype = frame.schema[col]
    if dtype == pl.Utf8:
        return frame.with_columns(
            pl.col(col).str.to_datetime(time_unit="us").dt.replace_time_zone("UTC")
        )
    if isinstance(dtype, pl.Datetime):
        if dtype.time_zone is None:
            return frame.with_columns(pl.col(col).dt.replace_time_zone("UTC"))
        return frame.with_columns(
            pl.col(col).dt.convert_time_zone("UTC").cast(pl.Datetime("us", "UTC"))
        )
    return frame


def normalize_scored_events(frame: pl.DataFrame) -> pl.DataFrame:
    """Validate + normalize a scored-events frame for aggregation.

    Ensures the required score/timestamp columns exist, timestamps are UTC microseconds,
    and an explicit ``degraded`` boolean column is present (defaults to False — scored
    rows are non-degraded; degraded raw items are simply absent from the scored lake).
    """
    missing = [c for c in _REQUIRED_SCORED_COLS if c not in frame.columns]
    if missing:
        raise ValueError(f"scored events frame missing columns {missing}")
    frame = _ensure_dt_utc(frame, "observed_at")
    frame = _ensure_dt_utc(frame, "scored_at")
    if "published_at" in frame.columns:
        frame = _ensure_dt_utc(frame, "published_at")
    if "degraded" not in frame.columns:
        frame = frame.with_columns(pl.lit(False).alias("degraded"))  # noqa: FBT003
    else:
        frame = frame.with_columns(pl.col("degraded").fill_null(False).cast(pl.Boolean))
    return frame


def dedup_scored_events(frame: pl.DataFrame) -> tuple[pl.DataFrame, int]:
    """Collapse to one row per ``content_hash`` (latest ``scored_at`` wins). Returns
    (deduped frame, duplicates removed).

    Exact ``(content_hash, model_name)`` duplicates collapse, and a headline scored by
    several models collapses to its latest score — so an event is never counted twice.
    """
    if frame.height == 0:
        return frame, 0
    before = frame.height
    deduped = frame.sort("scored_at", descending=True, nulls_last=True).unique(
        subset=["content_hash"], keep="first", maintain_order=True
    )
    return deduped, before - deduped.height


def read_sentiment_scores(lake: SentimentLake | None = None) -> pl.DataFrame:
    """Read the scored sentiment lake (normalized). Offline — no network, no scoring.

    Defaults to the project's ``SentimentLake`` (``Settings().data_dir``); pass a lake
    bound to a tmp root in tests. Returns an empty (typed) normalized frame when the lake
    has no scores yet.
    """
    if lake is None:
        from options_system.sentiment.lake import SentimentLake

        lake = SentimentLake()
    return normalize_scored_events(lake.read_scored())


# --- aggregation ------------------------------------------------------------- #


class _Channel(NamedTuple):
    """Per-channel arrays, sorted ascending by ``observed_at`` (microseconds)."""

    obs: np.ndarray  # int64 microseconds since epoch
    sent: np.ndarray  # float64 sentiment_score
    pos: np.ndarray  # float64 positive_score
    neg: np.ndarray  # float64 negative_score
    neu: np.ndarray  # float64 neutral_score


def _obs_us(frame: pl.DataFrame) -> np.ndarray:
    if frame.height == 0:
        return np.empty(0, dtype=np.int64)
    return frame["observed_at"].sort().to_numpy().astype("datetime64[us]").astype(np.int64)


def _channel(frame: pl.DataFrame) -> _Channel:
    if frame.height == 0:
        z = np.empty(0, dtype=np.float64)
        return _Channel(np.empty(0, dtype=np.int64), z, z.copy(), z.copy(), z.copy())
    f = frame.sort("observed_at")
    obs = f["observed_at"].to_numpy().astype("datetime64[us]").astype(np.int64)
    return _Channel(
        obs,
        f["sentiment_score"].to_numpy().astype(np.float64),
        f["positive_score"].to_numpy().astype(np.float64),
        f["negative_score"].to_numpy().astype(np.float64),
        f["neutral_score"].to_numpy().astype(np.float64),
    )


def _aggregate(ch: _Channel, t_us: int, delta_us: int) -> dict[str, float | int | None]:
    """Aggregate one channel over the half-open window ``(t - delta, t]``.

    ``side="right"`` puts an event exactly at ``t`` inside the window (knowable at ``t``)
    and an event exactly at ``t - delta`` outside it (just aged out).
    """
    hi = int(np.searchsorted(ch.obs, t_us, side="right"))
    lo = int(np.searchsorted(ch.obs, t_us - delta_us, side="right"))
    n = hi - lo
    if n <= 0:
        return {
            "count": 0,
            "has_any": 0,
            "mean_score": None,
            "sum_score": None,
            "mean_pos": None,
            "mean_neg": None,
            "mean_neu": None,
            "max_abs": None,
            "latest_age": None,
        }
    s = ch.sent[lo:hi]
    return {
        "count": n,
        "has_any": 1,
        "mean_score": float(s.mean()),
        "sum_score": float(s.sum()),
        "mean_pos": float(ch.pos[lo:hi].mean()),
        "mean_neg": float(ch.neg[lo:hi].mean()),
        "mean_neu": float(ch.neu[lo:hi].mean()),
        "max_abs": float(np.abs(s).max()),
        "latest_age": float((t_us - int(ch.obs[hi - 1])) / _US_PER_MINUTE),
    }


def _count_in_window(obs: np.ndarray, t_us: int, delta_us: int) -> int:
    hi = int(np.searchsorted(obs, t_us, side="right"))
    lo = int(np.searchsorted(obs, t_us - delta_us, side="right"))
    return max(0, hi - lo)


# aggregate field name -> key in the dict returned by :func:`_aggregate`.
_AGG_KEY = {
    "event_count": "count",
    "mean_sentiment_score": "mean_score",
    "sum_sentiment_score": "sum_score",
    "mean_positive_score": "mean_pos",
    "mean_negative_score": "mean_neg",
    "mean_neutral_score": "mean_neu",
    "max_abs_sentiment_score": "max_abs",
    "latest_observed_age_minutes": "latest_age",
}


def _target_times_us(target_times: pl.Series | Sequence[datetime] | np.ndarray) -> np.ndarray:
    """Target times -> int64 microseconds (UTC), preserving input order."""
    if isinstance(target_times, pl.Series):
        s = target_times
    else:
        s = pl.Series("target_time", list(target_times))
    s = s.cast(pl.Datetime("us")) if s.dtype == pl.Date else s
    dtype = s.dtype
    if isinstance(dtype, pl.Datetime):
        s = (
            s.dt.replace_time_zone("UTC")
            if dtype.time_zone is None
            else s.dt.convert_time_zone("UTC")
        )
    return s.to_numpy().astype("datetime64[us]").astype(np.int64)


def build_sentiment_features_for_times(
    scored_events: pl.DataFrame,
    target_times: pl.Series | Sequence[datetime] | np.ndarray,
    cfg: SentimentConfig,
) -> pl.DataFrame:
    """Aggregate sentiment features at each target time (point-in-time, no leakage).

    Returns one row per target time (in input order) with a ``target_time`` key, every
    ``sent_*`` column from :func:`sentiment_feature_names`, and the
    ``sentiment_feature_version`` stamp (the aggregate version). Missing data is explicit:
    count fields are 0, score aggregates are null, and the ``has_any`` flag is 0.
    """
    agg = cfg.aggregation
    specs = list(_iter_specs(agg))
    feature_version = agg.feature_version
    out_schema: dict[str, pl.DataType] = {"target_time": pl.Datetime("us", "UTC")}
    for spec in specs:
        out_schema[spec.col] = _col_dtype(spec.field)
    out_schema["sentiment_feature_version"] = pl.Utf8()

    t_us = _target_times_us(target_times)
    if t_us.size == 0:
        return pl.DataFrame(schema=out_schema)

    scored = normalize_scored_events(scored_events)
    scored, _ = dedup_scored_events(scored)
    non_degraded = scored.filter(~pl.col("degraded"))
    degraded_obs = _obs_us(scored.filter(pl.col("degraded")))

    all_ch = _channel(non_degraded)
    src_ch = {
        s: _channel(non_degraded.filter(pl.col("source") == s)) for s in agg.breakdown_sources
    }
    topic_ch = {
        t: _channel(non_degraded.filter(pl.col("query_topic") == t)) for t in agg.breakdown_topics
    }

    # Accumulate column-wise so the final frame keeps the declared dtypes (nulls typed).
    columns: dict[str, list[float | int | None]] = {spec.col: [] for spec in specs}
    for ti in t_us.tolist():
        # Cache per (kind/key, window) aggregate dicts so each spec is a cheap lookup.
        cache: dict[tuple[str, str | None, str], dict[str, float | int | None]] = {}
        for spec in specs:
            ck = (spec.kind if spec.kind != "all_flag" else "all", spec.key, spec.window)
            if ck not in cache:
                delta = spec.minutes * _US_PER_MINUTE
                if spec.kind in ("all", "all_flag"):
                    cache[ck] = _aggregate(all_ch, ti, delta)
                elif spec.kind == "source":
                    assert spec.key is not None
                    cache[ck] = _aggregate(src_ch[spec.key], ti, delta)
                else:
                    assert spec.key is not None
                    cache[ck] = _aggregate(topic_ch[spec.key], ti, delta)
            aggs = cache[ck]
            if spec.field == "degraded_count":
                columns[spec.col].append(
                    _count_in_window(degraded_obs, ti, spec.minutes * _US_PER_MINUTE)
                )
            elif spec.field == _HAS_ANY_TOKEN:
                columns[spec.col].append(aggs["has_any"])
            else:
                columns[spec.col].append(aggs[_AGG_KEY[spec.field]])

    data: dict[str, object] = {"target_time": t_us.astype("datetime64[us]")}
    data.update(columns)
    data["sentiment_feature_version"] = [feature_version] * int(t_us.size)
    frame = pl.DataFrame(data, schema=out_schema)
    return frame


def attach_sentiment_asof(
    labels_or_events: pl.DataFrame,
    scored_events: pl.DataFrame,
    cfg: SentimentConfig,
    *,
    time_col: str = "t0",
) -> pl.DataFrame:
    """Attach the point-in-time sentiment features onto ``labels_or_events`` by ``time_col``.

    Purely point-in-time: features at row ``r`` use only events with ``observed_at <=
    r[time_col]``. **Never** reads ``t1``, returns, the label outcome, or any future row.
    Every input row is preserved (missing sentiment is recorded as zero counts / null
    scores, not a dropped sample). Features are aligned positionally to the input rows.
    """
    if time_col not in labels_or_events.columns:
        raise ValueError(f"attach_sentiment_asof: time column {time_col!r} not in frame")
    feats = build_sentiment_features_for_times(scored_events, labels_or_events[time_col], cfg)
    feats = feats.drop("target_time")  # positional align; the join key stays as time_col
    return labels_or_events.hstack(feats)
