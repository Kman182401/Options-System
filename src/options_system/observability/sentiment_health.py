"""Sentiment lake health / QA — read-only, pure summary over already-loaded frames.

:func:`gather_sentiment_health` takes the raw (and optional scored) frames plus the
declared source policy and the network-usage flag, and returns a plain dict: row
counts by source and topic, the point-in-time timestamp ranges, the duplicate rate,
missing-timestamp counts, the scored sentiment distribution, the policy status of each
source seen, and whether any network was used. It performs no I/O and never fetches —
the gathering logic is unit-tested; any CLI/Streamlit view is a thin wrapper.
"""

from __future__ import annotations

from typing import Any

import polars as pl

from options_system.common.external_data_policy import classify

_RAW_TS_COLS = ("published_at", "observed_at", "ingested_at")


def _f(x: Any) -> float:
    """Coerce a polars scalar aggregate (widened in the stubs) to float."""
    return float(x)


def _minmax(df: pl.DataFrame, col: str) -> dict[str, Any]:
    if df.is_empty() or col not in df.columns:
        return {"min": None, "max": None}
    sub = df.select(col).drop_nulls()
    if sub.is_empty():
        return {"min": None, "max": None}
    return {"min": str(sub[col].min()), "max": str(sub[col].max())}


def gather_sentiment_health(
    raw: pl.DataFrame,
    scored: pl.DataFrame | None = None,
    *,
    source_policy: dict[str, str] | None = None,
    network_used: bool = False,
) -> dict:
    """Compute a QA summary over the raw (+ optional scored) sentiment frames (pure)."""
    info: dict[str, Any] = {
        "rows": raw.height,
        "network_used": network_used,
    }

    if raw.is_empty():
        info.update(
            {
                "rows_by_source": {},
                "rows_by_topic": {},
                "published_at": {"min": None, "max": None},
                "observed_at": {"min": None, "max": None},
                "duplicate_rate": 0.0,
                "missing_timestamp_count": 0,
                "degraded_rows": 0,
                "source_policy_status": {},
                "scored": None,
            }
        )
        return info

    rows_by_source = {
        r[0]: int(r[1]) for r in raw.group_by("source").len().sort("source").iter_rows()
    }
    rows_by_topic = {
        r[0]: int(r[1]) for r in raw.group_by("query_topic").len().sort("query_topic").iter_rows()
    }

    n = raw.height
    n_unique = raw.select("content_hash").n_unique()
    duplicate_rate = float(n - n_unique) / float(n) if n else 0.0

    missing = 0
    for c in _RAW_TS_COLS:
        if c in raw.columns:
            missing += int(raw.select(pl.col(c).is_null().sum()).item())

    degraded = int(raw.select(pl.col("degraded").sum()).item()) if "degraded" in raw.columns else 0

    # Policy status for every source present; fall back to the authoritative code
    # registry when the caller did not pass an explicit mapping.
    seen_sources = list(rows_by_source)
    policy_status: dict[str, str] = {}
    for s in seen_sources:
        if source_policy and s in source_policy:
            policy_status[s] = source_policy[s]
        else:
            policy_status[s] = classify(s).value

    info.update(
        {
            "rows_by_source": rows_by_source,
            "rows_by_topic": rows_by_topic,
            "published_at": _minmax(raw, "published_at"),
            "observed_at": _minmax(raw, "observed_at"),
            "duplicate_rate": duplicate_rate,
            "missing_timestamp_count": missing,
            "degraded_rows": degraded,
            "source_policy_status": policy_status,
        }
    )

    if scored is not None and not scored.is_empty() and "sentiment_score" in scored.columns:
        s = scored.select("sentiment_score").drop_nulls()["sentiment_score"]
        info["scored"] = {
            "rows": scored.height,
            "sentiment_score": {
                "mean": _f(s.mean()) if s.len() else None,
                "std": _f(s.std()) if s.len() > 1 else 0.0,
                "min": _f(s.min()) if s.len() else None,
                "max": _f(s.max()) if s.len() else None,
            },
            "models": (
                sorted(scored.select("model_name").unique()["model_name"].to_list())
                if "model_name" in scored.columns
                else []
            ),
        }
    else:
        info["scored"] = None

    return info


def gather_sentiment_feature_health(attached: pl.DataFrame, cfg: Any) -> dict:
    """Read-only summary over a frame with attached ``sent_*`` features (Phase 17, pure).

    ``cfg`` is a :class:`options_system.sentiment.config.SentimentConfig`. Reports feature
    row/column counts, the point-in-time coverage rate, how many rows have a non-null score
    aggregate per window, the latest-observed-age distribution, and per-source / per-topic
    coverage. No I/O, no network.
    """
    from options_system.sentiment.features import _sanitize, sentiment_feature_names

    names = sentiment_feature_names(cfg)
    present = [c for c in names if c in attached.columns]
    wname, _wmin = max(cfg.aggregation.windows.items(), key=lambda kv: kv[1])
    rows = attached.height

    def _rows_with(col: str) -> int:
        return int(attached.filter(pl.col(col) > 0).height) if col in attached.columns else 0

    widest_count = f"sent_{wname}_count"
    rows_with_any = _rows_with(widest_count)

    non_null_score_by_window: dict[str, int] = {}
    for w in cfg.aggregation.windows:
        col = f"sent_{w}_mean_score"
        non_null_score_by_window[w] = (
            int(attached.select(pl.col(col).is_not_null().sum()).item())
            if col in attached.columns
            else 0
        )

    age_col = f"sent_{wname}_latest_age_min"
    if age_col in attached.columns and rows:
        ages = attached.select(age_col).drop_nulls()[age_col]
        age_dist = (
            {
                "min": _f(ages.min()),
                "median": _f(ages.median()),
                "max": _f(ages.max()),
            }
            if ages.len()
            else {"min": None, "median": None, "max": None}
        )
    else:
        age_dist = {"min": None, "median": None, "max": None}

    return {
        "feature_rows": rows,
        "feature_columns": len(present),
        "feature_columns_stable": present == names,
        "coverage_rate": (rows_with_any / rows) if rows else 0.0,
        "non_null_score_by_window": non_null_score_by_window,
        "latest_observed_age_minutes": age_dist,
        "coverage_by_source": {
            s: _rows_with(f"sent_{wname}_source_{_sanitize(s)}_count")
            for s in cfg.aggregation.breakdown_sources
        },
        "coverage_by_topic": {
            t: _rows_with(f"sent_{wname}_topic_{_sanitize(t)}_count")
            for t in cfg.aggregation.breakdown_topics
        },
    }


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - thin CLI report
    import argparse

    from options_system.sentiment.config import SentimentConfig
    from options_system.sentiment.lake import SentimentLake

    p = argparse.ArgumentParser(prog="sentiment_health", description="Sentiment lake QA report")
    p.add_argument("--source", default=None, help="restrict to one source")
    args = p.parse_args(argv)

    cfg = SentimentConfig.load()
    lake = SentimentLake(
        raw_dataset=cfg.storage.raw_dataset, scored_dataset=cfg.storage.scored_dataset
    )
    raw = lake.read_raw(args.source)
    scored = lake.read_scored()
    summary = gather_sentiment_health(
        raw,
        scored,
        source_policy={k: v.value for k, v in cfg.source_policy.items()},
        network_used=False,
    )
    import json

    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
