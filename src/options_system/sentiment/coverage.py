"""Read-only, offline sentiment coverage report (Phase 17).

    uv run python -m options_system.sentiment.coverage [flags]

Answers one question without training anything: *given the sentiment scored on disk (or
a fixture), how much of our label set actually has prior point-in-time sentiment
coverage, and is the feature layer well-formed?* It attaches the aggregate features to
the labels (:mod:`options_system.sentiment.join`) and summarizes coverage. It performs
**no network access, no scoring, no model training, and no signal verdict**, and it never
writes to the data lake (``--output-json`` writes only the report, and only when not
``--no-write``).

Flags
-----
--label-type micro|daily   which labels to cover (default: micro)
--symbols ES NQ            symbols to load from the lake (default: config symbols)
--start / --end YYYY-MM-DD  label ``t0`` bounds (default: all)
--fixture PATH             read scored sentiment rows from a local fixture (offline)
--label-fixture PATH       read label rows from a local fixture (offline; for tests)
--no-write                 never write any output file
--output-json PATH         also write the report JSON here (default: print only)

If no scored sentiment and no labels are found, it prints a clean "coverage 0%" report
and exits 0.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from options_system.sentiment.config import SentimentConfig
from options_system.sentiment.features import (
    normalize_scored_events,
    read_sentiment_scores,
    sentiment_feature_names,
)
from options_system.sentiment.join import (
    attach_to_daily_labels,
    attach_to_micro_labels,
    feature_columns_stable,
    null_feature_count,
)

_WIDE_START = datetime(2000, 1, 1, tzinfo=UTC)
_WIDE_END = datetime(2100, 1, 1, tzinfo=UTC)


# --- offline fixture loaders ------------------------------------------------- #


def _scored_from_fixture(path: str) -> pl.DataFrame:
    """Load scored sentiment rows from a local JSON fixture (a list of row dicts)."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"scored fixture {path!r} must be a JSON list of row dicts")
    if not payload:
        return normalize_scored_events(_empty_scored())
    return normalize_scored_events(pl.DataFrame(payload))


def _empty_scored() -> pl.DataFrame:
    from options_system.sentiment.lake import _SCORED_SCHEMA

    return pl.DataFrame(schema=_SCORED_SCHEMA)


def _labels_from_fixture(path: str) -> pl.DataFrame:
    """Load label rows from a local JSON fixture; cast ``t0``/``t1`` to UTC microseconds."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"label fixture {path!r} must be a JSON list of row dicts")
    if not payload:
        return pl.DataFrame()
    frame = pl.DataFrame(payload)
    for col in ("t0", "t1"):
        if col in frame.columns and frame.schema[col] == pl.Utf8:
            frame = frame.with_columns(
                pl.col(col).str.to_datetime(time_unit="us").dt.replace_time_zone("UTC")
            )
    if "session_date" in frame.columns and frame.schema["session_date"] == pl.Utf8:
        frame = frame.with_columns(pl.col("session_date").str.to_date())
    return frame


# --- lake loaders (default path; offline, local only) ------------------------ #


def _load_scored(args: argparse.Namespace) -> pl.DataFrame:
    if args.fixture:
        return _scored_from_fixture(args.fixture)
    return read_sentiment_scores()


def _load_labels(args: argparse.Namespace, cfg: SentimentConfig) -> pl.DataFrame:
    if args.label_fixture:
        return _labels_from_fixture(args.label_fixture)

    start = datetime.fromisoformat(args.start).replace(tzinfo=UTC) if args.start else _WIDE_START
    end = datetime.fromisoformat(args.end).replace(tzinfo=UTC) if args.end else _WIDE_END

    frames: list[pl.DataFrame] = []
    if args.label_type == "micro":
        from options_system.microstructure.config import MicrostructureConfig
        from options_system.microstructure.labels import read_micro_labels

        symbols = args.symbols or MicrostructureConfig.load().symbols()
        for sym in symbols:
            df = read_micro_labels(sym, start, end)
            if df.height:
                frames.append(df)
    else:
        from config.settings import Settings
        from options_system.labeling.build import read_labels

        symbols = args.symbols or Settings().record_symbols
        for sym in symbols:
            df = read_labels(sym, start, end)
            if df.height:
                frames.append(df)
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="diagonal_relaxed")


# --- report ------------------------------------------------------------------ #


def build_coverage_report(
    labels: pl.DataFrame,
    scored_events: pl.DataFrame,
    cfg: SentimentConfig,
    *,
    label_type: str,
) -> dict[str, Any]:
    """Attach features and summarize coverage (pure; no I/O, no network)."""
    time_col = "t0"
    scored_norm = normalize_scored_events(scored_events)

    if labels.height == 0 or time_col not in labels.columns:
        non_degraded = scored_norm.filter(~pl.col("degraded"))
        return {
            "label_type": label_type,
            "label_rows": 0,
            "sentiment_rows": int(scored_norm.height),
            "rows_with_any_sentiment": 0,
            "coverage_rate": 0.0,
            "events_used": 0,
            "duplicate_count": 0,
            "degraded_count": int(scored_norm.filter(pl.col("degraded")).height),
            "null_feature_count": 0,
            "feature_columns_stable": True,
            "feature_count": len(sentiment_feature_names(cfg)),
            "windows": list(cfg.aggregation.windows),
            "coverage_by_window": {w: 0 for w in cfg.aggregation.windows},
            "coverage_by_source": {s: 0 for s in cfg.aggregation.breakdown_sources},
            "coverage_by_topic": {t: 0 for t in cfg.aggregation.breakdown_topics},
            "observed_at": {"min": None, "max": None}
            if non_degraded.height == 0
            else {
                "min": str(non_degraded["observed_at"].min()),
                "max": str(non_degraded["observed_at"].max()),
            },
            "label_time": {"min": None, "max": None},
            "feature_version": cfg.aggregation.feature_version,
        }

    attach = attach_to_micro_labels if label_type == "micro" else attach_to_daily_labels
    attached, cov = attach(labels, scored_events, cfg)

    wname, _ = max(cfg.aggregation.windows.items(), key=lambda kv: kv[1])

    def _rows_with(col: str) -> int:
        return int(attached.filter(pl.col(col) > 0).height) if col in attached.columns else 0

    coverage_by_source = {
        s: _rows_with(f"sent_{wname}_source_{_san(s)}_count")
        for s in cfg.aggregation.breakdown_sources
    }
    coverage_by_topic = {
        t: _rows_with(f"sent_{wname}_topic_{_san(t)}_count")
        for t in cfg.aggregation.breakdown_topics
    }

    return {
        "label_type": label_type,
        "label_rows": cov["rows"],
        "sentiment_rows": cov["scored_rows"],
        "rows_with_any_sentiment": cov["rows_with_any_sentiment"],
        "coverage_rate": cov["coverage_rate"],
        "events_used": cov["events_used"],
        "duplicate_count": cov["duplicate_count"],
        "degraded_count": cov["degraded_count"],
        "null_feature_count": null_feature_count(attached, cfg),
        "feature_columns_stable": feature_columns_stable(attached, cfg),
        "feature_count": len(sentiment_feature_names(cfg)),
        "windows": cov["windows"],
        "coverage_by_window": cov["coverage_by_window"],
        "coverage_by_source": coverage_by_source,
        "coverage_by_topic": coverage_by_topic,
        "observed_at": cov["observed_at"],
        "label_time": cov["label_time"],
        "feature_version": cov["feature_version"],
    }


def _san(name: str) -> str:
    from options_system.sentiment.features import _sanitize

    return _sanitize(name)


def _print_report(report: dict[str, Any]) -> None:
    print(
        f"[sentiment.coverage] label_type={report['label_type']} "
        f"feature_version={report['feature_version']} features={report['feature_count']}"
    )
    print(
        f"  label_rows={report['label_rows']} sentiment_rows={report['sentiment_rows']} "
        f"events_used={report['events_used']} duplicates={report['duplicate_count']} "
        f"degraded={report['degraded_count']}"
    )
    print(
        f"  rows_with_any_sentiment={report['rows_with_any_sentiment']} "
        f"coverage_rate={report['coverage_rate'] * 100:.1f}%"
    )
    print(f"  coverage_by_window={report['coverage_by_window']}")
    print(f"  coverage_by_source={report['coverage_by_source']}")
    print(f"  coverage_by_topic={report['coverage_by_topic']}")
    print(f"  observed_at={report['observed_at']} label_time={report['label_time']}")
    print(
        f"  null_feature_count={report['null_feature_count']} "
        f"feature_columns_stable={report['feature_columns_stable']}"
    )
    if report["label_rows"] == 0 or report["sentiment_rows"] == 0:
        print("  NOTE: no coverage to measure (0% coverage) — no sentiment/labels on disk yet.")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="sentiment.coverage", description="Offline sentiment coverage report"
    )
    p.add_argument("--label-type", choices=("micro", "daily"), default="micro")
    p.add_argument("--symbols", nargs="+", default=None, help="symbols to load (default: config)")
    p.add_argument("--start", help="label t0 lower bound YYYY-MM-DD (UTC)")
    p.add_argument("--end", help="label t0 upper bound YYYY-MM-DD (UTC)")
    p.add_argument("--fixture", default=None, help="scored sentiment fixture (offline)")
    p.add_argument("--label-fixture", default=None, help="label rows fixture (offline)")
    p.add_argument("--no-write", action="store_true", help="never write any output file")
    p.add_argument("--output-json", default=None, help="also write the report JSON here")
    return p


def run(args: argparse.Namespace) -> int:
    cfg = SentimentConfig.load()
    scored = _load_scored(args)
    labels = _load_labels(args, cfg)
    report = build_coverage_report(labels, scored, cfg, label_type=args.label_type)
    _print_report(report)
    if args.output_json and not args.no_write:
        Path(args.output_json).write_text(
            json.dumps(report, indent=2, default=str), encoding="utf-8"
        )
        print(f"  wrote {args.output_json}")
    return 0


def main(argv: list[str] | None = None) -> int:
    return run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
