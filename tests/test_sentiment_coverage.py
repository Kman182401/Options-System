"""Offline sentiment coverage CLI tests (fixture-only, never touches the network)."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from options_system.sentiment import coverage as cov_cli
from options_system.sentiment.config import SentimentConfig
from options_system.sentiment.coverage import (
    _labels_from_fixture,
    _scored_from_fixture,
    build_coverage_report,
    main,
)
from options_system.sentiment.features import normalize_scored_events, sentiment_feature_names
from options_system.sentiment.lake import _SCORED_SCHEMA

FIX = Path(__file__).parent / "fixtures" / "sentiment"
SCORED = str(FIX / "scored_events_pit.json")
MICRO = str(FIX / "micro_labels_for_join.json")
DAILY = str(FIX / "daily_labels_for_join.json")
CFG = SentimentConfig.load()


# --- 9a. coverage report on a fixture ---------------------------------------- #


def test_build_coverage_report_on_fixture():
    scored = _scored_from_fixture(SCORED)
    labels = _labels_from_fixture(MICRO)
    rep = build_coverage_report(labels, scored, CFG, label_type="micro")
    assert rep["label_rows"] == 2
    assert rep["sentiment_rows"] == 5
    assert rep["rows_with_any_sentiment"] == 1  # only the 2026-02-03 label has prior sentiment
    assert abs(rep["coverage_rate"] - 0.5) < 1e-9
    assert rep["events_used"] == 3  # h1, h2, h3 (non-degraded, inside the 1d window)
    assert rep["degraded_count"] == 1  # h5
    assert rep["duplicate_count"] == 0
    assert rep["feature_columns_stable"] is True
    assert rep["feature_count"] == len(sentiment_feature_names(CFG))
    assert rep["coverage_by_window"]["15m"] == 1
    assert rep["coverage_by_source"]["gdelt"] == 1


def test_cli_runs_on_fixture_exit_zero(capsys):
    rc = main(
        ["--label-type", "micro", "--fixture", SCORED, "--label-fixture", MICRO, "--no-write"]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "label_rows=2" in out
    assert "coverage_rate=50.0%" in out


def test_cli_daily_label_type(capsys):
    rc = main(
        ["--label-type", "daily", "--fixture", SCORED, "--label-fixture", DAILY, "--no-write"]
    )
    assert rc == 0
    assert "label_type=daily" in capsys.readouterr().out


# --- 9b. no local data -> exit 0, coverage 0% -------------------------------- #


def test_cli_no_local_data_exits_zero(monkeypatch, capsys):
    empty_scored = normalize_scored_events(pl.DataFrame(schema=_SCORED_SCHEMA))
    monkeypatch.setattr(cov_cli, "_load_scored", lambda _args: empty_scored)
    monkeypatch.setattr(cov_cli, "_load_labels", lambda _args, _cfg: pl.DataFrame())
    rc = main(["--label-type", "micro"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "0% coverage" in out
    assert "label_rows=0" in out


# --- 9c / 10. CLI never touches the network ---------------------------------- #


def test_cli_does_not_touch_network(monkeypatch):
    import urllib.request

    from options_system.sentiment import gdelt, sec_edgar

    def _boom(*_a, **_k):
        raise AssertionError("network access attempted by the coverage CLI")

    monkeypatch.setattr(urllib.request, "urlopen", _boom)
    monkeypatch.setattr(gdelt, "fetch_artlist", _boom)
    monkeypatch.setattr(sec_edgar, "fetch_submissions", _boom)
    rc = main(
        ["--label-type", "micro", "--fixture", SCORED, "--label-fixture", MICRO, "--no-write"]
    )
    assert rc == 0


# --- output-json honors --no-write ------------------------------------------- #


def test_output_json_written_only_without_no_write(tmp_path):
    out_path = tmp_path / "report.json"
    main(["--fixture", SCORED, "--label-fixture", MICRO, "--output-json", str(out_path)])
    assert out_path.exists()

    skipped = tmp_path / "skipped.json"
    main(
        ["--fixture", SCORED, "--label-fixture", MICRO, "--no-write", "--output-json", str(skipped)]
    )
    assert not skipped.exists()


# --- Phase 18: archive-region split ------------------------------------------ #


def test_archive_region_split_supported_vs_unsupported():
    from datetime import UTC, datetime

    scored = _scored_from_fixture(SCORED)
    labels = _labels_from_fixture(MICRO)
    cutoff = datetime(2026, 2, 1, tzinfo=UTC)
    rep = build_coverage_report(labels, scored, CFG, label_type="micro", archive_cutoff=cutoff)
    assert rep["archive_cutoff"] == "2026-02-01T00:00:00+00:00"
    sup = rep["by_archive_region"]["supported"]
    unsup = rep["by_archive_region"]["unsupported_archive"]
    # 2026-02-03 label (has prior sentiment) is at/after the cutoff; 2026-01-01 is before
    assert sup["label_rows"] == 1 and sup["rows_with_any_sentiment"] == 1
    assert abs(sup["coverage_rate"] - 1.0) < 1e-9
    assert unsup["label_rows"] == 1 and unsup["rows_with_any_sentiment"] == 0
    assert abs(unsup["coverage_rate"] - 0.0) < 1e-9
    # regions partition the pooled numbers
    assert sup["label_rows"] + unsup["label_rows"] == rep["label_rows"]
    assert (
        sup["rows_with_any_sentiment"] + unsup["rows_with_any_sentiment"]
        == rep["rows_with_any_sentiment"]
    )
    for window, rate in sup["coverage_by_window_rate"].items():
        assert 0.0 <= rate <= 1.0
        assert sup["coverage_by_window"][window] in (0, 1)


def test_archive_region_split_absent_without_cutoff():
    scored = _scored_from_fixture(SCORED)
    labels = _labels_from_fixture(MICRO)
    rep = build_coverage_report(labels, scored, CFG, label_type="micro")
    assert rep["archive_cutoff"] is None
    assert rep["by_archive_region"] is None


def test_cli_archive_cutoff_flag(capsys):
    rc = main(
        [
            "--label-type",
            "micro",
            "--fixture",
            SCORED,
            "--label-fixture",
            MICRO,
            "--no-write",
            "--archive-cutoff",
            "2026-02-01",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "region=supported" in out
    assert "region=unsupported_archive" in out
