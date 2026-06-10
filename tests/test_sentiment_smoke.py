"""Phase 16 bounded live-shape smoke: hard caps + fail-closed CLI guards (offline only).

No test here performs a real network fetch. Every case either exercises the pure
bounds helper or a CLI path that is rejected/skipped BEFORE any network call.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from options_system.sentiment import gdelt
from options_system.sentiment.build import (
    SmokeBoundsError,
    enforce_smoke_bounds,
    main,
)

_LIVE_FIX = Path(__file__).parent / "fixtures" / "sentiment" / "gdelt_live_shape.json"
_ING = datetime(2026, 6, 9, 18, 0, tzinfo=UTC)


# --- pure bounds helper ----------------------------------------------------- #


def test_bounds_default_to_cap():
    assert (
        enforce_smoke_bounds(
            "gdelt", max_records=None, start=date(2026, 6, 9), end=date(2026, 6, 10)
        )
        == 5
    )
    assert (
        enforce_smoke_bounds(
            "sec_edgar", max_records=None, start=date(2026, 6, 9), end=date(2026, 6, 10)
        )
        == 2
    )


def test_bounds_reject_over_cap():
    with pytest.raises(SmokeBoundsError):
        enforce_smoke_bounds("gdelt", max_records=6, start=date(2026, 6, 9), end=date(2026, 6, 10))
    with pytest.raises(SmokeBoundsError):
        enforce_smoke_bounds(
            "sec_edgar", max_records=3, start=date(2026, 6, 9), end=date(2026, 6, 10)
        )


def test_bounds_reject_wide_window():
    with pytest.raises(SmokeBoundsError):
        enforce_smoke_bounds("gdelt", max_records=5, start=date(2026, 6, 1), end=date(2026, 6, 10))


def test_bounds_reject_non_free_source():
    for src in ("finnhub", "databento", "mystery_api", "finbert_local"):
        with pytest.raises(SmokeBoundsError):
            enforce_smoke_bounds(src, max_records=1, start=date(2026, 6, 9), end=date(2026, 6, 10))


def test_bounds_accept_at_cap_and_window_edge():
    assert (
        enforce_smoke_bounds("gdelt", max_records=5, start=date(2026, 6, 8), end=date(2026, 6, 10))
        == 5
    )


# --- CLI fail-closed guards (no network reached) ---------------------------- #


def test_cli_smoke_requires_allow_network(capsys):
    rc = main(["--smoke", "--source", "gdelt", "--max-records", "5"])
    assert rc == 2
    out = capsys.readouterr().out
    assert "BLOCKED" in out
    assert "network_used=false" in out


def test_cli_smoke_rejects_over_cap(capsys):
    rc = main(["--smoke", "--source", "gdelt", "--max-records", "50", "--allow-network"])
    assert rc == 2
    assert "network_used=false" in capsys.readouterr().out


def test_cli_smoke_rejects_paid_source(capsys):
    rc = main(["--smoke", "--source", "finnhub", "--allow-network"])
    assert rc == 2
    assert "network_used=false" in capsys.readouterr().out


def test_cli_smoke_rejects_unknown_source(capsys):
    rc = main(["--smoke", "--source", "some_feed", "--allow-network"])
    assert rc == 2
    assert "network_used=false" in capsys.readouterr().out


def test_cli_smoke_rejects_wide_window(capsys):
    rc = main(
        [
            "--smoke",
            "--source",
            "gdelt",
            "--allow-network",
            "--start",
            "2026-01-01",
            "--end",
            "2026-02-01",
            "--max-records",
            "5",
        ]
    )
    assert rc == 2
    assert "network_used=false" in capsys.readouterr().out


def test_cli_smoke_sec_skips_on_placeholder_ua(capsys):
    rc = main(["--smoke", "--source", "sec_edgar", "--allow-network"])
    assert rc == 0  # clean skip, not a failure
    out = capsys.readouterr().out
    assert "SEC smoke skipped" in out
    assert "network_used=false" in out


def test_cli_smoke_gdelt_dry_run_no_network(capsys):
    rc = main(
        ["--smoke", "--source", "gdelt", "--allow-network", "--dry-run", "--max-records", "5"]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "would fetch (NOT executed)" in out
    assert "maxrecords=5" in out
    assert "network_used=false" in out


# --- live-shaped fixture parses unchanged (extra real fields ignored) ------- #


def test_live_shaped_gdelt_parses_to_same_schema():
    payload = json.loads(_LIVE_FIX.read_text())
    events = gdelt.parse_artlist(
        payload, topic="fed", sentiment_feature_version="s1", ingested_at=_ING
    )
    assert len(events) == 2
    for e in events:
        # Extra live fields (url_mobile, socialimage, domain, sourcecountry) are ignored;
        # the PIT schema is intact.
        assert e.source == "gdelt"
        assert e.published_at == e.observed_at  # GDELT seendate -> both
        assert e.observed_at <= e.ingested_at
        assert e.content_hash
        assert not e.degraded
    assert events[0].published_at == datetime(2026, 6, 9, 13, 15, tzinfo=UTC)
