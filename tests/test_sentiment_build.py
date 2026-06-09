"""Build CLI: fail-closed access decision + offline fixture run (never networks)."""

from __future__ import annotations

from pathlib import Path

import pytest

from options_system.common.external_data_policy import ExternalAccessNotAuthorized
from options_system.sentiment.build import decide_access, main

_FIX = Path(__file__).parent / "fixtures" / "sentiment" / "gdelt_fed.json"


def test_decide_access_paid_blocked():
    with pytest.raises(ExternalAccessNotAuthorized):
        decide_access("finnhub", allow_network=True, fixture=None, dry_run=False)


def test_decide_access_unknown_blocked():
    with pytest.raises(ExternalAccessNotAuthorized):
        decide_access("mystery_api", allow_network=True, fixture=None, dry_run=False)


def test_decide_access_network_off_by_default():
    plan = decide_access("gdelt", allow_network=False, fixture=None, dry_run=False)
    assert plan.mode == "dry_run"
    assert plan.network is False


def test_decide_access_network_requires_explicit_flag():
    plan = decide_access("gdelt", allow_network=True, fixture=None, dry_run=False)
    assert plan.mode == "network"
    assert plan.network is True


def test_decide_access_fixture_is_offline():
    plan = decide_access("gdelt", allow_network=False, fixture="x.json", dry_run=False)
    assert plan.mode == "fixture"
    assert plan.network is False


def test_cli_fixture_dry_run_no_network(capsys):
    rc = main(["--source", "gdelt", "--fixture", str(_FIX), "--score", "--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "mode=fixture" in out
    assert "network=False" in out
    assert "not written" in out  # dry-run never writes


def test_cli_rejects_paid_source(capsys):
    rc = main(["--source", "finnhub", "--allow-network"])
    assert rc == 2
    assert "BLOCKED" in capsys.readouterr().out


def test_cli_rejects_unknown_source(capsys):
    rc = main(["--source", "some_unknown_feed", "--allow-network"])
    assert rc == 2
    assert "BLOCKED" in capsys.readouterr().out


def test_cli_default_source_does_not_network(capsys):
    # Default source is the offline 'fixture' pseudo-source; with no --fixture it must
    # refuse cleanly rather than touch the network.
    rc = main([])
    assert rc == 2  # needs --fixture for the offline fixture source
    out = capsys.readouterr().out
    assert "network=False" in out
