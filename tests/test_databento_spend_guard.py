"""The fail-closed Databento billing guard (2026-06-09 incident).

Proves that real (billable) downloads are blocked by default and only proceed when
the operator explicitly attests via the environment — so the code can never again
charge a card without a deliberate, per-run opt-in.
"""

from __future__ import annotations

from datetime import date

import pytest

from options_system.common.databento_guard import (
    SPEND_ENV,
    DatabentoSpendNotAuthorized,
    assert_spend_authorized,
    spend_authorized,
)
from options_system.microstructure import ingest
from options_system.microstructure.config import MicrostructureConfig


def test_blocked_by_default(monkeypatch):
    monkeypatch.delenv(SPEND_ENV, raising=False)
    assert spend_authorized() is False
    with pytest.raises(DatabentoSpendNotAuthorized):
        assert_spend_authorized("unit test")


@pytest.mark.parametrize("val", ["1", "true", "YES", "on"])
def test_authorized_when_env_set(monkeypatch, val):
    monkeypatch.setenv(SPEND_ENV, val)
    assert spend_authorized() is True
    assert_spend_authorized("unit test")  # does not raise


@pytest.mark.parametrize("val", ["", "0", "false", "no", "nope"])
def test_not_authorized_for_falsey_values(monkeypatch, val):
    monkeypatch.setenv(SPEND_ENV, val)
    assert spend_authorized() is False
    with pytest.raises(DatabentoSpendNotAuthorized):
        assert_spend_authorized("unit test")


def test_run_ingest_refuses_before_any_network_when_unauthorized(monkeypatch):
    # run_ingest must raise at its choke point BEFORE constructing a Databento client
    # or touching the network. A dummy api_key proves no network call is reached.
    monkeypatch.delenv(SPEND_ENV, raising=False)
    cfg = MicrostructureConfig.load()
    with pytest.raises(DatabentoSpendNotAuthorized):
        ingest.run_ingest(
            cfg,
            ["ES"],
            date(2026, 1, 26),
            date(2026, 1, 27),
            api_key="dummy-not-used",
            cap=1.0,
            workers=1,
        )
