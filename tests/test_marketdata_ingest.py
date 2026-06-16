"""Market-data ingest gating + offline happy path — no real network."""

from __future__ import annotations

import urllib.request
from datetime import date
from functools import partial
from types import SimpleNamespace

import polars as pl
import pytest
from pydantic import SecretStr

from options_system.common.external_data_policy import ExternalAccessNotAuthorized
from options_system.marketdata import ingest as ingest_mod
from options_system.marketdata.config import MarketDataConfig
from options_system.marketdata.lake import MarketDailyLake


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    def _boom(*a, **k):
        raise AssertionError("network call attempted in an offline test")

    monkeypatch.setattr(urllib.request, "urlopen", _boom)


def test_blocked_without_allow_network():
    cfg = MarketDataConfig.load()
    with pytest.raises(ExternalAccessNotAuthorized):
        ingest_mod.ingest(cfg, settings=SimpleNamespace(fred_api_key=None), allow_network=False)


def test_no_key_is_noop():
    cfg = MarketDataConfig.load()
    out = ingest_mod.ingest(cfg, settings=SimpleNamespace(fred_api_key=None), allow_network=True)
    assert out == {}


def test_offline_happy_path_writes(tmp_path, monkeypatch):
    cfg = MarketDataConfig.load()

    def fake_fetch(series_id, api_key, *, observation_start, observation_end=None):
        return pl.DataFrame(
            {"obs_date": [date(2024, 1, 2), date(2024, 1, 3)], "value": [10.0, 11.0]}
        )

    monkeypatch.setattr(ingest_mod, "fetch_daily_series", fake_fetch)
    monkeypatch.setattr(ingest_mod, "MarketDailyLake", partial(MarketDailyLake, root=tmp_path))

    out = ingest_mod.ingest(
        cfg,
        settings=SimpleNamespace(fred_api_key=SecretStr("dummy")),
        allow_network=True,
        series_ids=["VIXCLS"],
    )
    assert out["VIXCLS"] == 2 and out["_written"] == 2
    assert MarketDailyLake(root=tmp_path).read(series_ids=["VIXCLS"]).height == 2
