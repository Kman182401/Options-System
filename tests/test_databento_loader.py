"""Tests for the Databento loader scaffold (data/databento_loader.py)."""

from __future__ import annotations

from options_system.data.databento_loader import main


def test_no_key_is_noop(monkeypatch, capsys):
    monkeypatch.delenv("OPTIONS_DATABENTO_API_KEY", raising=False)
    rc = main([])
    out = capsys.readouterr().out
    assert rc == 0
    assert "no-op" in out.lower()
    assert "No network call" in out


def test_no_key_noop_even_with_dates(monkeypatch, capsys):
    monkeypatch.delenv("OPTIONS_DATABENTO_API_KEY", raising=False)
    rc = main(["--start", "2026-01-01", "--end", "2026-02-01"])
    assert rc == 0
    assert "disabled" in capsys.readouterr().out.lower()
