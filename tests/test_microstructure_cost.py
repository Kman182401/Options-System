"""Budget-cap logic: the guard aborts over-cap pulls; estimate_cost sums chunks.

Uses a stubbed Databento client (no network, no credits) so the cost guard is
exercised deterministically.
"""

from __future__ import annotations

from datetime import date

import pytest

from options_system.microstructure.config import MicrostructureConfig
from options_system.microstructure.ingest import (
    BudgetExceededError,
    _trading_days,
    check_budget,
    estimate_cost,
)

CFG = MicrostructureConfig.load()


def test_check_budget_passes_under_cap():
    check_budget(running_usd=10.0, chunk_usd=5.0, cap_usd=40.0)  # 15 <= 40 -> ok


def test_check_budget_aborts_over_cap():
    with pytest.raises(BudgetExceededError):
        check_budget(running_usd=38.0, chunk_usd=5.0, cap_usd=40.0)  # 43 > 40


def test_check_budget_boundary_is_allowed():
    check_budget(running_usd=35.0, chunk_usd=5.0, cap_usd=40.0)  # exactly 40 -> ok


class _StubMeta:
    def __init__(self, per_chunk: float) -> None:
        self.per_chunk = per_chunk
        self.calls = 0

    def get_cost(self, **_kw) -> float:
        self.calls += 1
        return self.per_chunk


class _StubClient:
    def __init__(self, per_chunk: float) -> None:
        self.metadata = _StubMeta(per_chunk)


def test_estimate_cost_sums_per_day_chunks():
    start, end = date(2026, 3, 2), date(2026, 3, 7)
    n_days = len(_trading_days(start, end))
    assert n_days > 0
    client = _StubClient(per_chunk=0.40)
    est = estimate_cost(client, CFG, ["ES", "NQ"], start, end)
    assert est["ES"] == pytest.approx(0.40 * n_days)
    assert est["NQ"] == pytest.approx(0.40 * n_days)
    # one get_cost call per (symbol, day)
    assert client.metadata.calls == 2 * n_days


def test_estimate_then_cap_decision_with_stub():
    start, end = date(2026, 3, 2), date(2026, 3, 7)
    client = _StubClient(per_chunk=10.0)  # deliberately expensive
    total = sum(estimate_cost(client, CFG, ["ES", "NQ"], start, end).values())
    assert total > CFG.databento_budget_usd_cap  # the planned pull is over cap...
    with pytest.raises(BudgetExceededError):  # ...so the guard would abort
        check_budget(running_usd=0.0, chunk_usd=total, cap_usd=CFG.databento_budget_usd_cap)
