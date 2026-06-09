"""Unit tests for the pure micro-label QA summarizer (gather_label_health).

Synthetic frames only — no lake, no network. Validates that the read-only health
gatherer reuses label_qa correctly and adds version / t0-span / null-inf checks.
"""

from __future__ import annotations

import math
from datetime import UTC, date, datetime, timedelta

import polars as pl

from options_system.observability.micro_label_health import gather_label_health

_PERSIST = {
    "t0": pl.Datetime("us", "UTC"),
    "t1": pl.Datetime("us", "UTC"),
    "symbol": pl.Utf8,
    "contract_id": pl.Utf8,
    "session_date": pl.Date,
    "label": pl.Int8,
    "barrier_touched": pl.Utf8,
    "ret_t1": pl.Float64,
    "sigma": pl.Float64,
    "n_bars": pl.Int32,
    "resolved_at_close": pl.Boolean,
    "uniqueness_weight": pl.Float64,
    "sample_weight": pl.Float64,
    "micro_label_version": pl.Utf8,
    "ts_ingest": pl.Datetime("us", "UTC"),
}

_T0 = datetime(2026, 2, 17, 15, 0, tzinfo=UTC)


def _frame(
    *,
    labels: list[int],
    uniq: list[float],
    days: list[date],
    barriers: list[str],
    at_close: list[bool],
    sigma: list[float] | None = None,
    version: str = "ml1",
) -> pl.DataFrame:
    n = len(labels)
    t0 = [_T0 + timedelta(hours=i) for i in range(n)]
    t1 = [t + timedelta(minutes=30) for t in t0]
    return pl.DataFrame(
        {
            "t0": t0,
            "t1": t1,
            "symbol": ["ES"] * n,
            "contract_id": ["ES.v.0"] * n,
            "session_date": days,
            "label": labels,
            "barrier_touched": barriers,
            "ret_t1": [0.001 * (i + 1) for i in range(n)],
            "sigma": sigma if sigma is not None else [0.002] * n,
            "n_bars": [10 + i for i in range(n)],
            "resolved_at_close": at_close,
            "uniqueness_weight": uniq,
            "sample_weight": [u / (sum(uniq) / n) for u in uniq],
            "micro_label_version": [version] * n,
            "ts_ingest": [_T0] * n,
        },
        schema=_PERSIST,
    )


def test_gather_basic_metrics():
    d0, d1 = date(2026, 2, 17), date(2026, 2, 18)
    f = _frame(
        labels=[1, -1, 0, 1],
        uniq=[1.0, 0.5, 0.5, 1.0],  # effective_n = 3.0 over 2 days -> 1.5/day
        days=[d0, d0, d1, d1],
        barriers=["upper", "lower", "vertical", "close"],
        at_close=[False, False, False, True],
    )
    (info,) = gather_label_health({"ES": f})
    assert info["symbol"] == "ES"
    assert info["labels"] == 4
    qa = info["qa"]
    assert math.isclose(qa["effective_n"], 3.0, rel_tol=1e-9)
    assert qa["n_session_days"] == 2
    assert math.isclose(qa["effective_n_per_day"], 1.5, rel_tol=1e-9)
    assert math.isclose(qa["label_balance"]["pos"], 0.5)
    assert math.isclose(qa["label_balance"]["neg"], 0.25)
    assert math.isclose(qa["label_balance"]["zero"], 0.25)
    assert qa["barrier_touched"] == {"upper": 1, "lower": 1, "vertical": 1, "close": 1}
    assert math.isclose(qa["frac_resolved_at_close"], 0.25)
    assert math.isclose(qa["hold_minutes"]["median"], 30.0, rel_tol=1e-6)
    assert info["micro_label_versions"] == ["ml1"]
    assert info["t0_first"] == _T0
    assert info["null_inf_counts"] == {}


def test_gather_flags_null_and_inf():
    d0 = date(2026, 2, 17)
    f = _frame(
        labels=[1, -1, 1],
        uniq=[1.0, 1.0, 1.0],
        days=[d0, d0, d0],
        barriers=["upper", "lower", "vertical"],
        at_close=[False, False, False],
        sigma=[0.002, float("nan"), float("inf")],  # one nan + one inf in a core column
    )
    (info,) = gather_label_health({"ES": f})
    assert info["null_inf_counts"].get("sigma") == 2


def test_gather_empty_symbol():
    (info,) = gather_label_health({"NQ": pl.DataFrame()})
    assert info["symbol"] == "NQ"
    assert info["labels"] == 0
    assert "qa" not in info


def test_gather_multiple_versions_detected():
    d0 = date(2026, 2, 17)
    a = _frame(
        labels=[1], uniq=[1.0], days=[d0], barriers=["upper"], at_close=[False], version="ml1"
    )
    b = _frame(
        labels=[-1], uniq=[1.0], days=[d0], barriers=["lower"], at_close=[False], version="ml0"
    )
    (info,) = gather_label_health({"ES": pl.concat([a, b])})
    assert info["micro_label_versions"] == ["ml0", "ml1"]
