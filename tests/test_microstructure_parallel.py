"""Serial-vs-parallel equivalence for the MBP-1 day reducer (Phase 11).

The parallel path (``reduce_units`` over a process pool) is a **behavior-preserving
speedup**: it must produce bit-identical output to the serial reference path on the
same input — same rows, order, columns, dtypes, and exact float values. These tests
prove that with `polars.testing.assert_frame_equal(check_exact=True)` (no float
tolerance), and that the result is independent of worker count and completion order.

Everything here is synthetic, offline, zero-credit: no Databento, no network, no API
key. The work units carry in-memory ``events`` (the offline branch of
``reduce_work_unit``); the only untested-offline bit is the thin DBN-file read, which
is exercised by the existing real-path structure and ``test_microstructure_bars``.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest
from polars.testing import assert_frame_equal

from _micro_helpers import (
    TEST_INST,
    TEST_INST_U,
    comoving_stream,
    ev,
    partial_tail_stream,
    quote_only_then_trades_stream,
    roll_within_day_stream,
    ts,
)
from options_system.microstructure.bars import assemble_features, build_dollar_bars
from options_system.microstructure.config import MicrostructureConfig
from options_system.microstructure.ingest import (
    DayWorkUnit,
    _auto_workers,
    _parse,
    _workers_arg,
    main,
    reduce_units,
    reduce_units_to_frame,
    reduce_work_unit,
)

CFG = MicrostructureConfig.load()


def _sdate(day_off: int) -> date:
    return date(2025, 6, 2) + timedelta(days=day_off)


def _events(u: DayWorkUnit) -> tuple:
    """Narrow the (always-set, in these fixtures) events source for the reducer."""
    assert u.events is not None
    return u.events


def _units() -> list[DayWorkUnit]:
    """A spread of independent (symbol, session-day) units stressing every required
    case: 2 symbols, 4 distinct RTH session days, many bars/day, threshold-closed
    bars, quote-only events, a within-day contract roll, trailing partial bars, and
    deterministic intrabar order-flow continuity (comoving streams)."""
    return [
        DayWorkUnit(
            symbol="T",
            session_date=_sdate(0),
            instrument=TEST_INST,
            cfg=CFG,
            events=tuple(comoving_stream(12)),  # many bars, OFI continuity within session
        ),
        DayWorkUnit(
            symbol="T",
            session_date=_sdate(1),
            instrument=TEST_INST,
            cfg=CFG,
            events=tuple(partial_tail_stream(4, day=1)),  # trailing partial bar
        ),
        DayWorkUnit(
            symbol="U",
            session_date=_sdate(0),
            instrument=TEST_INST_U,
            cfg=CFG,
            events=tuple(quote_only_then_trades_stream(day=0)),  # quote-only + trades
        ),
        DayWorkUnit(
            symbol="U",
            session_date=_sdate(2),
            instrument=TEST_INST_U,
            cfg=CFG,
            events=tuple(roll_within_day_stream(day=2)),  # within-day contract roll
        ),
        DayWorkUnit(
            symbol="T",
            session_date=_sdate(3),
            instrument=TEST_INST,
            cfg=CFG,
            events=tuple(partial_tail_stream(3, day=3)),
        ),
    ]


def _reference(units: list[DayWorkUnit]) -> list:
    """Hand-built serial reference: the unchanged reducer applied per unit, in input
    order — independent of the parallel orchestration under test."""
    return [
        assemble_features(
            build_dollar_bars(_events(u), instrument=u.instrument, session=CFG.session),
            symbol=u.symbol,
            cfg=CFG,
            contract_map=u.contract_map or {},
        )
        for u in units
    ]


# --- A. serial vs parallel exact equivalence -------------------------------- #


def test_serial_parallel_exact_equivalence():
    units = _units()
    serial = reduce_units(units, workers=1)
    parallel = reduce_units(units, workers=2)
    assert len(serial) == len(parallel) == len(units)
    for a, b in zip(serial, parallel, strict=True):
        assert_frame_equal(a, b, check_exact=True, check_dtypes=True, check_column_order=True)


def test_parallel_matches_reference_reducer():
    units = _units()
    reference = _reference(units)
    parallel = reduce_units(units, workers=4)
    for a, b in zip(reference, parallel, strict=True):
        assert_frame_equal(a, b, check_exact=True, check_dtypes=True, check_column_order=True)


def test_reduce_work_unit_matches_reducer_single():
    """The worker on one unit == the raw reducer on the same events, exactly."""
    u = _units()[0]
    direct = assemble_features(
        build_dollar_bars(_events(u), instrument=u.instrument, session=CFG.session),
        symbol=u.symbol,
        cfg=CFG,
    )
    assert_frame_equal(reduce_work_unit(u), direct, check_exact=True, check_dtypes=True)


# --- B. determinism --------------------------------------------------------- #


def test_determinism_repeated_parallel_runs():
    units = _units()
    first = reduce_units(units, workers=3)
    second = reduce_units(units, workers=3)
    for a, b in zip(first, second, strict=True):
        assert_frame_equal(a, b, check_exact=True, check_dtypes=True)
    # and the canonical single-frame view is identical run-to-run
    assert_frame_equal(
        reduce_units_to_frame(units, workers=3),
        reduce_units_to_frame(units, workers=3),
        check_exact=True,
        check_dtypes=True,
    )


# --- C. worker-count invariance --------------------------------------------- #


def test_worker_count_invariance():
    units = _units()
    counts = [1, 2, min(4, len(units))]
    baseline = reduce_units(units, workers=counts[0])
    for w in counts[1:]:
        other = reduce_units(units, workers=w)
        for a, b in zip(baseline, other, strict=True):
            assert_frame_equal(a, b, check_exact=True, check_dtypes=True)


def test_canonical_reassembly_order_independent():
    """Completion order AND input order must not change the canonical concat frame."""
    units = _units()
    shuffled = [units[i] for i in (3, 0, 4, 2, 1)]  # deterministic non-identity permutation
    in_order = reduce_units_to_frame(units, workers=1)
    out_of_order = reduce_units_to_frame(shuffled, workers=4)
    assert_frame_equal(in_order, out_of_order, check_exact=True, check_dtypes=True)


# --- D. boundary safety ----------------------------------------------------- #


def test_boundary_safety_contract_roll_unit():
    """A within-day contract roll is preserved identically under parallel reduction:
    the bar splits at the seam (no bar spans two contracts) and OFI never crosses it.
    """
    unit = DayWorkUnit(
        symbol="U",
        session_date=_sdate(2),
        instrument=TEST_INST_U,
        cfg=CFG,
        events=tuple(roll_within_day_stream(day=2)),
    )
    serial = reduce_work_unit(unit)
    parallel = reduce_units([unit], workers=2)[0]
    assert_frame_equal(serial, parallel, check_exact=True, check_dtypes=True)
    # structural boundary facts (mirrors test_microstructure_bars roll test)
    assert serial["contract_id"].to_list() == ["id1", "id2"]
    assert serial["con_id"].to_list() == [1, 2]
    # contract-2 bar's OFI is computed only from contract-2 transitions (seam severs
    # continuity) -> the first id2 transition has no prior id2 state, so its bar OFI
    # is finite and not contaminated by the id1 book jump (100 -> 200).
    id2_ofi = serial.filter(serial["con_id"] == 2)["ofi_top"].to_list()[0]
    assert abs(id2_ofi) < 1e6, "OFI must not absorb the cross-contract book jump"


def test_session_boundary_severed_across_units():
    """Two session-days of the SAME symbol are independent units; flow never bleeds
    between them — proven by per-unit reduction matching whole-batch reduction."""
    units = [
        DayWorkUnit(
            symbol="T",
            session_date=_sdate(0),
            instrument=TEST_INST,
            cfg=CFG,
            events=tuple(comoving_stream(8)),
        ),
        DayWorkUnit(
            symbol="T",
            session_date=_sdate(1),
            instrument=TEST_INST,
            cfg=CFG,
            events=tuple(partial_tail_stream(3, day=1)),
        ),
    ]
    batch = reduce_units(units, workers=2)
    for u, got in zip(units, batch, strict=True):
        alone = reduce_work_unit(u)
        assert_frame_equal(got, alone, check_exact=True, check_dtypes=True)


# --- partial / empty handling ----------------------------------------------- #


def test_trailing_partial_bar_preserved():
    unit = DayWorkUnit(
        symbol="T",
        session_date=_sdate(1),
        instrument=TEST_INST,
        cfg=CFG,
        events=tuple(partial_tail_stream(4, day=1)),
    )
    serial = reduce_work_unit(unit)
    parallel = reduce_units([unit], workers=2)[0]
    assert_frame_equal(serial, parallel, check_exact=True, check_dtypes=True)
    assert serial["bar_complete"].to_list()[-1] is False  # trailing partial kept


def test_empty_and_mixed_units():
    """A trade-less unit reduces to an empty (schema-bearing) frame; mixing it with a
    real unit leaves the canonical frame equal to the real unit's bars alone."""
    quote_only = [ev(ts(i * 0.001), bid0=100.0 + i * 0.25) for i in range(5)]  # no trades
    empty_unit = DayWorkUnit(
        symbol="T", session_date=_sdate(0), instrument=TEST_INST, cfg=CFG, events=tuple(quote_only)
    )
    real_unit = DayWorkUnit(
        symbol="T",
        session_date=_sdate(0),
        instrument=TEST_INST,
        cfg=CFG,
        events=tuple(comoving_stream(6)),
    )
    frames = reduce_units([empty_unit], workers=2)
    assert frames[0].height == 0
    assert frames[0].columns == assemble_features([], symbol="", cfg=CFG).columns

    mixed = reduce_units_to_frame([empty_unit, real_unit], workers=2)
    only_real = reduce_units_to_frame([real_unit], workers=1)
    assert_frame_equal(mixed, only_real, check_exact=True, check_dtypes=True)

    all_empty = reduce_units_to_frame([empty_unit], workers=2)
    assert all_empty.height == 0


def test_reduce_units_empty_input():
    assert reduce_units([], workers=4) == []


# --- worker-count helper ---------------------------------------------------- #


def test_auto_workers_bounds():
    assert _auto_workers(0) == 1  # floored at 1
    assert _auto_workers(1) == 1  # never exceeds task count
    assert _auto_workers(3) <= 3
    assert _auto_workers(1000) <= 8  # hard cap
    assert _auto_workers(1000) >= 1


def test_day_work_unit_requires_exactly_one_source():
    with pytest.raises(ValueError, match="exactly one"):
        DayWorkUnit(symbol="T", session_date=_sdate(0), instrument=TEST_INST, cfg=CFG)
    with pytest.raises(ValueError, match="exactly one"):
        DayWorkUnit(
            symbol="T",
            session_date=_sdate(0),
            instrument=TEST_INST,
            cfg=CFG,
            dbn_path="/x.dbn",
            events=(),
        )


# --- F. CLI safety ---------------------------------------------------------- #


def test_cli_workers_arg_parsing():
    assert _parse([]).workers == 1  # default preserves serial behavior
    assert _parse(["--workers", "3"]).workers == 3
    assert _parse(["--workers", "auto"]).workers == "auto"
    assert _parse([]).confirm is False  # dry-run by default


def test_cli_workers_arg_rejects_bad_values():
    assert _workers_arg("auto") == "auto"
    assert _workers_arg("2") == 2
    with pytest.raises(Exception, match="must be"):
        _workers_arg("0")
    with pytest.raises(Exception, match="must be"):
        _workers_arg("nope")


def test_cli_dry_run_noop_is_offline_with_workers(monkeypatch, capsys):
    """With no API key the CLI is a no-op (no network) even with --workers; this also
    confirms --confirm is NOT required/used and the flag flows through safely."""
    import options_system.microstructure.ingest as ingest_mod

    monkeypatch.setattr(ingest_mod, "_get_api_key", lambda _settings: None)
    rc = main(["--workers", "4", "--symbols", "ES", "--start", "2026-03-02", "--end", "2026-03-04"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "no-op" in out.lower() or "no network" in out.lower()
