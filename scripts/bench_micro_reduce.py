"""Local, synthetic, ZERO-CREDIT benchmark for the MBP-1 parallel day reducer.

Builds synthetic per-(symbol, session-day) work units in memory and times the
serial vs parallel reduction paths, reporting seconds, speedup, worker count, and
peak row counts. No Databento, no network, no API key — nothing here spends credits
or runs the large pull. Not a pytest (lives under scripts/), so it never gates CI.

Run:

    uv run python scripts/bench_micro_reduce.py --units 16 --bars 800 --workers 4
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

# Make the repo root importable so `config` resolves under `python scripts/...`
# (and, since multiprocessing 'spawn' propagates sys.path, in the workers too).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from options_system.microstructure.bars import BookEvent  # noqa: E402
from options_system.microstructure.config import Instrument, MicrostructureConfig  # noqa: E402

_BASE = datetime(2025, 6, 2, 14, 0, 0, tzinfo=UTC)
_BASE_NS = int(_BASE.timestamp()) * 1_000_000_000
_DAY_NS = 86_400 * 1_000_000_000

_INST = Instrument(
    symbol="B",
    continuous_symbol="B.v.0",
    exec_symbol="b",
    multiplier=1.0,
    tick_size=0.25,
    dollar_threshold=5000.0,
)


def _stream(n_bars: int, day: int) -> tuple[BookEvent, ...]:
    """A comoving stream of ``n_bars`` threshold-closing bars on session ``day``."""
    out: list[BookEvent] = []
    mid = 100.0
    t = 0.0
    for i in range(n_bars):
        d = 1 if i % 2 == 0 else -1
        ts0 = _BASE_NS + day * _DAY_NS + int(round(t * 1e9))
        bid, ask = mid - 0.125, mid + 0.125
        out.append(BookEvent(ts0, 1, False, float("nan"), 0.0, 0, bid, 10.0, ask, 10.0))
        t += 0.001
        ts1 = _BASE_NS + day * _DAY_NS + int(round(t * 1e9))
        bid2, ask2 = bid + 0.25 * d, ask + 0.25 * d
        out.append(BookEvent(ts1, 1, False, float("nan"), 0.0, 0, bid2, 10.0, ask2, 10.0))
        t += 0.001
        ts2 = _BASE_NS + day * _DAY_NS + int(round(t * 1e9))
        out.append(BookEvent(ts2, 1, True, 100.0, 50.0, d, bid2, 10.0, ask2, 10.0))
        t += 0.001
        mid = (bid2 + ask2) / 2
    return tuple(out)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--units", type=int, default=12, help="number of (symbol,day) work units")
    p.add_argument("--bars", type=int, default=600, help="dollar bars per unit")
    p.add_argument("--workers", type=int, default=4, help="parallel worker processes")
    args = p.parse_args()

    # Import here so the heavy module load isn't timed.
    from polars.testing import assert_frame_equal

    from options_system.microstructure.ingest import DayWorkUnit, reduce_units

    cfg = MicrostructureConfig.load()
    base = _BASE.date()
    units = [
        DayWorkUnit(
            symbol="B",
            session_date=base + timedelta(days=i),
            instrument=_INST,
            cfg=cfg,
            events=_stream(args.bars, day=i),
        )
        for i in range(args.units)
    ]
    total_events = sum(len(u.events or ()) for u in units)

    t0 = time.perf_counter()
    serial = reduce_units(units, workers=1)
    t_serial = time.perf_counter() - t0

    t0 = time.perf_counter()
    parallel = reduce_units(units, workers=args.workers)
    t_parallel = time.perf_counter() - t0

    # Correctness alongside the timing: byte-identical or the benchmark is meaningless.
    for a, b in zip(serial, parallel, strict=True):
        assert_frame_equal(a, b, check_exact=True, check_dtypes=True)

    peak_rows = max(f.height for f in serial)
    total_rows = sum(f.height for f in serial)
    speedup = (t_serial / t_parallel) if t_parallel > 0 else float("nan")

    print(f"units={args.units}  bars/unit={args.bars}  events={total_events:,}")
    print(f"rows: total={total_rows:,}  peak/unit={peak_rows:,}")
    print(f"serial:   {t_serial:7.3f}s  (workers=1)")
    print(f"parallel: {t_parallel:7.3f}s  (workers={args.workers})")
    print(f"speedup:  {speedup:6.2f}x   [output verified bit-identical]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
