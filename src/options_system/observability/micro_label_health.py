"""Micro-label health / QA — read-only over the ``micro_labels`` lake.

Per symbol: raw label count, **effective N** (Σ ``uniqueness_weight``) and per RTH
day, label balance (+1 / −1 / 0), barrier-touched distribution, median hold
minutes, resolved-at-close rate, the ``micro_label_version`` values present, the
``t0`` span, and a null/inf check on the core numeric columns.

The metric math is **reused** from :func:`options_system.microstructure.labels.label_qa`
(the same pure summarizer the label builder uses) — this module only loads the
lake, adds version / span / null-inf checks, and prints. The pure gathering
function (:func:`gather_label_health`) is unit-tested on synthetic frames.

It only reads ``data/micro_labels/``. No network, no Databento, no writes.

    uv run python -m options_system.observability.micro_label_health \\
        --symbols ES NQ --start 2026-02-16 --end 2026-06-06
"""

from __future__ import annotations

from datetime import UTC, datetime

import polars as pl

from options_system.microstructure.labels import label_qa

# Core columns we assert are clean (null/inf would corrupt training downstream).
_CORE_NUMERIC = (
    "label",
    "ret_t1",
    "sigma",
    "n_bars",
    "uniqueness_weight",
    "sample_weight",
)


def gather_label_health(labels_by_symbol: dict[str, pl.DataFrame]) -> list[dict]:
    """Compute a QA summary per symbol from already-loaded label frames (pure).

    Wraps :func:`label_qa` (raw count, effective N, balance, barriers, hold,
    resolved-at-close, uniqueness) and adds label-version set, ``t0`` span, and a
    null/inf count on the core numeric columns.
    """
    out: list[dict] = []
    for symbol, lab in labels_by_symbol.items():
        info: dict = {"symbol": symbol, "labels": 0 if lab.width == 0 else lab.height}
        if lab.is_empty() or lab.width == 0:
            out.append(info)
            continue

        info["qa"] = label_qa(lab)
        info["micro_label_versions"] = (
            sorted(lab["micro_label_version"].unique().to_list())
            if "micro_label_version" in lab.columns
            else []
        )
        info["t0_first"] = lab["t0"].min()
        info["t0_last"] = lab["t0"].max()

        null_inf: dict[str, int] = {}
        for c in _CORE_NUMERIC:
            if c not in lab.columns:
                continue
            col = lab[c]
            bad = int(col.is_null().sum())
            if col.dtype in (pl.Float64, pl.Float32):
                bad += int((~col.is_finite()).sum())
            if bad:
                null_inf[c] = bad
        info["null_inf_counts"] = null_inf
        out.append(info)
    return out


def load_and_gather(symbols: list[str], start: datetime, end: datetime) -> list[dict]:
    """Read labels from the lake for ``symbols`` (``t0`` in [start, end]) and gather QA."""
    from options_system.microstructure.labels import read_micro_labels

    frames = {s: read_micro_labels(s, start, end) for s in symbols}
    return gather_label_health(frames)


def _fmt_dt(v: object) -> str:
    return "—" if v is None else str(v)


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - thin CLI report
    import argparse

    from options_system.microstructure.config import MicrostructureConfig

    p = argparse.ArgumentParser(
        prog="micro_label_health", description="Micro-label QA report (read-only)"
    )
    p.add_argument("--symbols", nargs="+", default=None)
    p.add_argument("--start", default="2000-01-01")
    p.add_argument("--end", default="2100-01-01")
    args = p.parse_args(argv)
    symbols = args.symbols or MicrostructureConfig.load().symbols()
    start = datetime.fromisoformat(args.start).replace(tzinfo=UTC)
    end = datetime.fromisoformat(args.end).replace(tzinfo=UTC)

    for info in load_and_gather(symbols, start, end):
        print(f"\n=== {info['symbol']} ===")
        if info["labels"] == 0:
            print("  no labels on disk for this window")
            continue
        qa = info["qa"]
        bal = qa["label_balance"]
        print(
            f"  labels={info['labels']:,}  versions={info['micro_label_versions']}  "
            f"t0=[{_fmt_dt(info['t0_first'])} .. {_fmt_dt(info['t0_last'])}]"
        )
        print(
            f"  effective_N={qa['effective_n']:.1f} over {qa['n_session_days']} RTH days "
            f"-> {qa['effective_n_per_day']:.1f}/day  avg_uniqueness={qa['avg_uniqueness']:.3f}"
        )
        print(
            f"  balance +/-/0 = {bal['pos']:.3f}/{bal['neg']:.3f}/{bal['zero']:.3f}  "
            f"barriers={qa['barrier_touched']}"
        )
        print(
            f"  hold(min) median={qa['hold_minutes']['median']:.1f} "
            f"IQR[{qa['hold_minutes']['iqr_lo']:.1f},{qa['hold_minutes']['iqr_hi']:.1f}]  "
            f"resolved_at_close={qa['frac_resolved_at_close']:.3f}"
        )
        if info["null_inf_counts"]:
            print(f"  NULL/INF in core columns: {info['null_inf_counts']}")
        else:
            print("  NULL/INF in core columns: none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
