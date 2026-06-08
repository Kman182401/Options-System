"""Microstructure-bar health / QA — read-only over the ``micro_bars`` lake table.

Per symbol: bar counts, bars-per-session distribution, thin/short sessions,
NaN/inf counts per feature, contract rolls observed, the fraction of incomplete
(trailing) bars, key-feature distribution summaries, and the headline
**OFI ↔ same-bar mid-change** sanity correlation (a well-established microstructure
stylized fact — strongly positive when construction is correct).

The gathering logic (:func:`gather_micro_health`) is pure and unit-tested; the
Streamlit code and :func:`main` text report are thin views over it. It only reads.

Run the dashboard:

    uv run streamlit run src/options_system/observability/micro_health.py
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import polars as pl

from options_system.microstructure.bars import feature_names
from options_system.microstructure.config import MicrostructureConfig

# Features whose distribution we surface explicitly (the rest still get NaN counts).
_KEY_FEATURES = (
    "ofi_top",
    "qimb_top_twa",
    "signed_vol",
    "trade_imbalance",
    "spread_ticks_twa",
    "rv_intrabar",
    "duration_s",
    "dmid",
)


def _num(x: Any) -> float:
    """Coerce a polars scalar aggregate (a widened union in the stubs) to float."""
    return float(x)


def _int(x: Any) -> int:
    return int(x)


def _safe_corr(df: pl.DataFrame, a: str, b: str) -> float | None:
    """Pearson correlation of two columns over finite rows; None if degenerate."""
    sub = df.select(a, b).drop_nulls()
    if sub.height < 3:
        return None
    sub = sub.filter(pl.col(a).is_finite() & pl.col(b).is_finite())
    if sub.height < 3 or sub[a].n_unique() < 2 or sub[b].n_unique() < 2:
        return None
    return float(pl.DataFrame(sub).select(pl.corr(a, b)).item())


def gather_micro_health(
    bars_by_symbol: dict[str, pl.DataFrame], cfg: MicrostructureConfig
) -> list[dict]:
    """Compute a QA summary per symbol from already-loaded bar frames (pure)."""
    feats = feature_names(cfg)
    out: list[dict] = []
    for symbol, bars in bars_by_symbol.items():
        info: dict = {"symbol": symbol, "bars": bars.height}
        if bars.is_empty():
            out.append(info)
            continue

        per_session = (
            bars.with_columns(pl.col("ts_event").dt.date().alias("_d"))
            .group_by("_d")
            .len()
            .sort("_d")
        )
        counts = per_session["len"]
        median_bps = _num(counts.median()) if counts.len() else 0.0
        thin_cut = max(1.0, median_bps / 3.0)

        info.update(
            {
                "sessions": per_session.height,
                "bars_per_session_median": median_bps,
                "bars_per_session_min": _int(counts.min()) if counts.len() else 0,
                "bars_per_session_max": _int(counts.max()) if counts.len() else 0,
                "thin_sessions": _int((counts < thin_cut).sum()),
                # rolls are detected by instrument_id (con_id) change; contract_id
                # resolves to the continuous alias (e.g. ES.v.0), not the outright.
                "rolls_observed": bars["con_id"].n_unique() - 1,
                "contracts": bars["contract_id"].unique().to_list(),
                "pct_incomplete_bars": _num((~bars["bar_complete"]).sum()) / bars.height * 100.0,
                "median_duration_s": _num(bars["duration_s"].median()),
                "ofi_vs_dmid_corr": _safe_corr(bars, "ofi_top", "dmid"),
                "nan_inf_counts": {},
                "distributions": {},
            }
        )
        for f in feats:
            if f not in bars.columns:
                continue
            col = bars[f]
            n_bad = _int(col.is_null().sum()) + _int((~col.is_finite()).sum())
            if n_bad:
                info["nan_inf_counts"][f] = n_bad
        for f in _KEY_FEATURES:
            if f in bars.columns:
                c = bars[f].filter(bars[f].is_finite())
                if c.len():
                    info["distributions"][f] = {
                        "mean": _num(c.mean()),
                        "std": _num(c.std()) if c.len() > 1 else 0.0,
                        "min": _num(c.min()),
                        "max": _num(c.max()),
                    }
        out.append(info)
    return out


def load_and_gather(
    symbols: list[str], start: datetime, end: datetime, cfg: MicrostructureConfig | None = None
) -> list[dict]:
    """Read bars from the lake for ``symbols`` and compute the health summary."""
    from options_system.microstructure.ingest import read_micro_bars

    cfg = cfg or MicrostructureConfig.load()
    frames = {s: read_micro_bars(s, start, end) for s in symbols}
    return gather_micro_health(frames, cfg)


def _fmt(v: float | None, nd: int = 4) -> str:
    return "—" if v is None else f"{v:.{nd}f}"


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - thin CLI report
    import argparse

    p = argparse.ArgumentParser(prog="micro_health", description="Microstructure bar QA report")
    p.add_argument("--symbols", nargs="+", default=None)
    p.add_argument("--start", default="2000-01-01")
    p.add_argument("--end", default="2100-01-01")
    args = p.parse_args(argv)
    cfg = MicrostructureConfig.load()
    symbols = args.symbols or cfg.symbols()
    start = datetime.fromisoformat(args.start).replace(tzinfo=UTC)
    end = datetime.fromisoformat(args.end).replace(tzinfo=UTC)
    for info in load_and_gather(symbols, start, end, cfg):
        print(f"\n=== {info['symbol']} ===")
        if info["bars"] == 0:
            print("  no bars on disk")
            continue
        print(
            f"  bars={info['bars']:,}  sessions={info['sessions']}  "
            f"bars/session median={info['bars_per_session_median']:.0f} "
            f"[{info['bars_per_session_min']}..{info['bars_per_session_max']}]  "
            f"thin_sessions={info['thin_sessions']}"
        )
        print(
            f"  rolls={info['rolls_observed']}  contracts={info['contracts']}  "
            f"incomplete_bars={info['pct_incomplete_bars']:.2f}%  "
            f"median_duration={info['median_duration_s']:.1f}s"
        )
        print(
            f"  SANITY OFI↔Δmid corr: ofi_top={_fmt(info['ofi_vs_dmid_corr'])}  "
            "(expect strongly positive)"
        )
        if info["nan_inf_counts"]:
            print(f"  NaN/inf counts: {info['nan_inf_counts']}")
        else:
            print("  NaN/inf counts: none")
    return 0


def render() -> None:  # pragma: no cover - Streamlit UI
    import streamlit as st

    from options_system.microstructure.ingest import read_micro_bars

    st.set_page_config(page_title="Options-System · Microstructure Health", layout="wide")
    st.title("Options-System — Microstructure (OFI) Bar Health")
    st.caption("Read-only view over the local micro_bars lake. No orders, ever.")

    cfg = MicrostructureConfig.load()
    lo = datetime(2000, 1, 1, tzinfo=UTC)
    hi = datetime(2100, 1, 1, tzinfo=UTC)
    frames = {s: read_micro_bars(s, lo, hi) for s in cfg.symbols()}
    for info in gather_micro_health(frames, cfg):
        st.subheader(info["symbol"])
        if info["bars"] == 0:
            st.warning("No microstructure bars on disk. Run the ingest with --confirm.")
            continue
        cols = st.columns(4)
        cols[0].metric("Bars", f"{info['bars']:,}")
        cols[1].metric("Sessions", info["sessions"])
        cols[2].metric("Bars/session (median)", f"{info['bars_per_session_median']:.0f}")
        cols[3].metric("OFI↔Δmid corr", _fmt(info["ofi_vs_dmid_corr"], 3))
        if info["nan_inf_counts"]:
            st.error(f"NaN/inf in features: {info['nan_inf_counts']}")
        st.text(f"Contracts: {info['contracts']}  rolls={info['rolls_observed']}")


if __name__ == "__main__":  # streamlit runs with __name__ == '__main__'
    import sys

    if any("streamlit" in a for a in sys.argv) or len(sys.argv) == 1:
        try:
            render()
        except Exception:  # noqa: BLE001 - fall back to text report outside streamlit
            raise SystemExit(main()) from None
    else:
        raise SystemExit(main())
