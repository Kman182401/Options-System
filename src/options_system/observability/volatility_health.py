"""Read-only summary of the Phase-21 volatility-forecast skill verdict.

Reads the saved combined verdict (``data/volatility/runs/verdict.json``, produced by
``volatility.run``) and prints, per symbol, the QLIKE table (HAR vs treatment) at the primary
horizon, the Diebold-Mariano statistic + p-value, the regime split, the calibration, the
arm-T SHAP block shares, and the frozen decision. Runs nothing, writes nothing — no model,
no network, no spend::

    uv run python -m options_system.observability.volatility_health --symbols MES MNQ
"""

from __future__ import annotations

import argparse
from typing import Any

from options_system.volatility.run import read_verdict


def _print_symbol(sym: str, v: dict[str, Any]) -> None:
    p = v["primary"]
    g = v["gates"]
    print(
        f"\n=== {sym} ===  objective={v['objective_used']}  "
        f"sentiment_coverage={v['sentiment_coverage']}"
    )
    print(
        f"    h={p['horizon']}  n_oos={p['n_oos']}  "
        f"QLIKE  HAR={p['qlike_har']}  treat={p['qlike_treat']}  "
        f"(improvement={round(p['qlike_har'] - p['qlike_treat'], 6)})"
    )
    print(f"    DM (HLN) stat={p['dm_stat_hln']}  one-sided p={p['dm_p_value']}")
    for name, r in p["regime"].items():
        print(
            f"    regime {name:9s} n={r['n']}  QLIKE HAR={r['qlike_har']} "
            f"treat={r['qlike_treat']}  treat<=HAR={r['treat_le_har']}"
        )
    cal = p["calibration_treat"]
    print(f"    calibration (treat): intercept={cal['intercept']} slope={cal['slope']}")
    print(
        f"    GATES: G1_accuracy={g['g1_accuracy']}  "
        f"G2_regime_robustness={g['g2_regime_robustness']}  -> {v['verdict']}"
    )
    if v.get("shap_top"):
        print(f"    arm-T SHAP top: {v['shap_top'][:6]}  block_shares={v.get('shap_block_shares')}")


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - thin CLI report
    p = argparse.ArgumentParser(
        prog="observability.volatility_health", description="Phase-21 volatility verdict summary"
    )
    p.add_argument("--symbols", nargs="+", default=None, help="default: all in the saved verdict")
    args = p.parse_args(argv)

    verdict = read_verdict()
    if verdict is None:
        print("no saved Phase-21 verdict — run volatility.run first")
        return 0

    print(
        f"Phase-21 volatility-forecast  (benchmark={verdict['benchmark']}, "
        f"primary h={verdict['primary_horizon']}, DM alpha={verdict['dm_alpha']}, "
        f"OOS from {verdict['oos_start']})"
    )
    symbols = args.symbols or list(verdict["symbols"])
    for sym in symbols:
        v = verdict["symbols"].get(sym)
        if v is None:
            print(f"\n=== {sym} ===  (not in saved verdict)")
            continue
        _print_symbol(sym, v)

    d = verdict["decision"]
    cands = d["candidates"] or "none"
    print(f"\nDECISION: {d['overall']}  candidates={cands}  fragile={d['fragile']}")
    print(f"  {d['note']}")
    print("  (forecast-skill verdict only; forecast skill is not tradeable money; no live trading)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
