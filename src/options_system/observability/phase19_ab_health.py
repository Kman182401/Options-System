"""Read-only summary of the Phase-19 sentiment A/B edge verdict.

Reads the saved combined verdict (``data/phase19_ab/runs/verdict.json``, produced by
``microstructure_model.phase19_ab``) and prints, per symbol, the baseline-vs-treatment
six-gate table, the SHAP top sentiment features the Treatment arm used (if any), the
attribution, and the frozen decision-rule outcome. It runs nothing and writes nothing —
no model, no network, no spend::

    uv run python -m options_system.observability.phase19_ab_health --symbols ES NQ
"""

from __future__ import annotations

import argparse
from typing import Any

from options_system.microstructure_model.phase19_ab import read_verdict

_GATES = (
    ("pbo_below_max", "PBO<max"),
    ("gross_dsr_above_min", "DSR>min"),
    ("positive_gross_return", "mean_gross>0"),
    ("action_rate_above_min", "action>=min"),
    ("macro_f1_above_min", "macroF1>=min"),
    ("cpcv_median_gross_sharpe_positive", "CPCV_med>0"),
)


def _gate_line(label: str, view: dict[str, Any]) -> str:
    checks = view["verdict_checks"]
    cells = " ".join(f"{short}={'P' if checks.get(key) else 'F'}" for key, short in _GATES)
    return f"    {label:9s} {view['verdict']:18s} [{cells}]"


def _print_symbol(sym: str, res: dict[str, Any]) -> None:
    b, t = res["baseline"], res["treatment"]
    print(f"\n=== {sym} ===  rows={res['n_rows']} effective_n={res['effective_n']}")
    bm, tm = b["metrics"], t["metrics"]
    print(
        f"    metrics       B(mm1): PBO={bm['pbo']} DSR={bm['gross_dsr']} "
        f"mean_gross={bm['mean_gross_return']} macroF1={bm['macro_f1']} "
        f"action={bm['action_rate']} CPCV_med={bm['cpcv_median_gross_sharpe']}"
    )
    print(
        f"    metrics       T(mm2): PBO={tm['pbo']} DSR={tm['gross_dsr']} "
        f"mean_gross={tm['mean_gross_return']} macroF1={tm['macro_f1']} "
        f"action={tm['action_rate']} CPCV_med={tm['cpcv_median_gross_sharpe']}"
    )
    print(_gate_line("baseline", b))
    print(_gate_line("treatment", t))
    if t.get("shap_top"):
        print(f"    T SHAP top: {t['shap_top'][:6]}")
    print(f"    attribution: {res['attribution']['attribution']}")


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - thin CLI report
    p = argparse.ArgumentParser(
        prog="observability.phase19_ab_health", description="Phase-19 A/B verdict summary"
    )
    p.add_argument("--symbols", nargs="+", default=None, help="default: all in the saved verdict")
    args = p.parse_args(argv)

    verdict = read_verdict()
    if verdict is None:
        print("no saved Phase-19 A/B verdict — run microstructure_model.phase19_ab first")
        return 0

    print(
        f"Phase-19 A/B  ({verdict['micro_model_version_baseline']} vs "
        f"{verdict['micro_model_version_treatment']}+{verdict['sentiment_feature_version']})  "
        f"window={verdict['window']['start']}..{verdict['window']['end']}"
    )
    print(f"gate thresholds: {verdict['verdict_thresholds']}")
    symbols = args.symbols or list(verdict["symbols"])
    for sym in symbols:
        res = verdict["symbols"].get(sym)
        if res is None:
            print(f"\n=== {sym} ===  (not in saved verdict)")
            continue
        _print_symbol(sym, res)

    d = verdict["decision"]
    print(
        f"\nDECISION: {d['overall']}  "
        f"candidates={d['candidates'] or 'none'}  fragile={d['fragile']}"
    )
    print(f"  {d['note']}")
    print("  (gross signal-return proxy; not an economic backtest; authorizes no live trading)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
