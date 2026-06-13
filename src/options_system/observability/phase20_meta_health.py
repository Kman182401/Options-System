"""Read-only summary of the Phase-20 meta-labeling edge verdict.

Reads the saved combined verdict (``data/phase20_meta/runs/verdict.json``, produced by
``microstructure_model.phase20_meta``) and prints, per symbol, the B0-reference-vs-M
gate table, the meta-skill metrics (balanced accuracy + acted-vs-always-act hit-rate),
the SHAP top features the meta-gate used (and the sentiment share), and the frozen
decision-rule outcome. It runs nothing and writes nothing — no model, no network, no
spend::

    uv run python -m options_system.observability.phase20_meta_health --symbols ES NQ
"""

from __future__ import annotations

import argparse
from typing import Any

from options_system.microstructure_model.phase20_meta import read_verdict

_GATES = (
    ("pbo_below_max", "PBO<max"),
    ("gross_dsr_above_min", "DSR>min"),
    ("positive_gross_return", "mean_gross>0"),
    ("action_rate_above_min", "action>=min"),
    ("cpcv_median_gross_sharpe_positive", "CPCV_med>0"),
    ("meta_skill", "meta_skill"),
)


def _gate_line(label: str, view: dict[str, Any]) -> str:
    checks = view["verdict_checks"]
    cells = " ".join(f"{short}={'P' if checks.get(key) else 'F'}" for key, short in _GATES)
    return f"    {label:9s} {view['verdict']:18s} [{cells}]"


def _metric_line(label: str, view: dict[str, Any]) -> str:
    m = view["metrics"]
    return (
        f"    {label:9s} PBO={m['pbo']} DSR={m['gross_dsr']} grossSR={m['gross_sharpe']} "
        f"mean_gross={m['mean_gross_return']} action={m['action_rate']} "
        f"CPCV_med={m['cpcv_median_gross_sharpe']} balAcc={m['balanced_accuracy']} "
        f"acted_hit={m['acted_hit_rate']} always_hit={m['always_act_hit_rate']}"
    )


def _print_symbol(sym: str, res: dict[str, Any]) -> None:
    b0, m = res["b0"], res["meta"]
    print(
        f"\n=== {sym} ===  meta-set={res['n_metaset']} "
        f"(excluded {res['n_excluded_no_side']} no-side) effective_n={res['effective_n']} "
        f"meta_label_rate={res['meta_label_rate']}"
    )
    print(_metric_line("B0 (ref)", b0))
    print(_metric_line("M (meta)", m))
    print(_gate_line("B0 (ref)", b0))
    print(_gate_line("M (meta)", m))
    if m.get("shap_top"):
        print(
            f"    M SHAP top: {m['shap_top'][:6]}  sentiment_share={m.get('shap_sentiment_share')}"
        )
    if m.get("shap_supported_top"):
        print(
            f"    M SHAP (supported region): {m['shap_supported_top'][:6]}  "
            f"sentiment_share={m.get('shap_supported_sentiment_share')}"
        )
    print(f"    m_pass={res['attribution']['m_pass']}  b0_pass={res['attribution']['b0_pass']}")


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - thin CLI report
    p = argparse.ArgumentParser(
        prog="observability.phase20_meta_health",
        description="Phase-20 meta-labeling verdict summary",
    )
    p.add_argument("--symbols", nargs="+", default=None, help="default: all in the saved verdict")
    args = p.parse_args(argv)

    verdict = read_verdict()
    if verdict is None:
        print("no saved Phase-20 meta verdict — run microstructure_model.phase20_meta first")
        return 0

    floor = verdict["meta_skill_min_balanced_accuracy"]
    print(
        f"Phase-20 meta-labeling  (primary={verdict['primary_rule']}, "
        f"tau={verdict['decision_threshold']}, meta-skill balAcc>={floor})"
        f"  window={verdict['window']['start']}..{verdict['window']['end']}"
    )
    print(f"gross gate thresholds (mm1): {verdict['verdict_thresholds']}")
    symbols = args.symbols or list(verdict["symbols"])
    for sym in symbols:
        res = verdict["symbols"].get(sym)
        if res is None:
            print(f"\n=== {sym} ===  (not in saved verdict)")
            continue
        _print_symbol(sym, res)

    d = verdict["decision"]
    cands = d["candidates"] or "none"
    print(f"\nDECISION: {d['overall']}  candidates={cands}  fragile={d['fragile']}")
    print(f"  {d['note']}")
    print("  (gross signal-return proxy; not an economic backtest; authorizes no live trading)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
