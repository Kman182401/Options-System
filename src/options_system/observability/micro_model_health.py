"""Micro-model health — read-only over the saved micro-model run summaries.

Per symbol, loads ``data/micro_models/runs/<symbol>.json`` (written by
``microstructure_model.run``) and prints the verdict, the individual gate checks,
the sample / effective-N counts, the true + predicted class balance, the action
rate, PBO, gross DSR, the CPCV gross-Sharpe path distribution, and the top SHAP
features when present. It does not recompute anything — it only surfaces the
auditable verdict. No network, no Databento, no writes.

    uv run python -m options_system.observability.micro_model_health --symbols ES NQ
"""

from __future__ import annotations

from typing import Any

from options_system.microstructure_model.evaluate import read_micro_run


def gather_model_health(summaries_by_symbol: dict[str, dict[str, Any] | None]) -> list[dict]:
    """Flatten the headline verdict + gate metrics from saved run summaries (pure)."""
    out: list[dict] = []
    for symbol, s in summaries_by_symbol.items():
        if s is None:
            out.append({"symbol": symbol, "available": False})
            continue
        sig = s.get("signal_return", {})
        cls = s.get("classification", {})
        cpcv_gs = s.get("cpcv", {}).get("gross_sharpe", {})
        out.append(
            {
                "symbol": symbol,
                "available": True,
                "verdict": s.get("verdict"),
                "verdict_checks": s.get("verdict_checks", {}),
                "n_samples": s.get("n_samples"),
                "effective_n": s.get("effective_n"),
                "n_features": s.get("n_features"),
                "class_balance": s.get("class_balance"),
                "pred_balance": s.get("pred_balance"),
                "action_rate": s.get("action_rate"),
                "macro_f1": cls.get("macro_f1"),
                "weighted_accuracy": cls.get("weighted_accuracy"),
                "balanced_accuracy": cls.get("balanced_accuracy"),
                "gross_sharpe": sig.get("gross_sharpe"),
                "gross_dsr": sig.get("gross_dsr"),
                "mean_gross_return": sig.get("mean_gross_return"),
                "pbo": (s.get("pbo") or {}).get("pbo"),
                "cpcv_gross_sharpe": cpcv_gs,
                "top_features": (s.get("shap") or {}).get("top_features"),
            }
        )
    return out


def _print_one(info: dict) -> None:
    print(f"\n=== {info['symbol']} ===")
    if not info["available"]:
        print("  no saved micro-model run for this symbol")
        return
    print(
        f"  VERDICT: {str(info['verdict']).upper()}  "
        f"n={info['n_samples']} eff_n={info['effective_n']} feat={info['n_features']}"
    )
    print(f"  class_balance(true) = {info['class_balance']}")
    print(f"  pred_balance        = {info['pred_balance']}  action_rate={info['action_rate']}")
    print(
        f"  macro_F1={info['macro_f1']} weighted_acc={info['weighted_accuracy']} "
        f"balanced_acc={info['balanced_accuracy']}"
    )
    print(
        f"  gross_SR={info['gross_sharpe']} gross_DSR={info['gross_dsr']} "
        f"mean_gross={info['mean_gross_return']} PBO={info['pbo']}"
    )
    gs = info["cpcv_gross_sharpe"] or {}
    print(
        f"  CPCV gross-Sharpe: mean={gs.get('mean')} median={gs.get('median')} "
        f"min={gs.get('min')} max={gs.get('max')}"
    )
    failed = [k for k, v in (info["verdict_checks"] or {}).items() if not v]
    print(f"  gates failed: {failed or 'none'}")
    if info.get("top_features"):
        print(f"  SHAP top: {info['top_features'][:6]}")


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - thin CLI report
    import argparse

    from options_system.microstructure.config import MicrostructureConfig

    p = argparse.ArgumentParser(
        prog="micro_model_health", description="Micro-model verdict/gate report (read-only)"
    )
    p.add_argument("--symbols", nargs="+", default=None)
    args = p.parse_args(argv)
    symbols = args.symbols or MicrostructureConfig.load().symbols()
    summaries = {s: read_micro_run(s) for s in symbols}
    for info in gather_model_health(summaries):
        _print_one(info)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
