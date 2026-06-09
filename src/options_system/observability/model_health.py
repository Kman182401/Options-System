"""Model-health view — read the saved evaluation run and surface the verdict.

A thin, read-only Streamlit view over the JSON runs written by
:func:`options_system.models.evaluate_model.save_model_run` (the same payload logged
to MLflow). For each symbol it shows the honest **VERDICT**, the pooled-OOS gate
metrics (directional accuracy, excess-over-long Sharpe/PSR/**DSR**, PBO, effective
N), the CPCV out-of-sample excess-Sharpe distribution, and the SHAP summary — the
numbers you actually gate a model on. It renders nothing tradeable and issues no
orders.

The data-gathering logic (:func:`gather_model_health`) is pure and unit-tested; the
Streamlit code is a thin view over it. Run with::

    uv run streamlit run src/options_system/observability/model_health.py

(after producing a run with ``python -m options_system.models.run``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from config.settings import Settings

from ..models.evaluate_model import _runs_dir, read_model_run


def _ta_comparison_path(symbol: str, *, runs_dir: Path | None = None) -> Path:
    """Path to the opt-in TA comparison JSON for ``symbol`` (may not exist)."""
    return (runs_dir or _runs_dir()) / f"{symbol}_ta_comparison.json"


def gather_model_health(
    symbols: list[str],
    *,
    runs_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Per-symbol model summary from the saved evaluation runs.

    Returns one fully-keyed dict per symbol (same keys whether or not a run exists,
    so the view never ``KeyError``s). ``has_run`` is ``False`` for symbols with no
    saved evaluation.
    """
    out: list[dict[str, Any]] = []
    for symbol in symbols:
        info: dict[str, Any] = {
            "symbol": symbol,
            "has_run": False,
            "verdict": None,
            "verdict_checks": {},
            "model_version": None,
            "n_samples": 0,
            "n_features": 0,
            "effective_n_total": 0.0,
            "n_trials": 0,
            "selected_overrides": {},
            "pooled_kfold": {},
            "pbo": None,
            "cpcv": {},
            "shap_top_features": [],
            "mlflow_run_id": None,
            # TA awareness (Phase 10): the canonical run is price+macro (with_ta=False);
            # the opt-in TA experiment lives in <symbol>_ta_comparison.json.
            "with_ta": False,
            "ta_feature_version": None,
            "n_ta_features": 0,
            "has_ta_comparison": _ta_comparison_path(symbol, runs_dir=runs_dir).exists(),
        }
        run = read_model_run(symbol, runs_dir=runs_dir)
        if run is not None:
            info.update(
                has_run=True,
                verdict=run.get("verdict"),
                verdict_checks=run.get("verdict_checks", {}),
                model_version=run.get("model_version"),
                n_samples=run.get("n_samples", 0),
                n_features=run.get("n_features", 0),
                effective_n_total=run.get("effective_n_total", 0.0),
                n_trials=run.get("n_trials", 0),
                selected_overrides=run.get("selected_overrides", {}),
                pooled_kfold=run.get("pooled_kfold", {}),
                pbo=(run.get("pbo") or {}).get("pbo") if run.get("pbo") else None,
                cpcv=run.get("cpcv", {}),
                shap_top_features=(run.get("shap") or {}).get("top_features", []),
                mlflow_run_id=run.get("mlflow_run_id"),
                with_ta=run.get("with_ta", False),
                ta_feature_version=run.get("ta_feature_version"),
                n_ta_features=run.get("n_ta_features", 0),
            )
        out.append(info)
    return out


def render() -> None:  # pragma: no cover - Streamlit UI
    import streamlit as st

    st.set_page_config(page_title="Options-System · Model Health", layout="wide")
    st.title("Options-System — Model Health")
    st.caption("Read-only view over the honest signal-model verdict. No orders, ever.")

    settings = Settings()
    health = gather_model_health(settings.record_symbols)
    for info in health:
        st.subheader(info["symbol"])
        if not info["has_run"]:
            st.warning("No saved model run. Run `python -m options_system.models.run`.")
            continue

        verdict = info["verdict"] or "unknown"
        (st.success if verdict == "edge" else st.info)(f"VERDICT: {verdict.upper()}")

        p = info["pooled_kfold"]
        cols = st.columns(5)
        cols[0].metric("Directional acc", p.get("directional_accuracy"))
        cols[1].metric("Excess Sharpe", p.get("excess_sharpe"))
        cols[2].metric("Excess DSR", p.get("excess_dsr"))
        cols[3].metric("PBO", "n/a" if info["pbo"] is None else f"{info['pbo']:.2f}")
        cols[4].metric("Effective N", f"{info['effective_n_total']:,.0f}")

        st.markdown(
            f"**Selected** {info['selected_overrides']} · **trials** {info['n_trials']} · "
            f"strategy Sharpe {p.get('strategy_sharpe')} vs long-benchmark "
            f"{p.get('long_benchmark_sharpe')} (skill must beat beta)"
        )
        st.markdown("**Verdict checks** (all must pass for an edge)")
        st.json(info["verdict_checks"])

        ex = info["cpcv"].get("excess_sharpe", {})
        st.markdown(
            f"**CPCV excess-over-long Sharpe** — mean {ex.get('mean')}, median "
            f"{ex.get('median')}, min {ex.get('min')}, max {ex.get('max')} "
            f"({info['cpcv'].get('n_paths')} paths)"
        )

        if info["shap_top_features"]:
            st.markdown(f"**SHAP top drivers:** {', '.join(info['shap_top_features'][:8])}")
        png = Path(Settings().data_dir) / "models" / "runs" / f"{info['symbol']}_shap.png"
        if png.exists():
            st.image(str(png), caption="SHAP summary (global)")
        if info["mlflow_run_id"]:
            st.caption(f"MLflow run: {info['mlflow_run_id']}")
        st.caption("PBO→1 or DSR≤0.5 ⇒ no convincing edge. Skill is reported over and above beta.")

    st.caption("Refresh the page to update.")


if __name__ == "__main__":  # streamlit runs the script with __name__ == "__main__"
    render()
