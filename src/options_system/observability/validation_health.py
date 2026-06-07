"""Validation-health view — read a saved evaluation run and surface the verdict.

A thin, read-only Streamlit view over the JSON runs written by
:func:`options_system.validation.evaluate.save_run`. For each symbol it shows the
out-of-sample metric distribution across CPCV paths, the overfitting verdict
(PBO, PSR, DSR), the **effective sample size**, and the config/version that
produced the run — the numbers you actually gate a model on.

The data-gathering logic (:func:`gather_validation_health`) is pure and
unit-tested; the Streamlit code is a thin view over it. Run with::

    uv run streamlit run src/options_system/observability/validation_health.py

(after producing a run with ``python -m options_system.validation.evaluate``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from config.settings import Settings

from ..validation.evaluate import read_run


def gather_validation_health(
    symbols: list[str],
    *,
    runs_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Per-symbol validation summary from the saved evaluation runs.

    Returns one fully-keyed dict per symbol (the same keys whether or not a run
    exists, so the view never ``KeyError``s). ``has_run`` is ``False`` for symbols
    with no saved evaluation.
    """
    out: list[dict[str, Any]] = []
    for symbol in symbols:
        info: dict[str, Any] = {
            "symbol": symbol,
            "has_run": False,
            "validation_version": None,
            "n_samples": 0,
            "n_features": 0,
            "effective_n_total": 0.0,
            "pbo": None,
            "estimators": [],
            "kfold": {},
            "cpcv": {},
        }
        run = read_run(symbol, runs_dir=runs_dir)
        if run is not None:
            pbo = run.get("pbo")
            info.update(
                has_run=True,
                validation_version=run.get("validation_version"),
                n_samples=run.get("n_samples", 0),
                n_features=run.get("n_features", 0),
                effective_n_total=run.get("effective_n_total", 0.0),
                pbo=(pbo or {}).get("pbo") if pbo else None,
                estimators=list(run.get("kfold", {})),
                kfold=run.get("kfold", {}),
                cpcv=run.get("cpcv", {}),
            )
        out.append(info)
    return out


def render() -> None:  # pragma: no cover - Streamlit UI
    import streamlit as st

    st.set_page_config(page_title="Options-System · Validation Health", layout="wide")
    st.title("Options-System — Validation Health")
    st.caption("Read-only view over saved evaluation runs. No orders, ever.")

    settings = Settings()
    health = gather_validation_health(settings.record_symbols)
    for info in health:
        st.subheader(info["symbol"])
        if not info["has_run"]:
            st.warning(
                "No saved validation run. Run `python -m options_system.validation.evaluate`."
            )
            continue

        cols = st.columns(4)
        cols[0].metric("Samples", f"{info['n_samples']:,}")
        cols[1].metric("Effective N", f"{info['effective_n_total']:,.0f}")
        cols[2].metric("PBO", "n/a" if info["pbo"] is None else f"{info['pbo']:.2f}")
        cols[3].metric("validation_version", info["validation_version"])

        kfold_rows = [
            {
                "estimator": name,
                "accuracy": met.get("accuracy"),
                "sharpe": met.get("sharpe"),
                "PSR": met.get("psr"),
                "DSR": met.get("dsr"),
                "effective_N": met.get("effective_n"),
            }
            for name, met in info["kfold"].items()
        ]
        st.markdown("**Purged K-fold (pooled OOS)**")
        st.dataframe(kfold_rows, hide_index=True)

        cpcv_rows = [
            {
                "estimator": name,
                "paths": met.get("n_paths"),
                "sharpe_mean": met.get("sharpe_mean"),
                "sharpe_std": met.get("sharpe_std"),
                "sharpe_min": met.get("sharpe_min"),
                "sharpe_max": met.get("sharpe_max"),
                "effective_N_per_path": met.get("effective_n_mean"),
            }
            for name, met in info["cpcv"].items()
        ]
        st.markdown("**CPCV out-of-sample path distribution**")
        st.dataframe(cpcv_rows, hide_index=True)
        st.caption("PBO→1 or DSR≤0 ⇒ the result is noise. A baseline should look unskilled here.")

    st.caption("Refresh the page to update.")


if __name__ == "__main__":  # streamlit runs the script with __name__ == "__main__"
    render()
