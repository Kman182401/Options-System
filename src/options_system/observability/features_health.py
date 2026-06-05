"""Features-health dashboard (Streamlit) — read-only over the feature lake.

Per symbol: row coverage, feature_version, degraded-row count, last feature
timestamp, and the null/NaN rate of every feature (so a feature that silently
stops populating is visible). It only *reads*.

Run it:

    uv run streamlit run src/options_system/observability/features_health.py

The data-gathering logic (:func:`gather_feature_health`) is pure and unit-tested;
the Streamlit code is a thin view over it.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import polars as pl

from config.settings import Settings
from options_system.data.store import DuckStore
from options_system.features.build import read_features
from options_system.features.compute import feature_names
from options_system.features.config import FeatureConfig


def gather_feature_health(
    store: DuckStore,
    symbols: list[str],
    cfg: FeatureConfig,
    as_of: datetime | None = None,
    lookback_days: int = 30,
) -> list[dict]:
    """Per-symbol feature-table health summary (pure; no Streamlit)."""
    now = as_of or datetime.now(UTC)
    start = now - timedelta(days=lookback_days)
    names = feature_names(cfg)
    out: list[dict] = []
    for symbol in symbols:
        feats = read_features(symbol, start, now, store=store)
        info: dict = {
            "symbol": symbol,
            "rows": feats.height,
            "feature_version": None,
            "degraded_rows": 0,
            "last_ts": None,
            "null_rate": {},
        }
        if not feats.is_empty():
            info["feature_version"] = sorted(set(feats["feature_version"].to_list()))
            info["degraded_rows"] = int(feats["degraded"].sum())
            info["last_ts"] = feats["ts_event"].max()
            n = feats.height
            info["null_rate"] = {
                name: round(feats[name].null_count() / n, 4)
                for name in names
                if name in feats.columns
            }
        out.append(info)
    return out


def render() -> None:  # pragma: no cover - Streamlit UI
    import streamlit as st

    st.set_page_config(page_title="Options-System · Features Health", layout="wide")
    st.title("Options-System — Features Health")
    st.caption("Read-only view over the local feature lake. No orders, ever.")

    settings = Settings()
    cfg = FeatureConfig.load()
    store = DuckStore()
    health = gather_feature_health(store, settings.record_symbols, cfg)

    for info in health:
        st.subheader(info["symbol"])
        if info["rows"] == 0:
            st.warning(
                "No features in the lookback. Run `python -m options_system.features.build`."
            )
            continue
        cols = st.columns(4)
        cols[0].metric("Feature rows", info["rows"])
        cols[1].metric("feature_version", ", ".join(info["feature_version"] or []))
        cols[2].metric("Degraded rows", info["degraded_rows"])
        last = info["last_ts"]
        age = (datetime.now(UTC) - last).total_seconds() if last else math.nan
        cols[3].metric("Last feature age (s)", f"{age:.0f}" if not math.isnan(age) else "—")

        nr = info["null_rate"]
        worst = sorted(nr.items(), key=lambda kv: kv[1], reverse=True)[:10]
        st.text("Highest null/NaN rates (post-warmup these should be ~0):")
        st.dataframe(
            pl.DataFrame(
                {"feature": [k for k, _ in worst], "null_rate": [v for _, v in worst]}
            ).to_pandas(),
            hide_index=True,
        )

    st.caption("Refresh the page to update.")


if __name__ == "__main__":  # streamlit runs the script with __name__ == "__main__"
    render()
