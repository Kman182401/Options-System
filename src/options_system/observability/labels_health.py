"""Labels-health dashboard (Streamlit) — read-only over the label lake.

Per symbol: row count, ``label_version``, **class balance** (+1/−1/0), the
barrier-hit distribution (up/dn/time/roll), mean average-uniqueness, the
fraction of labels whose window crossed a roll, the degraded fraction, and the
last event time. It only *reads* — no orders, no model, ever.

Run it:

    uv run streamlit run src/options_system/observability/labels_health.py

The data-gathering logic (:func:`gather_label_health`) is pure and unit-tested;
the Streamlit code is a thin view over it. Class imbalance is *expected* and is
surfaced here on purpose — it is handled later at the model stage (weights /
thresholds), never by retuning the barriers to balance the classes.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl

from config.settings import Settings
from options_system.data.store import DuckStore
from options_system.labeling.build import read_labels
from options_system.labeling.config import LabelConfig

_WIDE_START = datetime(2000, 1, 1, tzinfo=UTC)


def gather_label_health(
    store: DuckStore,
    symbols: list[str],
    as_of: datetime | None = None,
    lookback_days: int = 0,
) -> list[dict]:
    """Per-symbol label-table health summary (pure; no Streamlit).

    ``lookback_days=0`` summarizes the full history (labels are sparse — a wider
    window is the sensible default); a positive value restricts to a recent tail.
    """
    now = as_of or datetime.now(UTC)
    start = _WIDE_START if lookback_days <= 0 else now - timedelta(days=lookback_days)
    out: list[dict] = []
    for symbol in symbols:
        labels = read_labels(symbol, start, now, store=store)
        info: dict = {
            "symbol": symbol,
            "rows": labels.height,
            "label_version": None,
            "class_balance": {},
            "barrier_dist": {},
            "avg_uniqueness": None,
            "pct_roll_crossed": None,
            "pct_degraded": None,
            "last_t0": None,
        }
        if not labels.is_empty():
            n = labels.height
            info["label_version"] = sorted(set(labels["label_version"].to_list()))
            info["class_balance"] = {
                int(r["label"]): round(r["len"] / n, 4)
                for r in labels.group_by("label").len().sort("label").to_dicts()
            }
            info["barrier_dist"] = {
                r["barrier"]: round(r["len"] / n, 4)
                for r in labels.group_by("barrier").len().sort("len", descending=True).to_dicts()
            }
            info["avg_uniqueness"] = round(labels.select(pl.col("avg_uniqueness").mean()).item(), 4)
            info["pct_roll_crossed"] = round(
                100.0 * labels.select(pl.col("roll_crossed").mean()).item(), 3
            )
            info["pct_degraded"] = round(100.0 * labels.select(pl.col("degraded").mean()).item(), 3)
            info["last_t0"] = labels["t0"].max()
        out.append(info)
    return out


def render() -> None:  # pragma: no cover - Streamlit UI
    import streamlit as st

    st.set_page_config(page_title="Options-System · Labels Health", layout="wide")
    st.title("Options-System — Labels Health")
    st.caption("Read-only view over the local label lake. No orders, ever.")

    settings = Settings()
    LabelConfig.load()  # validates the config is loadable alongside the view
    store = DuckStore()
    health = gather_label_health(store, settings.record_symbols)

    for info in health:
        st.subheader(info["symbol"])
        if info["rows"] == 0:
            st.warning("No labels. Run `python -m options_system.labeling.build`.")
            continue
        cols = st.columns(4)
        cols[0].metric("Label rows", info["rows"])
        cols[1].metric("label_version", ", ".join(info["label_version"] or []))
        cols[2].metric("Avg uniqueness", info["avg_uniqueness"])
        cols[3].metric("% roll-crossed", info["pct_roll_crossed"])

        names = {1: "+1 (up)", -1: "−1 (dn)", 0: "0 (time)"}
        bal = info["class_balance"]
        st.text("Class balance (imbalance is expected — handled at the model, not here):")
        st.dataframe(
            pl.DataFrame(
                {
                    "class": [names.get(k, str(k)) for k in bal],
                    "proportion": list(bal.values()),
                }
            ).to_pandas(),
            hide_index=True,
        )
        st.text("Barrier-hit distribution:")
        st.dataframe(
            pl.DataFrame(
                {
                    "barrier": list(info["barrier_dist"].keys()),
                    "proportion": list(info["barrier_dist"].values()),
                }
            ).to_pandas(),
            hide_index=True,
        )
        st.caption(f"% degraded: {info['pct_degraded']} · last t0: {info['last_t0']}")

    st.caption("Refresh the page to update.")


if __name__ == "__main__":  # streamlit runs the script with __name__ == "__main__"
    render()
