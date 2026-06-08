"""Data-health dashboard (Streamlit) — read-only over the Parquet lake.

Per symbol: last-bar age, rows/day, gap summary, current front-month contract,
last roll event, and validation status. It only *reads* — it never records,
repairs, or trades.

Run it:

    uv run streamlit run src/options_system/observability/data_health.py

The data-gathering logic (:func:`gather_health`) is pure and unit-tested; the
Streamlit code below is a thin view over it.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import cast

import polars as pl

from config.settings import Settings
from options_system.data.store import DuckStore
from options_system.data.validate import validate_bars


def gather_health(
    store: DuckStore,
    symbols: list[str],
    as_of: datetime | None = None,
    lookback_days: int = 3,
) -> list[dict]:
    """Collect a health summary per symbol (pure; no Streamlit)."""
    now = as_of or datetime.now(UTC)
    start = now - timedelta(days=lookback_days)
    out: list[dict] = []
    for symbol in symbols:
        bars = store.get_bars(symbol, start, now, freq="1m")
        rolls = store._read_rolls(symbol)
        info: dict = {
            "symbol": symbol,
            "rows": bars.height,
            "last_bar_age_s": None,
            "front_contract": None,
            "rows_by_day": None,
            "validation": None,
            "last_roll": None,
        }
        if not bars.is_empty():
            last_ts = cast(datetime, bars["ts_event"].max())
            info["last_bar_age_s"] = (now - last_ts).total_seconds()
            info["front_contract"] = bars.sort("ts_event")["contract_id"][-1]
            info["rows_by_day"] = (
                bars.with_columns(pl.col("ts_event").dt.date().alias("date"))
                .group_by("date")
                .len()
                .sort("date")
            )
            info["validation"] = validate_bars(bars).summary()
        if not rolls.is_empty():
            last = rolls.sort("ts_event").row(-1, named=True)
            info["last_roll"] = {
                "ts_event": last["ts_event"],
                "from": last["from_contract_id"],
                "to": last["to_contract_id"],
                "rule": last["rule"],
            }
        out.append(info)
    return out


def render() -> None:  # pragma: no cover - Streamlit UI
    import streamlit as st

    st.set_page_config(page_title="Options-System · Data Health", layout="wide")
    st.title("Options-System — Data Health")
    st.caption("Read-only view over the local Parquet lake. No orders, ever.")

    settings = Settings()
    store = DuckStore()
    health = gather_health(store, settings.record_symbols)

    for info in health:
        st.subheader(info["symbol"])
        if info["rows"] == 0:
            st.warning("No bars recorded yet. Is the recorder running and Gateway up?")
            continue

        age = info["last_bar_age_s"]
        cols = st.columns(4)
        cols[0].metric("Last bar age (s)", f"{age:.0f}" if age is not None else "—")
        cols[1].metric("Rows (lookback)", info["rows"])
        cols[2].metric("Front month", info["front_contract"] or "—")
        validation = info["validation"] or {}
        cols[3].metric("Validation", "OK" if validation.get("ok") else "ISSUES")

        if age is not None and age > 300:
            st.warning(f"Last bar is {age:.0f}s old — recorder may be stalled.")
        if validation and not validation.get("ok"):
            st.error(f"Validation issues: {validation.get('checks_failed')}")
        elif validation.get("warnings"):
            st.info(f"Validation warnings: {validation.get('warnings')}")

        if info["last_roll"]:
            r = info["last_roll"]
            st.text(f"Last roll: {r['from']} → {r['to']} ({r['rule']}) at {r['ts_event']}")
        if info["rows_by_day"] is not None:
            st.dataframe(info["rows_by_day"].to_pandas(), hide_index=True)

    st.caption("Refresh the page to update.")


if __name__ == "__main__":  # streamlit runs the script with __name__ == "__main__"
    render()
