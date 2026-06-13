"""Daily volatility-forecast matrix assembly (leak-free, per symbol).

Builds, per symbol, a daily base frame keyed on the decision day ``t`` (its RTH session close):

* ``rv`` — the 5-minute sub-sampled daily realized variance (``realized.daily_realized_variance``);
* the three causal **HAR predictors** (daily/weekly/monthly trailing log-RV, as-of ``t``);
* the **treatment-only** feature blocks attached **as-of the session close** ``t`` — the existing
  leak-safe price/vol/volume/time/cross-asset features (the per-minute ``features`` lake, backward
  as-of), macro-event features, and the ``s2`` sentiment aggregates where covered (mostly null over
  2019-2026 since GDELT history is ~3 months — kept, never imputed; coverage is reported).

From the base, :func:`make_matrix` derives one matrix per horizon ``h``: the forward target
``y_t = log(mean RV over t+1..t+h)``, with ``t0`` = the decision timestamp and ``t1`` = the
decision timestamp ``h`` sessions later (so the walk-forward purge/embargo removes the ``h``-day
target overlap, exactly the triple-barrier ``t1`` discipline). The row gate requires the HAR
predictors and the target to be present; treatment-feature nulls are KEPT (LightGBM-native).

No network, no spend — reads only the local lakes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from glob import glob as _glob

import numpy as np
import polars as pl

from ..common.logging import get_logger
from ..data.store import DuckStore
from ..features.build import partition_glob as feature_partition_glob
from ..features.compute import feature_names
from ..features.config import FeatureConfig
from ..features.macro_features import compute_macro_features, macro_feature_names
from ..macro.config import MacroConfig
from ..macro.ingest import partition_glob as macro_partition_glob
from ..macro.ingest import read_macro_events
from ..sentiment.config import SentimentConfig
from ..sentiment.features import (
    attach_sentiment_asof,
    read_sentiment_scores,
    sentiment_feature_names,
)
from .config import VolatilityConfig
from .realized import daily_realized_variance, forward_log_rv, har_predictors

logger = get_logger(__name__)

_WIDE_START = datetime(2000, 1, 1, tzinfo=UTC)
_WIDE_END = datetime(2100, 1, 1, tzinfo=UTC)


@dataclass(frozen=True)
class DailyBase:
    """Per-symbol daily base: RV, HAR predictors, and as-of-``t`` treatment features."""

    symbol: str
    frame: pl.DataFrame  # session_date, t_close, rv, har_*, <treatment feature cols>
    har_cols: list[str]
    treat_cols: list[str]  # har_cols + price + macro + sentiment (the arm-T feature set)
    sentiment_coverage: float
    n_incomplete_sessions: int
    feature_blocks: dict[str, int]  # block -> n columns, for reporting


@dataclass(frozen=True)
class VolatilityMatrix:
    """Leak-free daily forecast matrix for one (symbol, horizon)."""

    symbol: str
    horizon: int
    x_har: np.ndarray  # (n, 3) HAR predictors only (arm B)
    x_treat: np.ndarray  # (n, 3+P) HAR + treatment feature blocks (arm T)
    y: np.ndarray  # (n,) forward log-RV target
    rv: np.ndarray  # (n,) daily RV at the decision day t (for the regime split)
    t0: np.ndarray  # (n,) decision timestamp (session close of day t)
    t1: np.ndarray  # (n,) decision timestamp of day t+h (drives purge/embargo)
    session_date: np.ndarray  # (n,) ET session date of t
    har_cols: list[str]
    treat_cols: list[str]
    sentiment_coverage: float = 0.0
    feature_blocks: dict[str, int] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return int(self.y.shape[0])


def _macro_available() -> bool:
    return bool(_glob(macro_partition_glob()))


def _asof_features(
    store: DuckStore, decisions: pl.DataFrame, symbol: str, feat_cols: list[str]
) -> pl.DataFrame:
    """Backward as-of attach of the per-minute feature lake at each decision ``t_close``.

    Mirrors the daily model's pushed-down attach: DuckDB projects only ``ts_event`` + ``feat_cols``
    (latest-ingest wins), then a polars ``join_asof`` (``feature.ts_event <= t_close``) brings the
    session-close snapshot over. Leak-free by ``strategy="backward"``. The matched feature
    timestamp is preserved as ``_feat_ts`` so the caller can enforce a freshness (same-session)
    check — a backward as-of would otherwise silently use a stale prior-session snapshot if a
    session's features were missing.
    """
    glob_str = feature_partition_glob(symbol)
    if not _glob(glob_str):
        raise ValueError(
            f"[{symbol}] no feature lake at data/features/symbol={symbol} — build it first:\n"
            "  uv run python -m options_system.features.build --symbols MES MNQ"
        )
    max_t = decisions["t_close"].max()
    feat_select = ", ".join(f'"{c}"' for c in feat_cols)
    feats = store.con.execute(
        f"""
        SELECT ts_event, {feat_select}
        FROM read_parquet('{glob_str}', hive_partitioning=false)
        WHERE ts_event <= ?
        QUALIFY row_number() OVER (PARTITION BY ts_event ORDER BY ts_ingest DESC) = 1
        ORDER BY ts_event
        """,
        [max_t],
    ).pl()
    left = decisions.sort("t_close")
    joined = left.join_asof(feats, left_on="t_close", right_on="ts_event", strategy="backward")
    return joined.rename({"ts_event": "_feat_ts"}) if "ts_event" in joined.columns else joined


def build_daily_base(
    symbol: str, vcfg: VolatilityConfig, *, store: DuckStore | None = None
) -> DailyBase:
    """Assemble the per-symbol daily base (RV + HAR predictors + as-of-``t`` treatment features)."""
    fcfg = FeatureConfig.load()
    own = store is None
    store = store or DuckStore()
    try:
        bars = store.get_bars(symbol, _WIDE_START, _WIDE_END, freq="1m", continuous=True)
        if bars.is_empty():
            raise ValueError(f"[{symbol}] no bars_1m; ingest 1-minute bars first")
        rv_frame, n_incomplete = daily_realized_variance(bars, fcfg.session, vcfg.rv)
        if rv_frame.height < 60:
            raise ValueError(
                f"[{symbol}] only {rv_frame.height} daily RV sessions — too few to proceed"
            )
        rv = rv_frame["rv"].to_numpy().astype(float)
        har = har_predictors(rv, tuple(vcfg.har.lags))
        har_cols = list(har.keys())
        base = rv_frame.with_columns([pl.Series(name, vals) for name, vals in har.items()])

        # --- treatment feature blocks, attached as-of the session close ---
        treat_cols: list[str] = list(har_cols)
        feature_blocks: dict[str, int] = {"har": len(har_cols)}
        decisions = base.select("session_date", "t_close")

        if vcfg.features.with_price:
            price_cols = list(feature_names(fcfg))
            attached = _asof_features(store, decisions, symbol, price_cols)
            # Freshness: the matched feature snapshot must be from the SAME ET session as the
            # decision close. A stale (cross-session) snapshot — from a missing/partial feature
            # build — has its price features NULLED (row kept; HAR + other blocks intact;
            # LightGBM-native NaN), never used as if current. Should be 0 (features come from the
            # same bars_1m); the guard is defensive against partial-build gaps.
            feat_date = pl.col("_feat_ts").dt.convert_time_zone(fcfg.session.tz).dt.date()
            attached = attached.with_columns((feat_date == pl.col("session_date")).alias("_fresh"))
            n_stale = int((~attached["_fresh"].fill_null(False)).sum())
            if n_stale:
                logger.warning(
                    f"[{symbol}] {n_stale} decision(s) had a stale cross-session feature "
                    "snapshot; nulling their price features"
                )
                attached = attached.with_columns(
                    pl.when(pl.col("_fresh")).then(pl.col(c)).otherwise(None).alias(c)
                    for c in price_cols
                )
            attached = attached.drop("_feat_ts", "_fresh")
            base = base.join(attached, on=["session_date", "t_close"], how="left")
            treat_cols += price_cols
            feature_blocks["price"] = len(price_cols)
            feature_blocks["price_stale_nulled"] = n_stale

        if vcfg.features.with_macro:
            # Macro is part of the FROZEN Phase-21 feature set — fail closed if absent so a
            # canonical verdict can never be silently built on a different (macro-less) feature
            # set. (To deliberately exclude macro, set features.with_macro=false in config.)
            if not _macro_available():
                raise ValueError(
                    f"[{symbol}] with_macro=True but the macro_events table is empty — the frozen "
                    "feature set requires it. Ingest it first:\n"
                    "  uv run python -m options_system.macro.ingest"
                )
            mcfg = MacroConfig.load()
            macro_cols = macro_feature_names(mcfg)
            events = read_macro_events(store=store)
            macro_df = compute_macro_features(base["t_close"], events, mcfg)
            base = base.hstack(macro_df.drop("t0"))
            treat_cols += macro_cols
            feature_blocks["macro"] = len(macro_cols)

        sentiment_coverage = 0.0
        if vcfg.features.with_sentiment:
            scfg = SentimentConfig.load()
            sent_cols = sentiment_feature_names(scfg)
            scored = read_sentiment_scores()
            attached = attach_sentiment_asof(
                base.select("t_close"), scored, scfg, time_col="t_close"
            )
            base = base.hstack(attached.drop("t_close"))
            treat_cols += sent_cols
            feature_blocks["sentiment"] = len(sent_cols)
            has_any_col = next((c for c in sent_cols if c.endswith("_has_any")), None)
            if has_any_col is not None and base.height:
                flags = base[has_any_col].fill_null(0).to_numpy().astype(float)
                sentiment_coverage = float(flags.mean()) if flags.size else 0.0

        logger.info(
            f"[{symbol}] daily base: {base.height} sessions "
            f"({n_incomplete} incomplete dropped), {len(treat_cols)} treat features, "
            f"sentiment coverage {sentiment_coverage:.3f}"
        )
        return DailyBase(
            symbol=symbol,
            frame=base,
            har_cols=har_cols,
            treat_cols=treat_cols,
            sentiment_coverage=sentiment_coverage,
            n_incomplete_sessions=n_incomplete,
            feature_blocks=feature_blocks,
        )
    finally:
        if own:
            store.close()


def make_matrix(base: DailyBase, horizon: int) -> VolatilityMatrix:
    """Derive the (symbol, horizon) matrix: forward target + ``[t0, t1]`` + leak-safe row gate.

    Row gate: the three HAR predictors present (22-day warmup) AND the forward target present (a
    full ``h``-day forward window exists). Treatment-feature nulls are KEPT. ``t1`` is the decision
    timestamp ``h`` sessions ahead, so the walk-forward purge removes the ``h``-day target overlap.
    """
    frame = base.frame.sort("session_date")
    rv = frame["rv"].to_numpy().astype(float)
    y = forward_log_rv(rv, horizon)
    t_close = frame["t_close"].to_numpy()
    # t1 = decision timestamp h sessions later (forward-window end); NaT-equivalent for last h rows.
    t1 = np.empty_like(t_close)
    t1[: t_close.size - horizon] = t_close[horizon:]
    t1[t_close.size - horizon :] = t_close[-1]

    har_ok = np.all(
        np.column_stack([np.isfinite(frame[c].to_numpy().astype(float)) for c in base.har_cols]),
        axis=1,
    )
    keep = har_ok & np.isfinite(y)
    idx = np.flatnonzero(keep)

    x_har = frame.select(base.har_cols).to_numpy()[idx]
    x_treat = frame.select(base.treat_cols).to_numpy()[idx]
    return VolatilityMatrix(
        symbol=base.symbol,
        horizon=horizon,
        x_har=x_har,
        x_treat=x_treat,
        y=y[idx],
        rv=rv[idx],
        t0=t_close[idx],
        t1=t1[idx],
        session_date=frame["session_date"].to_numpy()[idx],
        har_cols=list(base.har_cols),
        treat_cols=list(base.treat_cols),
        sentiment_coverage=base.sentiment_coverage,
        feature_blocks=dict(base.feature_blocks),
    )
