"""Causal TA engine (feature_version = v2) — Polars-native, leakage-safe.

Every feature value at time ``t`` depends only on bars with ``ts_event <= t``: all
windows are **trailing** (rolling/ewm/shift, never centered, never a negative
shift). Every emitted feature is **degree-0 in the price scale** — it is either a
ratio of price differences (Stochastic, CCI, MFI, Vortex) or a difference of
log-prices (TRIX). Under ratio back-adjustment the whole point-in-time series is
multiplied by a single global constant ``f_k`` (or, in log space, shifted by
``ln(f_k)``), which cancels in every feature here. So truncating the raw history
at ``t`` and rebuilding the continuous series leaves each value unchanged — the
truncation-invariance test (tests/test_ta_leakage.py) proves it.

This layer is **additive and isolated**: it deliberately does NOT duplicate the
v1 price layer's RSI / MACD / ADX / Bollinger / OBV / z-scores. It adds a curated
starter set of classic oscillators (Stochastic %K/%D, CCI, MFI, Vortex VI+/VI-,
TRIX), all ``ta_``-namespaced so they never collide with v1 or the macro layer.

Input: one symbol's **continuous** bars (already outright-only), with columns
``ts_event, high, low, close, volume, session, contract_id``. Output: a frame
keyed by ``ts_event`` with one column per feature plus ``session``, ``degraded``
and ``ta_feature_version``. ``ts_ingest`` and ``symbol`` are added by the writer
(ta/build.py).
"""

from __future__ import annotations

import polars as pl

from .config import TaConfig

# Columns the engine requires on the input continuous frame.
REQUIRED_INPUT = (
    "ts_event",
    "high",
    "low",
    "close",
    "volume",
    "session",
    "contract_id",
)


def ta_feature_names(cfg: TaConfig) -> list[str]:
    """The exact feature columns the engine emits for ``cfg`` (excludes meta cols).

    Mirrors :func:`compute_ta` one-to-one; the catalog test asserts the two agree,
    so this is the single source of truth for the emitted schema.
    """
    s, c, mf, vx, tx = cfg.stoch, cfg.cci, cfg.mfi, cfg.vortex, cfg.trix
    names: list[str] = []
    names += [f"ta_stoch_k_{s.k_window}", f"ta_stoch_d_{s.d_smooth}"]
    names += [f"ta_cci_{c.window}"]
    names += [f"ta_mfi_{mf.window}"]
    names += [f"ta_vi_plus_{vx.window}", f"ta_vi_minus_{vx.window}"]
    names += [f"ta_trix_{tx.window}"]
    return names


# --------------------------------------------------------------------------- #
# Stage builders. Each returns a list of named, trailing-only exprs.           #
# --------------------------------------------------------------------------- #


def _with_base(df: pl.DataFrame) -> pl.DataFrame:
    """Intermediate columns the indicator exprs depend on (logc, typical price, TR)."""
    prev_close = pl.col("close").shift(1)
    tr = pl.max_horizontal(
        pl.col("high") - pl.col("low"),
        (pl.col("high") - prev_close).abs(),
        (pl.col("low") - prev_close).abs(),
    )
    return df.with_columns(
        logc=pl.col("close").log(),
        _tp=(pl.col("high") + pl.col("low") + pl.col("close")) / 3.0,
        _tr=tr,
    )


def _stoch(cfg: TaConfig) -> list[pl.Expr]:
    """Stochastic %K (position of close in the trailing high/low range) and %D (SMA of %K)."""
    k, d = cfg.stoch.k_window, cfg.stoch.d_smooth
    hh = pl.col("high").rolling_max(window_size=k, min_samples=k)
    ll = pl.col("low").rolling_min(window_size=k, min_samples=k)
    rng = hh - ll
    # Guard the flat-bar case (high-max == low-min) so we emit null, never inf.
    stoch_k = pl.when(rng > 0).then(100.0 * (pl.col("close") - ll) / rng).otherwise(None)
    stoch_d = stoch_k.rolling_mean(window_size=d, min_samples=d)
    return [stoch_k.alias(f"ta_stoch_k_{k}"), stoch_d.alias(f"ta_stoch_d_{d}")]


def _cci(cfg: TaConfig) -> list[pl.Expr]:
    """Commodity Channel Index: typical-price deviation from its SMA over the mean abs deviation.

    The mean absolute deviation is computed as the trailing mean of ``|tp - SMA(tp)|``
    (a fully vectorized, causal variant of the textbook deviation-from-window-mean).
    """
    n = cfg.cci.window
    tp = pl.col("_tp")
    sma = tp.rolling_mean(window_size=n, min_samples=n)
    mad = (tp - sma).abs().rolling_mean(window_size=n, min_samples=n)
    cci = pl.when(mad > 0).then((tp - sma) / (0.015 * mad)).otherwise(None)
    return [cci.alias(f"ta_cci_{n}")]


def _mfi(cfg: TaConfig) -> list[pl.Expr]:
    """Money Flow Index: the volume-weighted RSI on typical price, in [0, 100]."""
    n = cfg.mfi.window
    tp = pl.col("_tp")
    raw_money_flow = tp * pl.col("volume")
    direction = tp.diff()
    pos = pl.when(direction > 0).then(raw_money_flow).otherwise(0.0)
    neg = pl.when(direction < 0).then(raw_money_flow).otherwise(0.0)
    pos_sum = pos.rolling_sum(window_size=n, min_samples=n)
    neg_sum = neg.rolling_sum(window_size=n, min_samples=n)
    total = pos_sum + neg_sum
    mfi = pl.when(total > 0).then(100.0 * pos_sum / total).otherwise(None)
    return [mfi.alias(f"ta_mfi_{n}")]


def _vortex(cfg: TaConfig) -> list[pl.Expr]:
    """Vortex indicator VI+ / VI- = summed directional movement over the summed true range."""
    n = cfg.vortex.window
    vm_plus = (pl.col("high") - pl.col("low").shift(1)).abs()
    vm_minus = (pl.col("low") - pl.col("high").shift(1)).abs()
    tr_sum = pl.col("_tr").rolling_sum(window_size=n, min_samples=n)
    vip = (
        pl.when(tr_sum > 0)
        .then(vm_plus.rolling_sum(window_size=n, min_samples=n) / tr_sum)
        .otherwise(None)
    )
    vim = (
        pl.when(tr_sum > 0)
        .then(vm_minus.rolling_sum(window_size=n, min_samples=n) / tr_sum)
        .otherwise(None)
    )
    return [vip.alias(f"ta_vi_plus_{n}"), vim.alias(f"ta_vi_minus_{n}")]


def _trix(cfg: TaConfig) -> list[pl.Expr]:
    """TRIX: 1-bar change of the triple-EWM of log(close) (smoothed rate of change).

    Computed on log price so a global ratio back-adjustment (a constant ``ln(f_k)``
    added to every bar) propagates exactly through the three affine EWMs and cancels
    in the final 1-bar difference — degree-0 by construction.
    """
    n = cfg.trix.window
    ema1 = pl.col("logc").ewm_mean(span=n, adjust=False, min_samples=n)
    ema2 = ema1.ewm_mean(span=n, adjust=False, min_samples=n)
    ema3 = ema2.ewm_mean(span=n, adjust=False, min_samples=n)
    return [(ema3 - ema3.shift(1)).alias(f"ta_trix_{n}")]


def compute_ta(df: pl.DataFrame, cfg: TaConfig) -> pl.DataFrame:
    """Compute the causal TA feature set for one symbol's continuous bars.

    ``df`` is a continuous (outright-only) bar frame. Returns a frame of
    ``ts_event, session, <features...>, degraded, ta_feature_version``.
    """
    missing = [c for c in REQUIRED_INPUT if c not in df.columns]
    if missing:
        raise ValueError(f"compute_ta: input missing columns {missing}")
    # Spreads must never reach the feature layer.
    if df.select(pl.col("contract_id").str.contains("-").any()).item():
        raise ValueError("compute_ta: spread contract_id (containing '-') in input")
    if df.is_empty():
        meta = ["ts_event", "session", "degraded", "ta_feature_version"]
        return (
            pl.DataFrame(schema={c: pl.Float64 for c in ta_feature_names(cfg)})
            .select(*[pl.lit(None).alias(c) for c in meta], pl.all())
            .clear()
        )

    df = df.sort("ts_event").pipe(_with_base)
    df = df.with_columns(
        *_stoch(cfg),
        *_cci(cfg),
        *_mfi(cfg),
        *_vortex(cfg),
        *_trix(cfg),
    )

    # warmup + degraded flags (both invariant to truncation: depend on bars <= t)
    max_w = cfg.max_window()
    degraded_dates = sorted(cfg.degraded_day_set())
    df = df.with_columns(
        (
            (pl.int_range(0, pl.len()) < max_w) | pl.col("ts_event").dt.date().is_in(degraded_dates)
        ).alias("degraded"),
        pl.lit(cfg.ta_feature_version).alias("ta_feature_version"),
    )
    return df.select(
        "ts_event", "session", *ta_feature_names(cfg), "degraded", "ta_feature_version"
    )
