"""Causal feature engine — Polars-native, leakage-safe by construction.

Every feature value at time ``t`` depends only on bars with ``ts_event <= t``:
all windows are **trailing** (rolling/ewm/shift, never centered, never a negative
shift). Price-derived features are **degree-0 in the price scale** (returns,
ratios, normalized z-scores) so they are invariant to ratio back-adjustment —
truncating the raw history at ``t`` and rebuilding the continuous series only
multiplies the point-in-time series by a single global constant ``f_k``, which
cancels in any degree-0 feature. Level features (raw price, raw ATR points, raw
VWAP) are deliberately NOT emitted; the only level we ever touch, the raw
front-month price, is recovered as ``close / adj_factor`` and used solely inside
ratios. The truncation-invariance test (tests/test_features_leakage.py) proves
all of this.

Input: one symbol's **continuous** bars (already outright-only), sorted by
``ts_event`` ascending, with columns ``ts_event, open, high, low, close, volume,
session, contract_id, adj_factor``. Cross-asset features take the paired symbol's
continuous bars and attach them with a backward as-of join (contemporaneous-or-
earlier only).

Output: a frame keyed by ``ts_event`` with one column per feature plus
``session``, ``degraded`` and ``feature_version``. ``ts_ingest`` and ``symbol``
are added by the writer (features/build.py).
"""

from __future__ import annotations

import math

import polars as pl

from .config import FeatureConfig

_LN2 = math.log(2.0)
_TWO_PI = 2.0 * math.pi

# Columns the engine requires on the input continuous frame.
REQUIRED_INPUT = (
    "ts_event",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "session",
    "contract_id",
    "adj_factor",
)


def _wilder(expr: pl.Expr, n: int) -> pl.Expr:
    """Wilder's smoothing (RMA) = EMA with alpha = 1/n, no bias correction."""
    return expr.ewm_mean(alpha=1.0 / n, adjust=False, min_samples=n)


def feature_names(cfg: FeatureConfig) -> list[str]:
    """The exact feature columns the engine emits for ``cfg`` (excludes meta cols).

    Mirrors :func:`compute_features` one-to-one; the catalog test asserts the two
    agree, so this is the single source of truth for the emitted schema.
    """
    m, mr, v, vol, xa = (
        cfg.momentum,
        cfg.mean_reversion,
        cfg.volatility,
        cfg.volume,
        cfg.cross_asset,
    )
    names: list[str] = []
    names += [f"ret_{h}" for h in cfg.returns.horizons]
    names += [f"ema_slope_{w}" for w in m.ema_windows]
    names += [f"ema_dist_z_{w}" for w in m.ema_windows]
    names += ["macd", "macd_signal", "macd_hist"]
    names += [f"roc_{w}" for w in m.roc_windows]
    names += [f"adx_{m.adx_window}"]
    names += [f"rsi_{mr.rsi_window}"]
    names += [f"bb_pctb_{mr.bb_window}"]
    names += [f"zscore_{w}" for w in mr.zscore_windows]
    names += [f"rv_{w}" for w in v.rv_windows]
    names += [f"atr_pct_{v.atr_window}"]
    names += [f"parkinson_{v.parkinson_window}"]
    names += [f"gk_{v.gk_window}"]
    names += [f"vol_regime_{v.regime_window}"]
    names += [f"rvol_{w}" for w in vol.rvol_windows]
    names += ["rvol_tod"]
    names += [f"vol_z_{vol.zscore_window}"]
    names += [f"obv_norm_{vol.obv_window}"]
    names += ["vwap_dist"]
    names += ["mod_sin", "mod_cos", "dow_sin", "dow_cos"]
    names += ["mins_since_rth_open", "mins_to_rth_close"]
    names += ["xa_ret_spread", f"xa_ratio_z_{xa.ratio_zscore_window}", f"xa_corr_{xa.corr_window}"]
    return names


# --------------------------------------------------------------------------- #
# Stage builders. Each returns a list of (name, expr) added causally.          #
# --------------------------------------------------------------------------- #


def _with_base(df: pl.DataFrame, cfg: FeatureConfig) -> pl.DataFrame:
    """Intermediate columns the feature exprs depend on (logc, ret_1, tr, ET time)."""
    et = pl.col("ts_event").dt.convert_time_zone(cfg.session.tz)
    prev_close = pl.col("close").shift(1)
    tr = pl.max_horizontal(
        pl.col("high") - pl.col("low"),
        (pl.col("high") - prev_close).abs(),
        (pl.col("low") - prev_close).abs(),
    )
    df = df.with_columns(
        logc=pl.col("close").log(),
        raw_close=pl.col("close") / pl.col("adj_factor"),  # true recorded price (invariant)
        _tr=tr,
        _et=et,
    )
    minute_of_day = pl.col("_et").dt.hour() * 60 + pl.col("_et").dt.minute()
    # CME session/trade date: the overnight session (>= roll hour ET) belongs to
    # the NEXT trade date. Used to anchor session-reset VWAP.
    roll = cfg.session.session_roll_hour_et
    session_date = (
        pl.when(pl.col("_et").dt.hour() >= roll)
        .then(pl.col("_et").dt.date() + pl.duration(days=1))
        .otherwise(pl.col("_et").dt.date())
    )
    return df.with_columns(
        ret_1=(pl.col("logc") - pl.col("logc").shift(1)),
        minute_of_day=minute_of_day,
        dow=(pl.col("_et").dt.weekday() - 1),  # 0=Mon .. 6=Sun
        session_date=session_date,
    )


def _returns(cfg: FeatureConfig) -> list[pl.Expr]:
    return [
        (pl.col("logc") - pl.col("logc").shift(h)).alias(f"ret_{h}") for h in cfg.returns.horizons
    ]


def _momentum(cfg: FeatureConfig) -> list[pl.Expr]:
    m = cfg.momentum
    out: list[pl.Expr] = []
    for w in m.ema_windows:
        ema = pl.col("logc").ewm_mean(span=w, adjust=False, min_samples=w)
        # slope of the log-price EMA over `slope_lookback` bars (degree-0: log diff)
        out.append(((ema - ema.shift(m.slope_lookback)) / m.slope_lookback).alias(f"ema_slope_{w}"))
        # price distance from EMA, z-scored by the rolling std of that distance
        dist = pl.col("logc") - ema
        out.append(
            (dist / dist.rolling_std(window_size=w, min_samples=w, ddof=1)).alias(f"ema_dist_z_{w}")
        )
    fast, slow, signal = m.macd
    ema_f = pl.col("logc").ewm_mean(span=fast, adjust=False, min_samples=fast)
    ema_s = pl.col("logc").ewm_mean(span=slow, adjust=False, min_samples=slow)
    macd = ema_f - ema_s
    macd_signal = macd.ewm_mean(span=signal, adjust=False, min_samples=slow + signal)
    out += [
        macd.alias("macd"),
        macd_signal.alias("macd_signal"),
        (macd - macd_signal).alias("macd_hist"),
    ]
    out += [
        (pl.col("close") / pl.col("close").shift(w) - 1.0).alias(f"roc_{w}") for w in m.roc_windows
    ]
    return out


def _adx(cfg: FeatureConfig) -> list[pl.Expr]:
    n = cfg.momentum.adx_window
    up = pl.col("high").diff()
    down = -pl.col("low").diff()
    plus_dm = pl.when((up > down) & (up > 0)).then(up).otherwise(0.0)
    minus_dm = pl.when((down > up) & (down > 0)).then(down).otherwise(0.0)
    atr = _wilder(pl.col("_tr"), n)
    plus_di = 100.0 * _wilder(plus_dm, n) / atr
    minus_di = 100.0 * _wilder(minus_dm, n) / atr
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return [_wilder(dx, n).alias(f"adx_{n}")]


def _mean_reversion(cfg: FeatureConfig) -> list[pl.Expr]:
    mr = cfg.mean_reversion
    out: list[pl.Expr] = []
    # RSI (Wilder)
    delta = pl.col("close").diff()
    gain = pl.when(delta > 0).then(delta).otherwise(0.0)
    loss = pl.when(delta < 0).then(-delta).otherwise(0.0)
    rs = _wilder(gain, mr.rsi_window) / _wilder(loss, mr.rsi_window)
    out.append((100.0 - 100.0 / (1.0 + rs)).alias(f"rsi_{mr.rsi_window}"))
    # Bollinger %B
    sma = pl.col("close").rolling_mean(window_size=mr.bb_window, min_samples=mr.bb_window)
    sd = pl.col("close").rolling_std(window_size=mr.bb_window, min_samples=mr.bb_window, ddof=0)
    lower = sma - mr.bb_std * sd
    upper = sma + mr.bb_std * sd
    out.append(((pl.col("close") - lower) / (upper - lower)).alias(f"bb_pctb_{mr.bb_window}"))
    # rolling z-score of close
    for w in mr.zscore_windows:
        mean = pl.col("close").rolling_mean(window_size=w, min_samples=w)
        std = pl.col("close").rolling_std(window_size=w, min_samples=w, ddof=1)
        out.append(((pl.col("close") - mean) / std).alias(f"zscore_{w}"))
    return out


def _volatility(cfg: FeatureConfig) -> list[pl.Expr]:
    v = cfg.volatility
    out: list[pl.Expr] = []
    for w in v.rv_windows:
        out.append(
            pl.col("ret_1").rolling_std(window_size=w, min_samples=w, ddof=1).alias(f"rv_{w}")
        )
    atr = _wilder(pl.col("_tr"), v.atr_window)
    out.append((atr / pl.col("close")).alias(f"atr_pct_{v.atr_window}"))
    # Parkinson (high/low range vol)
    hl = (pl.col("high") / pl.col("low")).log()
    park = (
        (hl.pow(2)).rolling_mean(window_size=v.parkinson_window, min_samples=v.parkinson_window)
        / (4.0 * _LN2)
    ).sqrt()
    out.append(park.alias(f"parkinson_{v.parkinson_window}"))
    # Garman-Klass
    co = (pl.col("close") / pl.col("open")).log()
    gk_term = 0.5 * hl.pow(2) - (2.0 * _LN2 - 1.0) * co.pow(2)
    gk = (
        gk_term.rolling_mean(window_size=v.gk_window, min_samples=v.gk_window)
        .clip(lower_bound=0.0)
        .sqrt()
    )
    out.append(gk.alias(f"gk_{v.gk_window}"))
    return out


def _vol_regime(cfg: FeatureConfig) -> list[pl.Expr]:
    v = cfg.volatility
    rv = pl.col("ret_1").rolling_std(
        window_size=v.regime_window, min_samples=v.regime_window, ddof=1
    )
    median = rv.rolling_median(window_size=v.regime_baseline, min_samples=v.regime_window)
    return [(rv / median).alias(f"vol_regime_{v.regime_window}")]


def _volume(cfg: FeatureConfig) -> list[pl.Expr]:
    vol = cfg.volume
    out: list[pl.Expr] = []
    for w in vol.rvol_windows:
        out.append(
            (pl.col("volume") / pl.col("volume").rolling_mean(window_size=w, min_samples=w)).alias(
                f"rvol_{w}"
            )
        )
    # time-of-day baseline: mean volume at the SAME minute-of-day over the prior N
    # days (shift(1) excludes the current day) — strips intraday seasonality.
    tod_base = (
        pl.col("volume")
        .rolling_mean(window_size=vol.tod_baseline_days, min_samples=1)
        .shift(1)
        .over("minute_of_day", order_by="ts_event")
    )
    out.append((pl.col("volume") / tod_base).alias("rvol_tod"))
    vmean = pl.col("volume").rolling_mean(
        window_size=vol.zscore_window, min_samples=vol.zscore_window
    )
    vstd = pl.col("volume").rolling_std(
        window_size=vol.zscore_window, min_samples=vol.zscore_window, ddof=1
    )
    out.append(((pl.col("volume") - vmean) / vstd).alias(f"vol_z_{vol.zscore_window}"))
    # normalized signed-volume fraction (bounded OBV-style flow), in [-1, 1]
    signed = pl.col("ret_1").sign().fill_null(0.0) * pl.col("volume")
    num = signed.rolling_sum(window_size=vol.obv_window, min_samples=vol.obv_window)
    den = pl.col("volume").rolling_sum(window_size=vol.obv_window, min_samples=vol.obv_window)
    out.append((num / den).alias(f"obv_norm_{vol.obv_window}"))
    return out


def _vwap_dist() -> list[pl.Expr]:
    typical = (pl.col("high") + pl.col("low") + pl.col("close")) / 3.0
    cum_pv = (typical * pl.col("volume")).cum_sum().over("session_date", order_by="ts_event")
    cum_v = pl.col("volume").cum_sum().over("session_date", order_by="ts_event")
    vwap = cum_pv / cum_v
    return [(pl.col("close") / vwap - 1.0).alias("vwap_dist")]


def _time(cfg: FeatureConfig) -> list[pl.Expr]:
    mod = pl.col("minute_of_day")
    return [
        (mod * (_TWO_PI / 1440.0)).sin().alias("mod_sin"),
        (mod * (_TWO_PI / 1440.0)).cos().alias("mod_cos"),
        (pl.col("dow") * (_TWO_PI / 7.0)).sin().alias("dow_sin"),
        (pl.col("dow") * (_TWO_PI / 7.0)).cos().alias("dow_cos"),
        (mod - cfg.session.rth_open_min).alias("mins_since_rth_open"),
        (cfg.session.rth_close_min - mod).alias("mins_to_rth_close"),
    ]


def _cross_asset(
    df: pl.DataFrame, other: pl.DataFrame | None, cfg: FeatureConfig, symbol: str
) -> pl.DataFrame:
    xa = cfg.cross_asset
    rz, cw = xa.ratio_zscore_window, xa.corr_window
    names = (f"xa_ratio_z_{rz}", f"xa_corr_{cw}")
    if other is None:
        return df.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("xa_ret_spread"),
            pl.lit(None, dtype=pl.Float64).alias(names[0]),
            pl.lit(None, dtype=pl.Float64).alias(names[1]),
        )
    # other's contemporaneous-or-earlier ret + raw price via backward as-of join
    other_sel = (
        other.sort("ts_event")
        .with_columns(
            o_logc=pl.col("close").log(),
            o_raw=pl.col("close") / pl.col("adj_factor"),
        )
        .with_columns(o_ret_1=(pl.col("o_logc") - pl.col("o_logc").shift(1)))
        .select("ts_event", "o_ret_1", "o_raw")
    )
    df = df.sort("ts_event").join_asof(other_sel, on="ts_event", strategy="backward")
    # ratio is always MNQ/MES (= pair[1]/pair[0]); orient by which symbol we are on
    this_is_num = symbol == xa.pair[1]
    ratio = (
        (pl.col("raw_close") / pl.col("o_raw"))
        if this_is_num
        else (pl.col("o_raw") / pl.col("raw_close"))
    )
    spread = (
        (pl.col("ret_1") - pl.col("o_ret_1"))
        if this_is_num
        else (pl.col("o_ret_1") - pl.col("ret_1"))
    )
    rmean = ratio.rolling_mean(window_size=rz, min_samples=rz)
    rstd = ratio.rolling_std(window_size=rz, min_samples=rz, ddof=1)
    df = df.with_columns(
        spread.alias("xa_ret_spread"),
        ((ratio - rmean) / rstd).alias(names[0]),
        pl.rolling_corr(pl.col("ret_1"), pl.col("o_ret_1"), window_size=cw, min_samples=cw).alias(
            names[1]
        ),
    )
    return df.drop("o_ret_1", "o_raw")


def compute_features(
    df: pl.DataFrame,
    cfg: FeatureConfig,
    *,
    symbol: str,
    other: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Compute the full causal feature set for one symbol's continuous bars.

    ``df`` / ``other`` are continuous (outright-only) bar frames. Returns a frame
    of ``ts_event, session, <features...>, degraded, feature_version``.
    """
    missing = [c for c in REQUIRED_INPUT if c not in df.columns]
    if missing:
        raise ValueError(f"compute_features: input missing columns {missing}")
    # Spreads must never reach the feature layer.
    if df.select(pl.col("contract_id").str.contains("-").any()).item():
        raise ValueError("compute_features: spread contract_id (containing '-') in input")
    if df.is_empty():
        meta = ["ts_event", "session", "degraded", "feature_version"]
        return (
            pl.DataFrame(schema={**{c: pl.Float64 for c in feature_names(cfg)}})
            .select(*[pl.lit(None).alias(c) for c in meta], pl.all())
            .clear()
        )

    df = df.sort("ts_event").pipe(_with_base, cfg)
    df = df.with_columns(*_returns(cfg))  # ret_h (ret_1 already exists; recompute is identical)
    df = df.with_columns(
        *_momentum(cfg),
        *_adx(cfg),
        *_mean_reversion(cfg),
        *_volatility(cfg),
        *_volume(cfg),
        *_vwap_dist(),
        *_time(cfg),
    )
    df = df.with_columns(*_vol_regime(cfg))  # depends on ret_1 only; own with_columns for clarity
    df = _cross_asset(df, other, cfg, symbol)

    # warmup + degraded flags (both invariant to truncation: depend on bars <= t)
    max_w = cfg.max_window()
    degraded_dates = sorted(cfg.degraded_day_set())
    df = df.with_columns(
        (
            (pl.int_range(0, pl.len()) < max_w) | pl.col("ts_event").dt.date().is_in(degraded_dates)
        ).alias("degraded"),
        pl.lit(cfg.feature_version).alias("feature_version"),
    )
    return df.select("ts_event", "session", *feature_names(cfg), "degraded", "feature_version")
