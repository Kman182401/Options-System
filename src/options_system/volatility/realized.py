"""Realized-variance estimation + HAR predictors + the forward target (pure, leak-safe).

The Phase-21 target is daily realized variance, estimated from the 1-minute ``bars_1m`` lake with
the **noise-reduced 5-minute sub-sampled** estimator (Zhang-Aït-Sahalia subsampling): within each
RTH session, sum squared 5-minute log returns on each of five offset grids (origins 0,1,2,3,4) and
**average the five grids**. 5-minute sampling is the literature standard (microstructure-noise vs
estimator-accuracy trade-off); the five offset grids cut the estimator's own variance.

From the daily RV series this module also builds the three **causal HAR predictors** (Corsi 2009:
log of the daily / weekly / monthly trailing RV, all using info up to and including the decision
day ``t``) and the **forward target** ``y_t = log( mean RV over the next h trading days )`` with its
``[t0, t1]`` interval recorded for purge/embargo.

Every function here is pure (numpy / polars in, frame/arrays out) and unit-tested for causality
(no future bar enters a decision-day quantity) and determinism. No I/O, no model.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl

from ..features.config import SessionCfg
from .config import RvCfg


def _rth(bars: pl.DataFrame, scfg: SessionCfg) -> pl.DataFrame:
    """Filter to regular-trading-hours weekday bars and tag each with its ET session date.

    Uses the project's :class:`SessionCfg` (tz + ``rth_open_min``/``rth_close_min``), the same
    RTH window the microstructure layer uses. Polars ``weekday()`` is 1=Mon..7=Sun ⇒ Mon-Fri is
    ``<= 5``. The session date is the ET calendar date (RTH is intraday, so no overnight roll).
    """
    et = pl.col("ts_event").dt.convert_time_zone(scfg.tz)
    # cast to Int32 BEFORE arithmetic — dt.hour() is Int8 and hour*60 overflows it.
    mod = et.dt.hour().cast(pl.Int32) * 60 + et.dt.minute().cast(pl.Int32)
    return (
        bars.with_columns(
            et.dt.date().alias("session_date"),
            et.dt.weekday().alias("_wd"),
            mod.alias("_mod"),
        )
        .filter(
            (pl.col("_wd") <= 5)
            & (pl.col("_mod") >= scfg.rth_open_min)
            & (pl.col("_mod") < scfg.rth_close_min)
        )
        .sort("ts_event")
    )


def _subsampled_rv(log_close: np.ndarray, sampling: int, n_grids: int) -> float | None:
    """Average of the ``n_grids`` offset-grid 5-minute realized variances for one session.

    Grid ``g`` samples positions ``g, g+sampling, g+2·sampling, …`` of the (time-ordered) 1-minute
    log-closes; its RV is the sum of squared consecutive differences. Returns the mean over grids,
    or ``None`` if no grid yields at least one return.
    """
    grid_rvs: list[float] = []
    for off in range(n_grids):
        idx = np.arange(off, log_close.size, sampling)
        if idx.size < 2:
            continue
        r = np.diff(log_close[idx])
        grid_rvs.append(float(np.dot(r, r)))
    if not grid_rvs:
        return None
    return float(np.mean(grid_rvs))


def daily_realized_variance(
    bars: pl.DataFrame, scfg: SessionCfg, rvcfg: RvCfg
) -> tuple[pl.DataFrame, int]:
    """Daily RV per RTH session for one symbol → ``(frame, n_incomplete_dropped)``.

    ``bars`` must carry ``ts_event`` (UTC) and ``close`` (the continuous back-adjusted close — log
    returns cancel the adjustment factor, so it is the correct series for RV). Returns a frame with
    ``session_date`` (ET date), ``rv`` (the 5-grid-averaged daily realized variance), ``t_close``
    (the session's last RTH bar ts_event — the decision timestamp), and ``n_bars``; plus the count
    of sessions dropped for having fewer than ``min_5min_returns_per_session`` base-grid returns.
    """
    rth = _rth(bars, scfg)
    sm = rvcfg.sampling_minutes
    n_grids = rvcfg.n_subsample_grids
    min_ret = rvcfg.min_5min_returns_per_session

    rows: list[dict] = []
    dropped = 0
    for (sd,), g in rth.group_by(["session_date"], maintain_order=True):
        close = g["close"].to_numpy().astype(float)
        if close.size == 0 or np.any(close <= 0):
            dropped += 1
            continue
        # Completeness on the base grid (origin 0): need enough 5-min returns.
        base_returns = max(0, int(np.arange(0, close.size, sm).size) - 1)
        if base_returns < min_ret:
            dropped += 1
            continue
        rv = _subsampled_rv(np.log(close), sm, n_grids)
        if rv is None or not math.isfinite(rv) or rv <= 0.0:
            dropped += 1
            continue
        rows.append(
            {
                "session_date": sd,
                "rv": rv,
                "t_close": g["ts_event"][-1],  # polars-native python datetime (tz-aware)
                "n_bars": int(close.size),
            }
        )
    schema = {
        "session_date": pl.Date,
        "rv": pl.Float64,
        "t_close": pl.Datetime("us", "UTC"),
        "n_bars": pl.Int64,
    }
    if not rows:
        return pl.DataFrame(schema=schema), dropped
    return pl.DataFrame(rows, schema=schema).sort("session_date"), dropped


def _trailing_log_mean(rv: np.ndarray, window: int) -> np.ndarray:
    """Log of the trailing-``window`` (inclusive of ``t``) mean of ``rv``; NaN before warmup."""
    n = rv.size
    out = np.full(n, np.nan)
    csum = np.concatenate([[0.0], np.cumsum(rv)])  # csum[i] = sum(rv[:i])
    for i in range(n):
        if i + 1 >= window:
            s = csum[i + 1] - csum[i + 1 - window]
            out[i] = math.log(s / window)
    return out


def har_predictors(rv: np.ndarray, lags: tuple[int, ...] = (1, 5, 22)) -> dict[str, np.ndarray]:
    """The three causal HAR predictors at each decision day ``t`` (info up to and incl. ``t``).

    ``lags = (1, 5, 22)`` (Corsi daily/weekly/monthly): daily = ``log(rv_t)``; weekly =
    ``log(mean rv over last 5 days)``; monthly = ``log(mean rv over last 22 days)``. All trailing
    and inclusive of ``t`` (``rv_t`` is known at the close of session ``t``), so they never read the
    future. NaN during the 22-day warmup (LightGBM-native; HAR rows are gated to require all three).
    """
    rv = np.asarray(rv, dtype=float)
    d, w, m = lags
    return {
        f"har_log_rv_d{d}": _trailing_log_mean(rv, d),
        f"har_log_rv_w{w}": _trailing_log_mean(rv, w),
        f"har_log_rv_m{m}": _trailing_log_mean(rv, m),
    }


def forward_log_rv(rv: np.ndarray, h: int) -> np.ndarray:
    """The forward target ``y_t = log( mean rv over days t+1 .. t+h )``; NaN where incomplete.

    Strictly future (excludes day ``t``), so it never overlaps the predictors at ``t``. Defined for
    ``t`` with ``t + h <= N − 1`` (a full forward window exists); the last ``h`` days are NaN.
    """
    rv = np.asarray(rv, dtype=float)
    n = rv.size
    out = np.full(n, np.nan)
    csum = np.concatenate([[0.0], np.cumsum(rv)])  # csum[i] = sum(rv[:i])
    for t in range(n):
        if t + h <= n - 1:
            s = csum[t + h + 1] - csum[t + 1]  # sum rv[t+1 .. t+h]
            out[t] = math.log(s / h)
    return out


def daily_rth_log_return(bars: pl.DataFrame, scfg: SessionCfg) -> pl.DataFrame:
    """Within-session RTH (open-to-close) log return per session → ``(session_date, ret)``.

    ``ret_t = log(last RTH close) − log(first RTH close)`` within session ``t`` — the cumulative
    intraday RTH move. It is computed on **exactly the RTH window the RV target uses**, so it
    **excludes the overnight/weekend gap** (a close-to-close return would not, mis-scaling a GARCH
    benchmark vs the RTH-only RV target). Used **only** by the Phase-23 GARCH(1,1) benchmark
    (which needs a return series the RV target does not provide). Same RTH sessionization as the RV
    estimator, so it aligns 1:1 with the daily RV sessions and every session is self-contained
    (no null first row). Pure (bars in, frame out); no model, no I/O.
    """
    rth = _rth(bars, scfg)
    sess = (
        rth.group_by(["session_date"], maintain_order=True)
        .agg(
            pl.col("close").first().alias("c_open"),
            pl.col("close").last().alias("c_close"),
        )
        .sort("session_date")
    )
    return sess.with_columns(
        (pl.col("c_close").log() - pl.col("c_open").log()).alias("ret")
    ).select("session_date", "ret")
