"""Causal volatility estimator + event sampler.

Two jobs, both leak-free by construction (every value at bar ``t`` uses only
bars ``<= t``):

1. :func:`compute_sigma` — σ_t, the bar-level return volatility, scaled to a
   session horizon. It SETS the barrier widths and the CUSUM threshold, so it
   must be causal: an EWM standard deviation of 1-bar log returns (a trailing
   recursion, ``adjust=False``) times ``sqrt(barrier_horizon_bars)``. Scaling a
   clean 1-bar σ by √H keeps the estimator simple and transparent while making
   "1σ" a meaningful intraday move instead of a one-minute wiggle.

2. :func:`sample_events` — pick the event times ``t0`` at which labels are set.
   Default is the López de Prado **symmetric CUSUM filter** (:func:`cusum_events`):
   accumulate signed returns and emit an event whenever the cumulative move
   since the last event exceeds a σ-scaled threshold, then reset. This collapses
   long stretches of noise into the handful of bars where something actually
   happened, so labels are far fewer than bars and not redundantly overlapping.
   A deterministic ``grid`` alternative (every k bars) is also provided.

σ is degree-0 in the price scale (it is built from log-return *differences*), so
like the features it is invariant to ratio back-adjustment.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl

from .config import LabelConfig

# Columns this module needs on the input continuous frame.
REQUIRED_INPUT = ("ts_event", "close")


def compute_sigma(df: pl.DataFrame, cfg: LabelConfig) -> pl.DataFrame:
    """Attach ``ret_1`` (1-bar log return) and ``sigma`` (σ_scaled) to ``df``.

    ``df`` must be one symbol's continuous bars sorted by ``ts_event`` ascending.
    ``sigma`` is the causal EWM std of 1-bar log returns scaled to the barrier
    horizon; it is ``null`` during warmup (first ``min_samples`` bars).
    """
    v = cfg.volatility
    logc = pl.col("close").log()
    ret_1 = logc - logc.shift(1)
    sigma_bar = ret_1.ewm_std(span=v.ewm_span, adjust=False, bias=False, min_samples=v.min_samples)
    scale = math.sqrt(v.barrier_horizon_bars)
    return df.with_columns(ret_1=ret_1, sigma=(sigma_bar * scale))


def cusum_events(rets: np.ndarray, thresh: np.ndarray) -> np.ndarray:
    """Symmetric CUSUM filter (López de Prado, AFML §2.5.2.1).

    Maintains a positive and a negative running sum of ``rets``; emits an event
    index whenever either crosses ``±thresh`` (per-bar, time-varying), then
    resets that accumulator. Bars where ``rets`` or ``thresh`` is NaN / non-finite
    / ``<= 0`` are treated as warmup: the accumulators reset and no event fires.

    Returns the sorted array of event bar indices. Purely causal — the decision
    at bar ``i`` depends only on returns up to ``i``.
    """
    if rets.shape != thresh.shape:
        raise ValueError("rets and thresh must have the same shape")
    s_pos = 0.0
    s_neg = 0.0
    out: list[int] = []
    r_list = rets.tolist()  # python floats: ~3x faster than per-element numpy access
    h_list = thresh.tolist()
    for i, (r, h) in enumerate(zip(r_list, h_list, strict=True)):
        if not (math.isfinite(r) and math.isfinite(h)) or h <= 0.0:
            s_pos = 0.0
            s_neg = 0.0
            continue
        s_pos = max(0.0, s_pos + r)
        s_neg = min(0.0, s_neg + r)
        if s_pos >= h:
            out.append(i)
            s_pos = 0.0
        elif s_neg <= -h:
            out.append(i)
            s_neg = 0.0
    return np.asarray(out, dtype=np.int64)


def grid_events(n: int, step: int, start: int) -> np.ndarray:
    """Deterministic grid sampler: event bar indices ``start, start+step, ...``."""
    if step <= 0:
        raise ValueError("grid step must be positive")
    start = max(start, 0)
    return np.arange(start, n, step, dtype=np.int64)


def sample_events(df: pl.DataFrame, cfg: LabelConfig) -> np.ndarray:
    """Return the integer bar indices of sampled events for ``df``.

    ``df`` must already carry ``ret_1`` and ``sigma`` (run :func:`compute_sigma`
    first). Warmup bars (``sigma`` null) never produce events. For ``cusum`` the
    threshold is ``cusum_mult · sigma``; for ``grid`` events start after warmup.
    """
    missing = [c for c in ("ret_1", "sigma") if c not in df.columns]
    if missing:
        raise ValueError(f"sample_events: run compute_sigma first; missing {missing}")
    n = df.height
    if n == 0:
        return np.asarray([], dtype=np.int64)

    sigma = df["sigma"].to_numpy()
    # first bar where sigma is defined (end of warmup)
    valid = np.flatnonzero(np.isfinite(sigma) & (sigma > 0))
    if valid.size == 0:
        return np.asarray([], dtype=np.int64)
    warmup_end = int(valid[0])

    if cfg.events.method == "grid":
        return grid_events(n, cfg.events.grid_step_bars, warmup_end)

    rets = np.nan_to_num(df["ret_1"].to_numpy(), nan=0.0)
    thresh = sigma * cfg.events.cusum_mult
    return cusum_events(rets, thresh)
