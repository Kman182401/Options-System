"""Forecast-comparison statistics — QLIKE loss and the Diebold-Mariano test.

The Phase-4 :mod:`options_system.validation.stats` module judges *classifiers / signals*
(PSR / DSR / PBO). This sibling judges *point forecasts*: it provides the QLIKE loss (the
literature's preferred volatility-forecast loss) and the Diebold-Mariano (1995) test for whether
one forecaster is significantly more accurate than another.

Two correctness requirements baked in here, both integrity-critical for an honest verdict:

* **QLIKE on the variance scale** — ``QLIKE(σ², RV) = RV/σ² − ln(RV/σ²) − 1`` (Patton 2011). It is
  the loss with the highest power in the DM test and is robust to the noise in the realized-variance
  proxy. It is minimized (in expectation) by the true conditional variance, so comparing two
  forecasters under QLIKE is a fair, proxy-robust accuracy test.
* **HAC + small-sample-corrected DM** — multi-day-overlapping forecast targets autocorrelate the
  per-period loss differential, so a naive (i.i.d.) DM variance *overstates* significance. We use a
  Newey-West (Bartlett) long-run variance with lag ``L`` (set to the overlap ``h − 1`` by callers)
  and the Harvey-Leybourne-Newbold (1997) small-sample correction, comparing the corrected statistic
  to a Student-t with ``n − 1`` df. Skipping this would manufacture false significance.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def qlike_from_variance(rv: np.ndarray, var_pred: np.ndarray) -> np.ndarray:
    """Per-period QLIKE on the variance scale: ``rv/var_pred − ln(rv/var_pred) − 1``.

    ``rv`` is the realized variance, ``var_pred`` the predicted variance (both > 0). Lower is
    better; it is 0 iff ``var_pred == rv`` and strictly positive otherwise (convex in the ratio).
    """
    rv = np.asarray(rv, dtype=float)
    var_pred = np.asarray(var_pred, dtype=float)
    if np.any(rv <= 0) or np.any(var_pred <= 0):
        raise ValueError("QLIKE requires strictly positive rv and var_pred (variance scale)")
    ratio = rv / var_pred
    return ratio - np.log(ratio) - 1.0


def qlike_from_log(y_true_log: np.ndarray, forecast_log: np.ndarray) -> np.ndarray:
    """Per-period QLIKE when realized and forecast are in **log-variance** space.

    With ``d = log(rv) − log(var_pred)``, ``QLIKE = exp(d) − d − 1`` — identical to
    :func:`qlike_from_variance` but computed from log inputs (the scale the models predict on).
    The exponent is clipped to a safe range so a wildly-off early forecast cannot overflow.
    """
    d = np.asarray(y_true_log, dtype=float) - np.asarray(forecast_log, dtype=float)
    d = np.clip(d, -30.0, 30.0)
    return np.exp(d) - d - 1.0


def _newey_west_lrv(x: np.ndarray, lag: int) -> float:
    """Newey-West (Bartlett-kernel) long-run variance of a mean: γ0 + 2 Σ_{k=1}^L w_k γ_k.

    ``w_k = 1 − k/(L+1)`` (Bartlett weights guarantee a non-negative estimate). Autocovariances
    use the ``1/n`` convention. ``lag = 0`` reduces to the plain sample variance (γ0).
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    xc = x - x.mean()
    gamma0 = float(np.dot(xc, xc) / n)
    lrv = gamma0
    for k in range(1, lag + 1):
        if k >= n:
            break
        gamma_k = float(np.dot(xc[k:], xc[:-k]) / n)
        weight = 1.0 - k / (lag + 1.0)
        lrv += 2.0 * weight * gamma_k
    return lrv


def diebold_mariano(
    loss_a: np.ndarray,
    loss_b: np.ndarray,
    *,
    horizon: int,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """One-sided Diebold-Mariano test that forecaster **B** is more accurate than **A**.

    Operates on the per-period loss differential ``d = loss_a − loss_b`` (positive ⇒ B has the
    lower loss). Tests H0: ``E[d] = 0`` against H1: ``E[d] > 0`` (B better). For an ``horizon``-step
    forecast the loss differential is MA(``h − 1``), so:

    * the **HAC truncation lag is ``h − 1``** (Newey-West Bartlett long-run variance), and
    * the **Harvey-Leybourne-Newbold (1997) small-sample factor uses the horizon ``h``**:
      ``sqrt((n + 1 − 2h + h(h−1)/n) / n)`` (< 1, shrinking the statistic). The two are distinct —
      using ``h − 1`` in the HLN factor would *inflate* the statistic (e.g. ``sqrt((n+1)/n) > 1`` at
      h = 1). The corrected statistic is referred to a Student-t (``n − 1`` df).

    Returns a JSON-able dict: ``mean_diff``, ``dm_stat`` (uncorrected), ``dm_stat_hln`` (corrected),
    ``p_value`` (one-sided), ``n``, ``horizon``, ``lag`` (= h−1), ``lrv``, and ``significant``.
    A degenerate sample (n < 3, non-positive LRV, or a zero differential) is reported as not
    significant rather than raising — absence of evidence is never counted as evidence.
    """
    from scipy.stats import t as student_t

    a = np.asarray(loss_a, dtype=float)
    b = np.asarray(loss_b, dtype=float)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("loss_a and loss_b must be 1-D arrays of equal length")
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    lag = horizon - 1  # HAC truncation: an h-step forecast error is MA(h-1)
    d = a - b
    n = int(d.size)
    mean_diff = float(d.mean()) if n else float("nan")

    insignificant = {
        "mean_diff": None if n == 0 else round(mean_diff, 8),
        "dm_stat": None,
        "dm_stat_hln": None,
        "p_value": 1.0,
        "n": n,
        "horizon": horizon,
        "lag": lag,
        "lrv": None,
        "significant": False,
    }
    if n < 3 or np.allclose(d, d[0] if n else 0.0):
        return insignificant

    lrv = _newey_west_lrv(d, lag)
    if lrv <= 0.0:
        return {**insignificant, "lrv": round(lrv, 12)}

    dm = mean_diff / math.sqrt(lrv / n)
    # HLN small-sample factor uses the HORIZON h (not the lag h-1), then compare to t_{n-1}.
    h = horizon
    hln_factor = math.sqrt(max((n + 1 - 2 * h + h * (h - 1) / n) / n, 0.0))
    dm_hln = dm * hln_factor
    p_value = float(student_t.sf(dm_hln, df=n - 1))  # one-sided H1: mean_diff > 0
    return {
        "mean_diff": round(mean_diff, 8),
        "dm_stat": round(float(dm), 5),
        "dm_stat_hln": round(float(dm_hln), 5),
        "p_value": round(p_value, 6),
        "n": n,
        "horizon": horizon,
        "lag": lag,
        "lrv": round(float(lrv), 12),
        "significant": bool(p_value < alpha),
    }
