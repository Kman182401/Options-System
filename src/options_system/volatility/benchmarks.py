"""Phase-23 hard-benchmark battery for the 1-day RV-forecast confirmation study.

The Phase-21 verdict only required the LightGBM treatment to beat **HAR-RV**. Phase 23 hardens the
bar (frozen contract: ``docs/PHASE23_PREREGISTRATION.md``): the treatment must beat HAR **and** each
of three additional benchmarks at the 1-day horizon. This module provides the three additions as
pure, leak-safe functions on the daily series. All forecasts are returned on the **log-variance
scale** (the same scale HAR and the treatment predict on, the input to ``qlike_from_log``).

* **Random walk (RW)** — "tomorrow's variance looks like today's": forecast = ``log(rv_t)``. Uses
  only information through the decision day ``t``; notoriously hard to beat one step ahead.
* **EWMA / RiskMetrics** — exponentially-weighted moving average of realized variance with a fixed
  ``λ = 0.94`` (the RiskMetrics daily standard, no tuning): ``e_t = λ·e_{t-1} + (1−λ)·rv_t``, the
  forecast is ``log(e_t)`` (uses ``rv`` through ``t`` only).
* **GARCH(1,1)** — Gaussian quasi-MLE GARCH(1,1) with a constant mean on the daily RTH log-return
  series; one-step-ahead conditional-variance forecast. *The* canonical volatility benchmark ("does
  anything beat a GARCH(1,1)?", Hansen & Lunde 2005). Estimated with ``scipy`` only — no ``arch`` /
  ``statsmodels`` dependency is added.

Every function is causal: the forecast aligned to a decision day ``t`` uses only data observable at
the close of ``t``. Determinism: GARCH MLE is a fixed-start deterministic optimization.
"""

from __future__ import annotations

import numpy as np

# RiskMetrics daily decay (frozen) — also the GARCH first-fold non-convergence fallback.
RISKMETRICS_LAMBDA = 0.94
_STATIONARITY_CAP = 0.9999  # alpha + beta strictly below 1 (covariance-stationary)


# --------------------------------------------------------------------------- #
# Parameter-free benchmarks on the RV series (log-variance forecasts)
# --------------------------------------------------------------------------- #
def rw_forecast_log(rv_decision: np.ndarray) -> np.ndarray:
    """Random-walk forecast of next-day log-RV = ``log(rv_t)`` (today's realized variance)."""
    rv = np.asarray(rv_decision, dtype=float)
    if np.any(rv <= 0) or not np.all(np.isfinite(rv)):
        raise ValueError("rw_forecast_log requires strictly positive, finite rv")
    return np.log(rv)


def ewma_forecast_log(rv_decision: np.ndarray, lam: float = RISKMETRICS_LAMBDA) -> np.ndarray:
    """RiskMetrics EWMA forecast of next-day log-RV (causal, fixed ``λ``); returns ``log(e_t)``.

    Causal: ``e_t`` (the forecast made at the close of day ``t`` for day ``t+1``) uses ``rv`` only
    through ``t``. Seeded at ``e_0 = rv_0``; with ``λ = 0.94`` the ~17-day memory makes the seed
    irrelevant long before the 2022 OOS window.
    """
    rv = np.asarray(rv_decision, dtype=float)
    if not 0.0 < lam < 1.0:
        raise ValueError(f"ewma lambda must be in (0,1), got {lam}")
    if np.any(rv <= 0) or not np.all(np.isfinite(rv)):
        raise ValueError("ewma_forecast_log requires strictly positive, finite rv")
    e = np.empty_like(rv)
    e[0] = rv[0]
    for i in range(1, rv.size):
        e[i] = lam * e[i - 1] + (1.0 - lam) * rv[i]
    return np.log(e)


# --------------------------------------------------------------------------- #
# GARCH(1,1) — Gaussian QMLE, constant mean, dependency-free
# --------------------------------------------------------------------------- #
def _garch_filter(
    e2: np.ndarray, omega: float, alpha: float, beta: float, s2_0: float
) -> np.ndarray:
    """Conditional-variance recursion ``σ²_t = ω + α e²_{t-1} + β σ²_{t-1}`` (σ²_0 = s2_0)."""
    n = e2.size
    s2 = np.empty(n)
    s2[0] = s2_0
    for t in range(1, n):
        s2[t] = omega + alpha * e2[t - 1] + beta * s2[t - 1]
    return s2


def _garch_nll(theta: np.ndarray, e2: np.ndarray, s2_0: float) -> float:
    """Negative Gaussian quasi-log-likelihood (dropped constants) for GARCH(1,1) on demeaned e²."""
    omega, alpha, beta = float(theta[0]), float(theta[1]), float(theta[2])
    if omega <= 0.0 or alpha < 0.0 or beta < 0.0 or alpha + beta >= _STATIONARITY_CAP:
        return 1e10
    s2 = _garch_filter(e2, omega, alpha, beta, s2_0)
    if np.any(s2 <= 0.0) or not np.all(np.isfinite(s2)):
        return 1e10
    return 0.5 * float(np.sum(np.log(s2) + e2 / s2))


def fit_garch11(returns_train: np.ndarray) -> dict[str, float | bool]:
    """Gaussian QMLE GARCH(1,1) (constant mean) on a training return series.

    Variance-targeting initialization, ``scipy`` L-BFGS-B over (ω, α, β) under ω>0, α≥0, β≥0,
    α+β<1. Returns the fitted params plus ``mu`` (the constant mean), ``s2`` (sample variance, the
    filter seed), and ``converged``. Never raises — a failed fit returns ``converged=False`` and the
    caller applies the frozen fallback (carry-forward / RiskMetrics).
    """
    from scipy.optimize import minimize

    r = np.asarray(returns_train, dtype=float)
    r = r[np.isfinite(r)]
    mu = float(r.mean()) if r.size else 0.0
    e = r - mu
    e2 = e * e
    s2 = float(np.mean(e2)) if e2.size else 1.0
    if e2.size < 50 or s2 <= 0.0:
        return {
            "omega": 0.0,
            "alpha": 0.0,
            "beta": 0.0,
            "mu": mu,
            "s2": max(s2, 1e-12),
            "converged": False,
        }

    # Variance-targeting initialization: omega = s2 * (1 - alpha - beta).
    a0, b0 = 0.05, 0.90
    x0 = np.array([s2 * (1.0 - a0 - b0), a0, b0])
    bounds = [(1e-12, max(10.0 * s2, 1e-6)), (0.0, 0.5), (0.0, 0.9995)]
    try:
        res = minimize(_garch_nll, x0, args=(e2, s2), method="L-BFGS-B", bounds=bounds)
        omega, alpha, beta = float(res.x[0]), float(res.x[1]), float(res.x[2])
        ok = bool(res.success) and omega > 0.0 and (alpha + beta) < _STATIONARITY_CAP - 1e-6
    except Exception:  # noqa: BLE001 - a failed fit falls back, never crashes the verdict
        omega, alpha, beta, ok = 0.0, 0.0, 0.0, False
    return {"omega": omega, "alpha": alpha, "beta": beta, "mu": mu, "s2": s2, "converged": ok}


def garch11_forecast_log(returns_full: np.ndarray, params: dict[str, float]) -> np.ndarray:
    """One-step-ahead GARCH(1,1) log-variance forecasts over a (causal) return series.

    ``params`` come from :func:`fit_garch11` (fit on the fold's training returns; **frozen** here).
    Filters ``σ²_t`` forward with observed returns, then the forecast aligned to day ``t`` is
    ``log(ω + α e²_t + β σ²_t)`` — the variance of day ``t+1`` conditional on info through ``t``
    (leak-safe: parameters never saw the OOS rows, the recursion uses only past/observed returns).
    """
    omega = float(params["omega"])
    alpha = float(params["alpha"])
    beta = float(params["beta"])
    mu = float(params["mu"])
    s2_0 = float(params["s2"])
    e = np.asarray(returns_full, dtype=float) - mu
    e = np.where(np.isfinite(e), e, 0.0)  # a missing first return contributes 0 to the recursion
    e2 = e * e
    s2 = _garch_filter(e2, omega, alpha, beta, s2_0)
    fcast_var = omega + alpha * e2 + beta * s2  # 1-step-ahead variance at each t
    fcast_var = np.clip(fcast_var, 1e-300, None)
    return np.log(fcast_var)


def riskmetrics_params(
    returns_train: np.ndarray, lam: float = RISKMETRICS_LAMBDA
) -> dict[str, float]:
    """The integrated GARCH (RiskMetrics) parameterization: ω=0, α=1−λ, β=λ — the GARCH fallback."""
    r = np.asarray(returns_train, dtype=float)
    r = r[np.isfinite(r)]
    mu = float(r.mean()) if r.size else 0.0
    s2 = float(np.var(r - mu)) if r.size else 1.0
    return {"omega": 0.0, "alpha": 1.0 - lam, "beta": lam, "mu": mu, "s2": max(s2, 1e-12)}
