"""Overfitting statistics — the verdict, not the accuracy score.

A high cross-validated accuracy means nothing if the selection process is overfit
or the track record is too short to tell skill from luck. These statistics are the
gate every model passes through:

* :func:`probabilistic_sharpe_ratio` (PSR) — probability the *true* Sharpe exceeds
  a benchmark, correcting the estimate for sample length, skew and fat tails
  (Bailey & López de Prado 2012).
* :func:`deflated_sharpe_ratio` (DSR) — PSR against a threshold that rises with the
  **number of trials** tested, so a strategy cherry-picked from many must clear the
  best you'd expect from pure noise (Bailey & López de Prado 2014).
* :func:`min_track_record_length` — how many observations you'd need before the
  Sharpe is trustworthy at a confidence level.
* :func:`probability_of_backtest_overfitting` (PBO) via CSCV — the probability the
  in-sample-best configuration underperforms the out-of-sample median
  (Bailey, Borwein, López de Prado & Zhu 2015).

Conventions (read once, applied everywhere):

* ``observed_sr``, ``benchmark_sr``, ``skew`` and ``kurt`` are all on the **same
  return frequency** (per-bar here). Mixing an annualised Sharpe with per-bar
  moments is the classic √periods scaling bug — don't.
* ``kurt`` is **raw (non-excess) kurtosis**: a Normal distribution has ``kurt = 3``.
  ``scipy.stats.kurtosis`` defaults to *excess* — convert with ``kurt = excess + 3``.
  Under Normality (``skew=0, kurt=3``) the variance term ``(kurt-1)/4 = 0.5`` and the
  PSR denominator becomes ``sqrt(1 + 0.5·SR²)`` (Lo 2002), not 1.
* Φ and Φ⁻¹ come from :class:`statistics.NormalDist` (stdlib) — no scipy dependency.
"""

from __future__ import annotations

import math
from statistics import NormalDist
from typing import Any

import numpy as np

_NORM = NormalDist()
_EULER_GAMMA = 0.5772156649015329  # Euler–Mascheroni constant


# --------------------------------------------------------------------------- #
# Sharpe-ratio family (PSR / DSR / minTRL)
# --------------------------------------------------------------------------- #
def _sr_denominator(observed_sr: float, skew: float, kurt: float) -> float:
    """The PSR variance term sqrt(1 - skew·SR + (kurt-1)/4·SR²). Raises if non-positive."""
    var_term = 1.0 - skew * observed_sr + ((kurt - 1.0) / 4.0) * observed_sr**2
    if var_term <= 0.0:
        raise ValueError(
            f"PSR variance term non-positive ({var_term:.6g}); inputs imply a degenerate "
            "Sharpe distribution (check skew/kurtosis sign and frequency)"
        )
    return math.sqrt(var_term)


def probabilistic_sharpe_ratio(
    observed_sr: float,
    benchmark_sr: float,
    n: int,
    skew: float = 0.0,
    kurt: float = 3.0,
) -> float:
    """Probabilistic Sharpe Ratio: P(true SR > ``benchmark_sr``) given the estimate.

    ``n`` = number of return observations (Bessel ``n-1`` is used). ``skew`` /
    ``kurt`` are the return skewness and **raw** kurtosis (Normal ``kurt=3``).
    Returns a probability in (0, 1).
    """
    if n < 2:
        raise ValueError(f"n must be >= 2, got {n}")
    z = (observed_sr - benchmark_sr) * math.sqrt(n - 1) / _sr_denominator(observed_sr, skew, kurt)
    return _NORM.cdf(z)


def min_track_record_length(
    observed_sr: float,
    benchmark_sr: float = 0.0,
    skew: float = 0.0,
    kurt: float = 3.0,
    prob: float = 0.95,
) -> float:
    """Minimum number of observations for PSR to reach ``prob`` confidence.

    Requires ``observed_sr > benchmark_sr`` (otherwise the confidence is never
    reached and the length is undefined). Result is in return-frequency units and
    is generally non-integer — ``ceil`` it for a usable count.
    """
    if not 0.0 < prob < 1.0:
        raise ValueError(f"prob must be in (0, 1), got {prob}")
    if observed_sr <= benchmark_sr:
        raise ValueError(
            f"observed_sr ({observed_sr}) must exceed benchmark_sr ({benchmark_sr}); "
            "minTRL is undefined otherwise"
        )
    var_term = 1.0 - skew * observed_sr + ((kurt - 1.0) / 4.0) * observed_sr**2
    z = _NORM.inv_cdf(prob)
    return 1.0 + var_term * (z / (observed_sr - benchmark_sr)) ** 2


def expected_max_sharpe(sr_variance: float, n_trials: int) -> float:
    """Expected maximum of ``n_trials`` Sharpe ratios drawn from N(0, ``sr_variance``).

    The DSR deflation threshold (Bailey & López de Prado 2014), via the Gumbel
    extreme-value approximation
    ``SR0 = sqrt(V)·[(1-γ)·Φ⁻¹(1 - 1/N) + γ·Φ⁻¹(1 - 1/(N·e))]``.
    With a single trial there is no selection bias, so the threshold is 0.
    """
    if sr_variance < 0.0:
        raise ValueError(f"sr_variance must be >= 0, got {sr_variance}")
    if n_trials < 1:
        raise ValueError(f"n_trials must be >= 1, got {n_trials}")
    if n_trials == 1:
        return 0.0
    z1 = _NORM.inv_cdf(1.0 - 1.0 / n_trials)
    z2 = _NORM.inv_cdf(1.0 - 1.0 / (n_trials * math.e))
    return math.sqrt(sr_variance) * ((1.0 - _EULER_GAMMA) * z1 + _EULER_GAMMA * z2)


def deflated_sharpe_ratio(
    observed_sr: float,
    sr_estimates: np.ndarray,
    n: int,
    skew: float = 0.0,
    kurt: float = 3.0,
) -> float:
    """Deflated Sharpe Ratio: PSR against the noise-derived selection threshold.

    ``sr_estimates`` is the set of Sharpe ratios from all trials tested (its
    variance and count drive the deflation). ``observed_sr`` is the selected
    (best) strategy's Sharpe; ``n``/``skew``/``kurt`` describe its track record.
    """
    sr_estimates = np.asarray(sr_estimates, dtype=float)
    n_trials = int(sr_estimates.size)
    if n_trials < 1:
        raise ValueError("sr_estimates must contain at least one trial")
    sr_variance = float(np.var(sr_estimates, ddof=1)) if n_trials > 1 else 0.0
    sr0 = expected_max_sharpe(sr_variance, n_trials)
    return probabilistic_sharpe_ratio(observed_sr, sr0, n, skew, kurt)


# --------------------------------------------------------------------------- #
# Sharpe / moment helpers (per-observation, raw kurtosis)
# --------------------------------------------------------------------------- #
def sharpe_ratio(returns: np.ndarray) -> float:
    """Per-observation Sharpe ratio mean/std (sample std, ddof=1). 0 if std==0."""
    r = np.asarray(returns, dtype=float)
    if r.size < 2:
        return 0.0
    sd = r.std(ddof=1)
    if sd == 0.0:
        return 0.0
    return float(r.mean() / sd)


def return_moments(returns: np.ndarray) -> tuple[int, float, float, float]:
    """Return ``(n, sharpe, skew, raw_kurt)`` for a per-observation return series."""
    r = np.asarray(returns, dtype=float)
    n = int(r.size)
    sr = sharpe_ratio(r)
    if n < 2:
        return n, sr, 0.0, 3.0
    mu = r.mean()
    sd_pop = math.sqrt(float(np.mean((r - mu) ** 2)))
    if sd_pop == 0.0:
        return n, sr, 0.0, 3.0
    skew = float(np.mean((r - mu) ** 3) / sd_pop**3)
    raw_kurt = float(np.mean((r - mu) ** 4) / sd_pop**4)
    return n, sr, skew, raw_kurt


# --------------------------------------------------------------------------- #
# Probability of Backtest Overfitting (CSCV)
# --------------------------------------------------------------------------- #
def _average_ranks(a: np.ndarray) -> np.ndarray:
    """1-based ranks with ties resolved by their average (smallest value → rank 1)."""
    a = np.asarray(a, dtype=float)
    order = np.argsort(a, kind="mergesort")
    sorted_a = a[order]
    ranks = np.empty(a.size, dtype=float)
    i = 0
    while i < a.size:
        j = i
        while j + 1 < a.size and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def _config_sharpes(block: np.ndarray) -> np.ndarray:
    """Per-column (per-config) Sharpe over a block of rows (periods × configs)."""
    if block.shape[0] < 2:
        return np.zeros(block.shape[1])
    mean = block.mean(axis=0)
    sd = block.std(axis=0, ddof=1)
    out = np.zeros(block.shape[1])
    nz = sd > 0
    out[nz] = mean[nz] / sd[nz]
    return out


def probability_of_backtest_overfitting(
    performance: np.ndarray,
    n_partitions: int = 10,
) -> dict[str, Any]:
    """PBO via Combinatorially-Symmetric Cross-Validation (Bailey et al. 2015).

    ``performance`` is a ``(T periods, N configs)`` matrix of per-period performance
    (e.g. returns) — one column per strategy configuration being compared. Rows are
    split into ``n_partitions`` (even) contiguous chunks; for each way of choosing
    half the chunks as in-sample, the best-IS config's relative rank out-of-sample
    gives a logit ``λ``. PBO = fraction of combinations with ``λ <= 0`` (IS winner at
    or below the OOS median → consistent with overfitting).

    Returns ``{"pbo": float, "logits": np.ndarray, "n_combinations": int}``.
    """
    from itertools import combinations

    M = np.asarray(performance, dtype=float)
    if M.ndim != 2:
        raise ValueError("performance must be a 2-D (periods × configs) matrix")
    t_rows, n_configs = M.shape
    if n_configs < 2:
        raise ValueError(f"PBO needs >= 2 configurations to compare, got {n_configs}")
    if n_partitions < 2 or n_partitions % 2 != 0:
        raise ValueError(f"n_partitions must be an even integer >= 2, got {n_partitions}")
    if n_partitions > t_rows:
        raise ValueError(f"n_partitions ({n_partitions}) > n periods ({t_rows})")

    chunks = [c for c in np.array_split(np.arange(t_rows), n_partitions) if c.size]
    s = len(chunks)
    all_idx = set(range(s))
    eps = 1.0 / (n_configs + 1.0)  # rank/(N+1) ⇒ never hits 0 or 1, so the logit is finite

    logits: list[float] = []
    for is_groups in combinations(range(s), s // 2):
        oos_groups = sorted(all_idx - set(is_groups))
        is_rows = np.concatenate([chunks[g] for g in is_groups])
        oos_rows = np.concatenate([chunks[g] for g in oos_groups])
        is_perf = _config_sharpes(M[is_rows])
        oos_perf = _config_sharpes(M[oos_rows])
        best = int(np.argmax(is_perf))
        oos_rank = _average_ranks(oos_perf)[best]  # 1..N
        omega = oos_rank / (n_configs + 1.0)
        omega = min(max(omega, eps), 1.0 - eps)
        logits.append(math.log(omega / (1.0 - omega)))

    logits_arr = np.asarray(logits, dtype=float)
    pbo = float(np.mean(logits_arr <= 0.0)) if logits_arr.size else float("nan")
    return {"pbo": pbo, "logits": logits_arr, "n_combinations": int(logits_arr.size)}
