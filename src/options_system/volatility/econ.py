"""Phase-25 economic layer — pure, deterministic, unit-tested primitives (no I/O, no model).

Everything here is a deterministic transform of the **frozen** Phase-23 forecasts under the
**frozen** Phase-25 knobs — it adds *zero* fitted parameters. The functions implement, exactly as
``docs/PHASE25_PREREGISTRATION.md`` specifies:

* the two long-only weight rules — the mean-variance (FKO money-leg) weight and the return-free
  vol-target (VTE-leg) weight, both clipped;
* realized quadratic utility ``U(R) = R − ½·a·R²`` with ``a = γ/(1+γ)`` (FKO realized-utility form);
* the Fleming-Kirby-Ostdiek **performance fee** ``φ`` solving ``Σ U(R_treat − φ) = Σ U(R_k)`` on
  net-of-cost returns, taking the economically-admissible (smaller-``|φ|``) root and **failing
  closed** (never silently re-spec'ing) when no real root exists;
* the return-free **volatility-targeting error** ``VTE = |log(realized_vol(wv·r) / σ_target)|``;
* the **Politis-Romano (1994) stationary bootstrap** one-sided p-value on a per-day differential;
* the FKO/Moreira-Muir **leverage-matching** constant (match mean exposure to ``STATIC``'s);
* the **E9 active-exposure correlation** ``corr(w_treat − w_k, r_{t+1})`` (leakage void).

These are the integrity core of the study: the long-only clamp, the single fixed ``μ_bar``/``γ``
across arms, the identical realized return, and the return-free VTE leg are what make any economic
differential attributable to the variance forecast alone rather than a smuggled directional bet.
"""

from __future__ import annotations

import math

import numpy as np


class FeeNonConvergence(RuntimeError):
    """Raised when the FKO performance-fee quadratic has no real root — a FAIL, never re-spec'd."""


# --------------------------------------------------------------------------- #
# t+1 alignment — the one-session forward shift
# --------------------------------------------------------------------------- #
def forward_shift_one(same_session_ret: np.ndarray) -> np.ndarray:
    """One-session forward shift: ``next[i] = same[i+1]``; the last element is NaN (no ``t+1``).

    Maps each session ``t``'s same-session RTH return to the RTH return of session ``t+1``, so a
    weight set at the close of forecast row ``t`` earns the NEXT session's return — never the
    already-realized same-session ``r_t`` (a look-ahead leak). The final session has no ``t+1``
    and is set to NaN (dropped by the caller).
    """
    same = np.asarray(same_session_ret, dtype=float)
    out = np.full(same.shape, np.nan)
    out[:-1] = same[1:]
    return out


# --------------------------------------------------------------------------- #
# Weight rules (long-only)
# --------------------------------------------------------------------------- #
def mean_variance_weight(
    mu_bar: float | np.ndarray,
    gamma: float,
    sigma2: np.ndarray,
    w_cap: float,
) -> np.ndarray:
    """The FKO mean-variance weight ``clip(μ_bar / (γ·σ²), 0, w_cap)`` (the money-leg weight).

    Long-only: the lower clip is 0 (no shorting, no sign decision). ``μ_bar`` and ``γ`` are the
    single fixed causal scalars held identical across every arm; ``σ²`` is the only input that
    differs across arms, so the cross-arm weight difference is a function of the variance forecasts
    alone.
    """
    sigma2 = np.asarray(sigma2, dtype=float)
    raw = np.asarray(mu_bar, dtype=float) / (gamma * sigma2)
    return np.clip(raw, 0.0, w_cap)


def vol_target_weight(
    sigma_target_daily: float,
    sigma: np.ndarray,
    w_floor: float,
    w_cap: float,
) -> np.ndarray:
    """The return-free vol-target weight ``clip(σ_target / σ, w_floor, w_cap)`` (the VTE leg).

    Drift is 0 — **no expected-return assumption enters** — so a directional bet cannot pass the
    leg this weight drives. Floored at ``w_floor > 0`` (never flat: no participation/sign decision).
    """
    sigma = np.asarray(sigma, dtype=float)
    return np.clip(sigma_target_daily / sigma, w_floor, w_cap)


# --------------------------------------------------------------------------- #
# Realized quadratic utility (FKO)
# --------------------------------------------------------------------------- #
def utility_coefficient(gamma: float) -> float:
    """``a = γ/(1+γ)`` — the curvature of the FKO realized quadratic utility."""
    return gamma / (1.0 + gamma)


def realized_utility(net: np.ndarray, gamma: float) -> np.ndarray:
    """Per-period realized quadratic utility ``U(R) = R − ½·a·R²`` with ``R = 1 + net``.

    ``net`` is the net-of-cost daily portfolio return; ``rf = 0`` (excess-return convention). This
    is the Fleming-Kirby-Ostdiek realized-utility form used both for the performance fee and the
    fold/regime utility comparisons.
    """
    a = utility_coefficient(gamma)
    r = 1.0 + np.asarray(net, dtype=float)
    return r - 0.5 * a * r * r


# --------------------------------------------------------------------------- #
# FKO performance fee (the money leg)
# --------------------------------------------------------------------------- #
def performance_fee(r_treat: np.ndarray, r_k: np.ndarray, gamma: float) -> float:
    """The per-period FKO fee ``φ`` solving ``Σ_t U(R_treat_t − φ) = Σ_t U(R_k_t)``.

    ``R = 1 + net`` on **net-of-cost** returns. With ``U(R) = R − ½a R²`` the equation is quadratic
    in ``φ``: ``A φ² + B φ + C = 0`` with ``A = ½ a T``, ``B = T − a·Σ R_treat``,
    ``C = Σ U(R_k) − Σ U(R_treat)``. The **economically-admissible** root is the one with the
    smaller ``|φ|`` (the local fee; the far root is the spurious large-curvature solution). A
    positive ``φ`` means the γ-investor pays to switch *to* the treatment. No real root (negative
    discriminant) or an empty sample **raises** :class:`FeeNonConvergence` — a FAIL, no re-spec.
    """
    rt = 1.0 + np.asarray(r_treat, dtype=float)
    rk = 1.0 + np.asarray(r_k, dtype=float)
    t = rt.size
    if t == 0 or rk.size != t:
        raise FeeNonConvergence(f"degenerate fee sample (n_treat={t}, n_k={rk.size})")
    a = utility_coefficient(gamma)
    s1t = float(np.sum(rt))
    u_treat = float(np.sum(rt - 0.5 * a * rt * rt))
    u_k = float(np.sum(rk - 0.5 * a * rk * rk))

    qa = 0.5 * a * t
    qb = t - a * s1t
    qc = u_k - u_treat
    disc = qb * qb - 4.0 * qa * qc
    if qa <= 0.0 or disc < 0.0:
        raise FeeNonConvergence(
            f"fee quadratic has no real root (a={qa:.3e}, disc={disc:.3e}); FAIL, no re-spec"
        )
    sq = math.sqrt(disc)
    r1 = (-qb + sq) / (2.0 * qa)
    r2 = (-qb - sq) / (2.0 * qa)
    return r1 if abs(r1) <= abs(r2) else r2


def fee_bps_per_year(phi: float, ann_factor: int = 252) -> float:
    """Annualize a per-period fee to basis points: ``Δ_bps = φ · ann_factor · 1e4``."""
    return float(phi) * ann_factor * 1e4


# --------------------------------------------------------------------------- #
# Return-free volatility-targeting error (the VTE leg)
# --------------------------------------------------------------------------- #
def vte(weights_vt: np.ndarray, r_next: np.ndarray, sigma_target_daily: float) -> float:
    """``|log( realized_vol(wv · r) / σ_target_daily )|`` — how tightly the inverse-forecast overlay
    holds realized portfolio volatility to target.

    Return-free: only the *realized* return enters (to measure realized vol); no expected-return
    assumption. ``realized_vol`` is the sample standard deviation (``ddof = 1``) of the daily
    vol-targeted portfolio return ``wv · r``. Lower is better. The ``ddof`` choice is common to
    every arm (same ``n``), so it cancels in the cross-arm VTE comparison.
    """
    wv = np.asarray(weights_vt, dtype=float)
    r = np.asarray(r_next, dtype=float)
    port = wv * r
    realized = float(np.std(port, ddof=1))
    if realized <= 0.0 or sigma_target_daily <= 0.0:
        return float("inf")
    return abs(math.log(realized / sigma_target_daily))


# --------------------------------------------------------------------------- #
# Politis-Romano (1994) stationary bootstrap
# --------------------------------------------------------------------------- #
def stationary_bootstrap_pvalue(
    diff: np.ndarray,
    *,
    n_resamples: int,
    expected_block_length: int,
    seed: int,
    one_sided: bool = True,
) -> dict[str, float | int | bool | None]:
    """One-sided stationary-bootstrap p-value for ``H1: E[diff] > 0`` (Politis-Romano 1994).

    Resamples the per-day differential in geometrically-distributed blocks (expected length ``L``,
    wrap-around) ``B`` times and reports the fraction of resample means at/below zero (with the
    standard ``+1`` smoothing): ``p = (1 + #{ d̄*_b ≤ 0 }) / (B + 1)``. A small ``p`` means almost no
    resample of the dependent series dips to a non-positive mean — durable positive value. Seeded
    for exact reproducibility. A degenerate sample (``n < 3``) is reported as not-significant rather
    than raising — absence of evidence is never counted as evidence.
    """
    d = np.asarray(diff, dtype=float)
    n = int(d.size)
    mean_diff = float(d.mean()) if n else float("nan")
    if n < 3:
        return {
            "mean_diff": None if n == 0 else round(mean_diff, 10),
            "p_value": 1.0,
            "n": n,
            "n_resamples": n_resamples,
            "expected_block_length": expected_block_length,
        }
    rng = np.random.default_rng(seed)
    p_restart = 1.0 / float(expected_block_length)
    b = int(n_resamples)
    # Column-wise recurrence: advance B independent index paths one position at a time, accumulating
    # the per-resample sum (O(B) memory, never materializing the B×n resample matrix).
    cur = rng.integers(0, n, size=b)
    sums = d[cur].astype(float)
    for _ in range(1, n):
        restart = rng.random(b) < p_restart
        cur = np.where(restart, rng.integers(0, n, size=b), (cur + 1) % n)
        sums += d[cur]
    means = sums / n
    p = (1.0 + float(np.count_nonzero(means <= 0.0))) / (b + 1.0)
    return {
        "mean_diff": round(mean_diff, 10),
        "p_value": round(p, 6),
        "n": n,
        "n_resamples": b,
        "expected_block_length": expected_block_length,
    }


# --------------------------------------------------------------------------- #
# Leverage matching (FKO / Moreira-Muir)
# --------------------------------------------------------------------------- #
def leverage_match_constant(wbar_arm_train: float, w_static: float) -> float:
    """The per-fold scalar ``c = w̄_static / w̄_arm^train`` matching an arm's mean exposure to STATIC.

    Causal (training-window means only). Falls back to ``1.0`` (disclosed) when the arm's
    training-window mean weight is non-positive (degenerate — cannot rescale a zero exposure).
    """
    if wbar_arm_train <= 0.0:
        return 1.0
    return w_static / wbar_arm_train


# --------------------------------------------------------------------------- #
# E9 — directional-leakage backstop
# --------------------------------------------------------------------------- #
def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation, returning 0.0 for a degenerate (zero-variance) input rather than NaN."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 3 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def active_exposure_corr(w_treat: np.ndarray, w_k: np.ndarray, r_next: np.ndarray) -> float:
    """``corr(w_treat − w_k, r_{t+1})`` — the directional correlation of the **active** overweight.

    E9 voids the verdict if ``|active_exposure_corr| ≥ max_abs_corr`` for any gated arm: the active
    overweight that drives a money differential must not be a directional / risk-premium bet (a low
    absolute-weight correlation can still hide a directionally-correlated *overweight*).
    """
    active = np.asarray(w_treat, dtype=float) - np.asarray(w_k, dtype=float)
    return _safe_corr(active, r_next)


def weight_return_corr(w: np.ndarray, r_next: np.ndarray) -> float:
    """``corr(w_t, r_{t+1})`` — per-arm absolute-weight directional correlation (diagnostic)."""
    return _safe_corr(np.asarray(w, dtype=float), r_next)
