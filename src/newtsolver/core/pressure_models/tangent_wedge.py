from __future__ import annotations

"""Tangent-wedge oblique-shock relations."""

import math
from functools import lru_cache

import numpy as np

from .modified_newtonian import modified_newtonian_cp_max


def _oblique_theta_from_beta(Mach: float, gamma: float, beta: float) -> float:
    """Return wedge deflection angle ``theta`` [rad] from shock angle ``beta``."""
    m2 = float(Mach) * float(Mach)
    g = float(gamma)
    sin_b = math.sin(beta)
    sin_b2 = sin_b * sin_b
    denom = m2 * (g + math.cos(2.0 * beta)) + 2.0
    if denom <= 0.0:
        return 0.0
    rhs = 2.0 * (m2 * sin_b2 - 1.0) / (math.tan(beta) * denom)
    if rhs <= 0.0:
        return 0.0
    return math.atan(rhs)


def _oblique_theta_from_beta_array(Mach: float, gamma: float, beta: np.ndarray) -> np.ndarray:
    """Vectorized variant of :func:`_oblique_theta_from_beta`."""
    m2 = float(Mach) * float(Mach)
    g = float(gamma)
    b = np.asarray(beta, dtype=float)
    sin_b = np.sin(b)
    sin_b2 = sin_b * sin_b
    denom = m2 * (g + np.cos(2.0 * b)) + 2.0
    rhs = np.zeros_like(b)
    valid_denom = denom > 0.0
    rhs[valid_denom] = 2.0 * (m2 * sin_b2[valid_denom] - 1.0) / (
        np.tan(b[valid_denom]) * denom[valid_denom]
    )
    out = np.zeros_like(b)
    valid_rhs = valid_denom & (rhs > 0.0)
    out[valid_rhs] = np.arctan(rhs[valid_rhs])
    return out


@lru_cache(maxsize=256)
def _tangent_wedge_detach_limit(Mach: float, gamma: float) -> tuple[float, float]:
    """Return ``(theta_max, cp_crit)`` for weak attached tangent-wedge branch.

    Uses the discriminant-zero condition of the theta-beta-M cubic
    (double root at detach) to compute ``theta_max`` analytically.
    """
    M = float(Mach)
    g = float(gamma)
    if M <= 1.0:
        raise ValueError(f"tangent_wedge requires Mach > 1, got {M}")
    if g <= 1.0:
        raise ValueError(f"gamma must be > 1, got {g}")

    m2 = M * M
    c = -2.0 * (m2 - 1.0)
    t1 = m2 * (g + 1.0) + 2.0
    t2 = m2 * (g - 1.0) + 2.0

    # Discriminant of 2*x^3 + b*x^2 + c*x + d = 0 with
    # b = tan(theta)*t1, d = tan(theta)*t2 and x = cot(beta).
    # Delta=0 gives detach limit and reduces to quadratic in y = tan(theta)^2:
    # A*y^2 + B*y + C0 = 0.
    A = -4.0 * (t1**3) * t2
    B = 36.0 * t1 * t2 * c + (t1 * t1) * (c * c) - 108.0 * (t2 * t2)
    C0 = -8.0 * (c**3)

    y_candidates: list[float] = []
    if abs(A) > 1e-30:
        q = B * B - 4.0 * A * C0
        if q < 0.0 and q > -1e-24:
            q = 0.0
        if q < 0.0:
            raise RuntimeError("Failed to compute tangent-wedge detach limit (negative discriminant).")
        sq = math.sqrt(q)
        y_candidates.extend([(-B + sq) / (2.0 * A), (-B - sq) / (2.0 * A)])
    elif abs(B) > 1e-30:
        y_candidates.append(-C0 / B)
    else:
        raise RuntimeError("Failed to compute tangent-wedge detach limit (degenerate quadratic).")

    theta_candidates = [math.atan(math.sqrt(y)) for y in y_candidates if y > 0.0 and math.isfinite(y)]
    if not theta_candidates:
        raise RuntimeError("Failed to compute tangent-wedge detach limit (no physical theta candidate).")

    # Physical detach limit is the larger positive root.
    theta_max = max(theta_candidates)

    # Recover beta at detach from the double-root condition f(x)=f'(x)=0.
    tan_t = math.tan(theta_max)
    b = tan_t * t1
    d = tan_t * t2
    deriv_disc = b * b - 6.0 * c
    if deriv_disc < 0.0 and deriv_disc > -1e-24:
        deriv_disc = 0.0
    if deriv_disc < 0.0:
        raise RuntimeError("Failed to compute tangent-wedge detach beta (negative derivative discriminant).")
    sqd = math.sqrt(deriv_disc)
    x_roots = [(-b + sqd) / 6.0, (-b - sqd) / 6.0]

    mu = math.asin(1.0 / M)
    x_max = 1.0 / math.tan(mu)
    beta_peak = None
    best_key = None
    for x in x_roots:
        if not math.isfinite(x) or not (0.0 < x < x_max):
            continue
        beta = math.atan2(1.0, x)
        if not (mu < beta < 0.5 * math.pi):
            continue
        # Choose candidate with the smallest cubic residual and theta mismatch.
        f_res = abs(2.0 * x**3 + b * x * x + c * x + d)
        th_res = abs(_oblique_theta_from_beta(M, g, beta) - theta_max)
        key = (f_res, th_res)
        if best_key is None or key < best_key:
            best_key = key
            beta_peak = beta

    if beta_peak is None:
        raise RuntimeError("Failed to compute tangent-wedge detach beta (no physical root).")

    mn1 = M * math.sin(beta_peak)
    p2_p1 = 1.0 + (2.0 * g / (g + 1.0)) * (mn1 * mn1 - 1.0)
    cp_crit = (2.0 / (g * M * M)) * (p2_p1 - 1.0)
    return float(theta_max), float(max(cp_crit, 0.0))


def _weak_oblique_shock_beta_array(Mach: float, gamma: float, theta: np.ndarray) -> np.ndarray:
    """Vectorized weak-shock ``beta`` [rad] solver.

    Returns ``NaN`` where no attached weak solution exists.
    """
    M = float(Mach)
    g = float(gamma)
    t = np.asarray(theta, dtype=float)
    if M <= 1.0:
        raise ValueError(f"tangent_wedge requires Mach > 1, got {M}")
    if g <= 1.0:
        raise ValueError(f"gamma must be > 1, got {g}")

    out = np.full_like(t, np.nan, dtype=float)
    if t.size == 0:
        return out

    active = t > 0.0
    if not np.any(active):
        return out

    t_active = t[active]
    tan_t = np.tan(t_active)
    m2 = M * M

    # 2*x^3 + b*x^2 + c*x + d = 0, where x = cot(beta).
    a = 2.0
    b = tan_t * (m2 * (g + 1.0) + 2.0)
    c = -2.0 * (m2 - 1.0)
    d = tan_t * (m2 * (g - 1.0) + 2.0)

    ba = b / a
    ca = c / a
    da = d / a
    p = ca - (ba * ba) / 3.0
    q = (2.0 * ba * ba * ba) / 27.0 - (ba * ca) / 3.0 + da
    disc = (0.5 * q) * (0.5 * q) + (p / 3.0) ** 3

    n = t_active.size
    roots = np.full((n, 3), np.nan, dtype=float)
    eps = 1e-16

    pos = disc > eps
    if np.any(pos):
        s = np.sqrt(disc[pos])
        y = np.cbrt(-0.5 * q[pos] + s) + np.cbrt(-0.5 * q[pos] - s)
        roots[pos, 0] = y - ba[pos] / 3.0

    zero = np.abs(disc) <= eps
    if np.any(zero):
        q_zero = q[zero]
        ba_zero = ba[zero]
        q_small = np.abs(q_zero) <= eps
        idx_all = np.flatnonzero(zero)
        if np.any(q_small):
            idx = idx_all[q_small]
            roots[idx, 0] = -ba_zero[q_small] / 3.0
        if np.any(~q_small):
            idx = idx_all[~q_small]
            u = np.cbrt(-0.5 * q_zero[~q_small])
            roots[idx, 0] = 2.0 * u - ba_zero[~q_small] / 3.0
            roots[idx, 1] = -u - ba_zero[~q_small] / 3.0

    neg = disc < -eps
    if np.any(neg):
        p_neg = p[neg]
        q_neg = q[neg]
        ba_neg = ba[neg]
        r = np.sqrt(np.maximum(-p_neg / 3.0, 0.0))
        idx = np.flatnonzero(neg)
        zero_r = r <= 0.0
        if np.any(zero_r):
            rz = idx[zero_r]
            roots[rz, 0] = -ba_neg[zero_r] / 3.0
        nz = ~zero_r
        if np.any(nz):
            rnz = r[nz]
            arg = -q_neg[nz] / (2.0 * rnz * rnz * rnz)
            arg = np.clip(arg, -1.0, 1.0)
            phi = np.arccos(arg)
            ridx = idx[nz]
            for k in range(3):
                yk = 2.0 * rnz * np.cos((phi + 2.0 * math.pi * k) / 3.0)
                roots[ridx, k] = yk - ba_neg[nz] / 3.0

    x_max = math.sqrt(max(m2 - 1.0, 0.0))
    valid = np.isfinite(roots) & (roots > 0.0) & (roots < x_max)
    roots_valid = np.where(valid, roots, -np.inf)
    weak_x = np.max(roots_valid, axis=1)

    weak_ok = np.isfinite(weak_x) & (weak_x > 0.0)
    if not np.any(weak_ok):
        return out

    beta = np.full_like(weak_x, np.nan, dtype=float)
    beta[weak_ok] = np.arctan2(1.0, weak_x[weak_ok])
    theta_est = _oblique_theta_from_beta_array(M, g, beta)
    residual = np.abs(theta_est - t_active)
    beta[residual > 1e-8] = np.nan

    out_active = np.full_like(t_active, np.nan, dtype=float)
    out_active[:] = beta
    out[active] = out_active
    return out


def tangent_wedge_pressure_coefficients(
    Mach: float,
    gamma: float,
    deltar: np.ndarray,
    *,
    cp_cap: float | None = None,
) -> np.ndarray:
    """Vectorized tangent-wedge pressure coefficient evaluation."""
    M = float(Mach)
    g = float(gamma)
    theta_all = np.asarray(deltar, dtype=float)
    if M <= 1.0:
        raise ValueError(f"tangent_wedge requires Mach > 1, got {M}")
    if g <= 1.0:
        raise ValueError(f"gamma must be > 1, got {g}")

    out = np.zeros_like(theta_all, dtype=float)
    windward = theta_all > 0.0
    if not np.any(windward):
        return out

    cap = float(cp_cap) if cp_cap is not None else modified_newtonian_cp_max(Mach=M, gamma=g)
    theta = theta_all[windward]
    theta_max, cp_crit_raw = _tangent_wedge_detach_limit(M, g)
    cp_crit = min(max(cp_crit_raw, 0.0), cap)

    cp_w = np.zeros_like(theta, dtype=float)
    detached = theta > theta_max
    if np.any(detached):
        s2 = np.square(np.sin(theta[detached]))
        s0_2 = math.sin(theta_max) ** 2
        denom = max(1.0 - s0_2, 1e-12)
        w = np.clip((s2 - s0_2) / denom, 0.0, 1.0)
        cp_w[detached] = cp_crit + (cap - cp_crit) * w

    attached = ~detached
    if np.any(attached):
        beta = _weak_oblique_shock_beta_array(Mach=M, gamma=g, theta=theta[attached])
        cp_a = np.full(beta.shape, cp_crit, dtype=float)
        valid_beta = np.isfinite(beta)
        if np.any(valid_beta):
            mn1 = M * np.sin(beta[valid_beta])
            supersonic = mn1 > 1.0
            cp_valid = np.zeros_like(mn1, dtype=float)
            if np.any(supersonic):
                mn1_sup = mn1[supersonic]
                p2_p1 = 1.0 + (2.0 * g / (g + 1.0)) * (mn1_sup * mn1_sup - 1.0)
                cp_raw = (2.0 / (g * M * M)) * (p2_p1 - 1.0)
                cp_valid[supersonic] = np.clip(cp_raw, 0.0, cap)
            cp_a[valid_beta] = cp_valid
        cp_w[attached] = cp_a

    out[windward] = np.clip(cp_w, 0.0, cap)
    return out
