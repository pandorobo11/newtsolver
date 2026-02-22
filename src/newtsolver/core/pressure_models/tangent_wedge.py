from __future__ import annotations

"""Tangent-wedge oblique-shock relations."""

import math
from functools import lru_cache

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


def _real_cuberoot(value: float) -> float:
    """Return real cube root for any real input."""
    if value >= 0.0:
        return value ** (1.0 / 3.0)
    return -((-value) ** (1.0 / 3.0))


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


def _weak_oblique_shock_beta(Mach: float, gamma: float, theta: float) -> float | None:
    """Return weak-shock ``beta`` [rad] for wedge angle ``theta`` [rad], or ``None``.

    Solves the theta-beta-M relation using the cubic in ``x = cot(beta)``:
    ``2*x^3 + B*x^2 + C*x + D = 0`` with
    ``B = tan(theta) * (M^2*(gamma+1) + 2)``,
    ``C = -2 * (M^2 - 1)``,
    ``D = tan(theta) * (M^2*(gamma-1) + 2)``.
    """
    M = float(Mach)
    g = float(gamma)
    t = float(theta)
    if M <= 1.0:
        raise ValueError(f"tangent_wedge requires Mach > 1, got {M}")
    if g <= 1.0:
        raise ValueError(f"gamma must be > 1, got {g}")
    if t <= 0.0:
        return math.asin(1.0 / M)

    tan_t = math.tan(t)
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

    roots_x: list[float] = []
    if disc > 0.0:
        s = math.sqrt(disc)
        y = _real_cuberoot(-0.5 * q + s) + _real_cuberoot(-0.5 * q - s)
        roots_x = [y - ba / 3.0]
    elif abs(disc) <= 1e-16:
        if abs(q) <= 1e-16:
            roots_x = [-ba / 3.0]
        else:
            u = _real_cuberoot(-0.5 * q)
            roots_x = [2.0 * u - ba / 3.0, -u - ba / 3.0]
    else:
        r = math.sqrt(max(-p / 3.0, 0.0))
        if r == 0.0:
            roots_x = [-ba / 3.0]
        else:
            arg = -q / (2.0 * r * r * r)
            arg = max(-1.0, min(1.0, arg))
            phi = math.acos(arg)
            for k in range(3):
                y = 2.0 * r * math.cos((phi + 2.0 * math.pi * k) / 3.0)
                roots_x.append(y - ba / 3.0)

    x_max = math.sqrt(max(m2 - 1.0, 0.0))
    weak_x: float | None = None
    valid_x: list[float] = []
    for x in roots_x:
        if not math.isfinite(x):
            continue
        if not (0.0 < x < x_max):
            continue
        valid_x.append(x)
        if weak_x is None or x > weak_x:
            weak_x = x

    if weak_x is None:
        return None

    # Weak branch has the smaller beta (larger cot(beta)).
    weak_beta = math.atan2(1.0, weak_x)
    weak_res = abs(_oblique_theta_from_beta(M, g, weak_beta) - t)
    if weak_res <= 1e-8:
        return weak_beta
    for x in valid_x:
        if x == weak_x:
            continue
        beta = math.atan2(1.0, x)
        res = abs(_oblique_theta_from_beta(M, g, beta) - t)
        if res <= 1e-8:
            return beta

    # Above theta_max (or numerically degenerate), treat as detached.
    return None


def tangent_wedge_pressure_coefficient(
    Mach: float,
    gamma: float,
    deltar: float,
    *,
    cp_cap: float | None = None,
) -> float:
    """Return windward ``Cp`` from tangent-wedge oblique-shock relation.

    Args:
        Mach: Freestream Mach number (>1).
        gamma: Ratio of specific heats (>1).
        deltar: Local positive turning angle [rad] on windward side.
        cp_cap: Optional modified-Newtonian cap ``Cp_max`` (defaults to Cp_max).
    """
    M = float(Mach)
    g = float(gamma)
    theta = float(deltar)
    if M <= 1.0:
        raise ValueError(f"tangent_wedge requires Mach > 1, got {M}")
    if g <= 1.0:
        raise ValueError(f"gamma must be > 1, got {g}")
    if theta <= 0.0:
        return 0.0

    cap = float(cp_cap) if cp_cap is not None else modified_newtonian_cp_max(Mach=M, gamma=g)
    theta_max, cp_crit_raw = _tangent_wedge_detach_limit(M, g)
    cp_crit = min(max(cp_crit_raw, 0.0), cap)
    if theta > theta_max:
        # Detached regime: use a shifted modified-Newtonian curve that matches
        # Cp(theta_max)=Cp_crit and Cp(90deg)=Cp_cap.
        s = math.sin(theta)
        s2 = s * s
        s0 = math.sin(theta_max)
        s0_2 = s0 * s0
        denom = max(1.0 - s0_2, 1e-12)
        w = (s2 - s0_2) / denom
        w = min(max(w, 0.0), 1.0)
        cp = cp_crit + (cap - cp_crit) * w
        return min(max(cp, 0.0), cap)

    beta = _weak_oblique_shock_beta(Mach=M, gamma=g, theta=theta)
    if beta is None:
        return cp_crit

    mn1 = M * math.sin(beta)
    if mn1 <= 1.0:
        return 0.0
    p2_p1 = 1.0 + (2.0 * g / (g + 1.0)) * (mn1 * mn1 - 1.0)
    cp = (2.0 / (g * M * M)) * (p2_p1 - 1.0)
    return min(max(float(cp), 0.0), cap)
