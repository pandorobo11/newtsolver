from __future__ import annotations

"""Prandtl-Meyer expansion relations."""

import math

import numpy as np


def _prandtl_meyer_nu(Mach: np.ndarray, gamma: float) -> np.ndarray:
    """Return Prandtl-Meyer angle ``nu`` [rad] for ``Mach > 1``."""
    g = float(gamma)
    m = np.asarray(Mach, dtype=float)
    m2 = np.square(m)
    beta = np.sqrt(np.maximum(m2 - 1.0, 0.0))
    a = math.sqrt((g + 1.0) / (g - 1.0))
    b = math.sqrt((g - 1.0) / (g + 1.0))
    return a * np.arctan(b * beta) - np.arctan(beta)


def _inverse_prandtl_meyer(nu_target: np.ndarray, gamma: float) -> np.ndarray:
    """Invert Prandtl-Meyer angle ``nu`` [rad] to Mach number.

    Uses safeguarded Newton iterations (with bisection fallback) inside a
    monotone bracket ``M in [1+, +inf)``. This prevents low-``nu`` oscillation.
    """
    g = float(gamma)
    nu = np.asarray(nu_target, dtype=float)
    if g <= 1.0:
        raise ValueError(f"gamma must be > 1, got {g}")
    if np.any(nu < 0.0):
        raise ValueError("nu must be >= 0.")

    nu_max = 0.5 * math.pi * (math.sqrt((g + 1.0) / (g - 1.0)) - 1.0)
    if np.any(nu >= nu_max):
        raise ValueError("nu must be < nu_max for finite Mach.")

    lo = np.full_like(nu, 1.0 + 1e-8, dtype=float)
    hi = np.full_like(nu, 2.0, dtype=float)

    # Expand high bracket until nu(hi) >= nu_target for every element.
    for _ in range(40):
        f_hi = _prandtl_meyer_nu(hi, g) - nu
        need = f_hi < 0.0
        if not np.any(need):
            break
        hi[need] *= 2.0
    else:
        raise RuntimeError("Failed to bracket inverse Prandtl-Meyer root.")

    # Initial guess at interval midpoint.
    m = 0.5 * (lo + hi)

    tol_m = 1e-12
    tol_f = 1e-12
    for _ in range(60):
        nu_m = _prandtl_meyer_nu(m, g)
        f = nu_m - nu

        m2 = np.square(m)
        deriv = np.sqrt(np.maximum(m2 - 1.0, 1e-16)) / (
            m * (1.0 + 0.5 * (g - 1.0) * m2)
        )
        newton = m - f / np.maximum(deriv, 1e-16)

        mid = 0.5 * (lo + hi)
        use_newton = np.isfinite(newton) & (newton > lo) & (newton < hi)
        m_next = np.where(use_newton, newton, mid)

        f_next = _prandtl_meyer_nu(m_next, g) - nu
        root_hit = np.abs(f_next) < tol_f
        hi = np.where((f_next > 0.0) | root_hit, m_next, hi)
        lo = np.where((f_next < 0.0) | root_hit, m_next, lo)

        m = m_next
        width = hi - lo
        if np.all(width < tol_m * np.maximum(1.0, m)):
            break

    return 0.5 * (lo + hi)


def prandtl_meyer_pressure_coefficient(Mach: float, gamma: float, deltar: float) -> float:
    """Return expansion pressure coefficient from Prandtl-Meyer relation.

    Args:
        Mach: Freestream Mach number (>1).
        gamma: Ratio of specific heats (>1).
        deltar: Local flow turning angle [rad], negative for expansion.
    """
    M1 = float(Mach)
    g = float(gamma)
    d = float(deltar)
    if M1 <= 1.0:
        raise ValueError(f"prandtl_meyer requires Mach > 1, got {M1}")
    if g <= 1.0:
        raise ValueError(f"gamma must be > 1, got {g}")
    if d >= 0.0:
        return 0.0

    nu1 = float(_prandtl_meyer_nu(np.array([M1], dtype=float), g)[0])
    nu2 = nu1 - d
    nu_max = 0.5 * math.pi * (math.sqrt((g + 1.0) / (g - 1.0)) - 1.0)
    cp_vac = -2.0 / (g * M1 * M1)
    if nu2 >= nu_max:
        return cp_vac

    M2 = float(_inverse_prandtl_meyer(np.array([nu2], dtype=float), g)[0])
    bracket = (1.0 + 0.5 * (g - 1.0) * M2 * M2) / (1.0 + 0.5 * (g - 1.0) * M1 * M1)
    p2_p1 = bracket ** (-g / (g - 1.0))
    cp = (2.0 / (g * M1 * M1)) * (p2_p1 - 1.0)
    return max(cp, cp_vac)


def prandtl_meyer_pressure_coefficients(Mach: float, gamma: float, deltar: np.ndarray) -> np.ndarray:
    """Vectorized Prandtl-Meyer pressure coefficient evaluation.

    Args:
        Mach: Freestream Mach number (>1).
        gamma: Ratio of specific heats (>1).
        deltar: Local turning-angle array [rad] (negative for expansion).
    """
    M1 = float(Mach)
    g = float(gamma)
    d = np.asarray(deltar, dtype=float)
    if M1 <= 1.0:
        raise ValueError(f"prandtl_meyer requires Mach > 1, got {M1}")
    if g <= 1.0:
        raise ValueError(f"gamma must be > 1, got {g}")

    out = np.zeros_like(d, dtype=float)
    expansion = d < 0.0
    if not np.any(expansion):
        return out

    nu1 = float(_prandtl_meyer_nu(np.array([M1], dtype=float), g)[0])
    nu2 = nu1 - d[expansion]
    nu_max = 0.5 * math.pi * (math.sqrt((g + 1.0) / (g - 1.0)) - 1.0)
    cp_vac = -2.0 / (g * M1 * M1)

    cp_exp = np.full_like(nu2, cp_vac, dtype=float)
    finite = nu2 < nu_max
    if np.any(finite):
        M2 = _inverse_prandtl_meyer(nu2[finite], g)
        bracket = (1.0 + 0.5 * (g - 1.0) * M2 * M2) / (1.0 + 0.5 * (g - 1.0) * M1 * M1)
        p2_p1 = bracket ** (-g / (g - 1.0))
        cp = (2.0 / (g * M1 * M1)) * (p2_p1 - 1.0)
        cp_exp[finite] = np.maximum(cp, cp_vac)

    out[expansion] = cp_exp
    return out
