from __future__ import annotations

"""Newtonian pressure model and coordinate transforms."""

import math

import numpy as np

ATTITUDE_INPUT_VALUES = {"beta_tan", "beta_sin", "bank"}
WINDWARD_EQUATION_VALUES = {"newtonian", "modified_newtonian", "tangent_wedge", "shield"}
LEEWARD_EQUATION_VALUES = {"shield", "newtonian_mirror", "prandtl_meyer"}

def _resolve_attitude_mode(attitude_input: str | None) -> str:
    """Return canonical attitude mode with default and validation."""
    mode = str(attitude_input or "").strip().lower() or "beta_tan"
    if mode not in ATTITUDE_INPUT_VALUES:
        raise ValueError(
            f"Invalid attitude_input: '{attitude_input}'. "
            "Expected one of: beta_tan, beta_sin, bank."
        )
    return mode


def _resolve_windward_equation(value: str | None) -> str:
    """Normalize windward equation selector to canonical keyword."""
    eq = str(value or "").strip().lower() or "newtonian"
    if eq not in WINDWARD_EQUATION_VALUES:
        raise ValueError(
            f"Invalid windward_eq: '{value}'. "
            "Expected one of: newtonian, modified_newtonian, tangent_wedge, shield."
        )
    return eq


def _resolve_leeward_equation(value: str | None) -> str:
    """Normalize leeward equation selector to canonical keyword."""
    eq = str(value or "").strip().lower() or "shield"
    if eq not in LEEWARD_EQUATION_VALUES:
        raise ValueError(
            f"Invalid leeward_eq: '{value}'. "
            "Expected one of: shield, newtonian_mirror, prandtl_meyer."
        )
    return eq


def modified_newtonian_cp_max(Mach: float, gamma: float) -> float:
    """Return Modified-Newtonian stagnation pressure coefficient ``Cp_max``.

    ``Cp_max`` is computed from a normal-shock + isentropic recovery model:
    ``Cp_max = 2/(gamma*M^2) * (p02/p1 - 1)``.
    """
    M1 = float(Mach)
    g = float(gamma)
    if M1 <= 1.0:
        raise ValueError(f"modified_newtonian requires Mach > 1, got {M1}")
    if g <= 1.0:
        raise ValueError(f"gamma must be > 1, got {g}")

    m1_sq = M1 * M1
    p2_p1 = 1.0 + (2.0 * g / (g + 1.0)) * (m1_sq - 1.0)
    m2_sq = (1.0 + 0.5 * (g - 1.0) * m1_sq) / (g * m1_sq - 0.5 * (g - 1.0))
    if m2_sq <= 0.0:
        raise ValueError(f"Invalid post-shock state for Mach={M1}, gamma={g}")
    p02_p2 = (1.0 + 0.5 * (g - 1.0) * m2_sq) ** (g / (g - 1.0))
    p02_p1 = p2_p1 * p02_p2
    cp_max = (2.0 / (g * m1_sq)) * (p02_p1 - 1.0)
    if not math.isfinite(cp_max) or cp_max < 0.0:
        raise ValueError(f"Invalid Cp_max computed: {cp_max}")
    return float(cp_max)


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

    mu = math.asin(1.0 / M)
    x_max = 1.0 / math.tan(mu)
    beta_candidates: list[tuple[float, float]] = []
    for x in roots_x:
        if not math.isfinite(x):
            continue
        if not (0.0 < x < x_max):
            continue
        beta = math.atan2(1.0, x)
        if not (mu < beta < 0.5 * math.pi):
            continue
        res = abs(_oblique_theta_from_beta(M, g, beta) - t)
        beta_candidates.append((x, res))

    if not beta_candidates:
        return None

    # Weak branch has the smaller beta (larger cot(beta)).
    beta_candidates.sort(key=lambda item: item[0], reverse=True)
    for x, res in beta_candidates:
        if res <= 1e-8:
            return math.atan2(1.0, x)

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
        cp_cap: Optional cap for detached-shock fallback (defaults to Cp_max).
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
    beta = _weak_oblique_shock_beta(Mach=M, gamma=g, theta=theta)
    if beta is None:
        return cap

    mn1 = M * math.sin(beta)
    if mn1 <= 1.0:
        return 0.0
    p2_p1 = 1.0 + (2.0 * g / (g + 1.0)) * (mn1 * mn1 - 1.0)
    cp = (2.0 / (g * M * M)) * (p2_p1 - 1.0)
    return min(max(float(cp), 0.0), cap)


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


def _resolve_attitude_beta_tan(alpha_in: float, beta_in: float) -> tuple[np.ndarray, float, float]:
    """Resolve ``Vhat`` when inputs are ``alpha_t`` and ``beta_t``.

    Uses:
    ``Vhat = normalize([cos(alpha)cos(beta), -sin(beta)cos(alpha), sin(alpha)cos(beta)])``
    """
    a = math.radians(alpha_in)
    b = math.radians(beta_in)
    ca = math.cos(a)
    cb = math.cos(b)
    vhat = np.array([ca * cb, -math.sin(b) * ca, math.sin(a) * cb], dtype=float)
    n = np.linalg.norm(vhat)
    if n < 1e-14:
        raise ValueError("Invalid alpha/beta leading to zero direction.")
    vhat = vhat / n
    return vhat, alpha_in, beta_in


def _resolve_attitude_bank(alpha_in: float, beta_in: float) -> tuple[np.ndarray, float, float]:
    """Resolve ``Vhat`` when inputs are ``alpha_i`` and bank angle ``phi``."""
    ai = math.radians(alpha_in)
    phi = math.radians(beta_in)
    vhat = np.array(
        [
            math.cos(ai),
            -math.sin(ai) * math.sin(phi),
            math.sin(ai) * math.cos(phi),
        ],
        dtype=float,
    )
    n = np.linalg.norm(vhat)
    if n < 1e-14:
        raise ValueError("Invalid bank-angle inputs leading to zero direction.")
    vhat = vhat / n
    alpha_t = math.degrees(math.atan2(float(vhat[2]), float(vhat[0])))
    beta_t = math.degrees(math.atan2(float(-vhat[1]), float(vhat[0])))
    return vhat, alpha_t, beta_t


def _resolve_attitude_beta_sin(alpha_in: float, beta_in: float) -> tuple[np.ndarray, float, float]:
    """Resolve ``Vhat`` when inputs are ``alpha_t`` and ``beta_s``."""
    a = math.radians(alpha_in)
    bs = math.radians(beta_in)
    tan_a = math.tan(a)
    sin_bs = math.sin(bs)
    denom = 1.0 + tan_a * tan_a
    x2 = (1.0 - sin_bs * sin_bs) / denom
    if x2 < -1e-14:
        raise ValueError("Inconsistent alpha_t/beta_s inputs.")
    if x2 < 0.0:
        x2 = 0.0
    sign_x = 1.0 if math.cos(a) >= 0.0 else -1.0
    vx = sign_x * math.sqrt(x2)
    vy = -sin_bs
    vz = tan_a * vx
    vhat = np.array([vx, vy, vz], dtype=float)
    n = np.linalg.norm(vhat)
    if n < 1e-14:
        raise ValueError("Invalid beta-sin inputs leading to zero direction.")
    vhat = vhat / n
    alpha_t = math.degrees(math.atan2(float(vhat[2]), float(vhat[0])))
    beta_t = math.degrees(math.atan2(float(-vhat[1]), float(vhat[0])))
    return vhat, alpha_t, beta_t


def resolve_attitude_to_vhat(
    alpha_deg: float,
    beta_deg: float,
    attitude_input: str | None = None,
) -> tuple[np.ndarray, float, float, str]:
    """Resolve freestream vector from one of multiple attitude-angle definitions.

    Args:
        alpha_deg: First attitude angle in degrees.
        beta_deg: Second attitude angle in degrees.
        attitude_input: Input definition selector.

    Returns:
        Tuple ``(Vhat_stl, alpha_t_deg, beta_t_deg, attitude_mode)`` where
        ``alpha_t_deg`` and ``beta_t_deg`` are tangent-definition angles
        corresponding to the resolved ``Vhat_stl``.
    """
    mode = _resolve_attitude_mode(attitude_input)
    alpha_in = float(alpha_deg)
    beta_in = float(beta_deg)

    if mode == "beta_tan":
        vhat, alpha_t, beta_t = _resolve_attitude_beta_tan(alpha_in, beta_in)
        return vhat, alpha_t, beta_t, mode

    if mode == "bank":
        vhat, alpha_t, beta_t = _resolve_attitude_bank(alpha_in, beta_in)
        return vhat, alpha_t, beta_t, mode

    vhat, alpha_t, beta_t = _resolve_attitude_beta_sin(alpha_in, beta_in)
    return vhat, alpha_t, beta_t, mode


def newtonian_dC_dA_vector(
    Vhat: np.ndarray,
    n_out: np.ndarray,
    Aref: float,
    shielded: bool = False,
    cp_max: float = 2.0,
    windward_eq: str = "newtonian",
    leeward_eq: str = "shield",
    Mach: float | None = None,
    gamma: float | None = None,
) -> np.ndarray:
    """Compute panel force-coefficient density vector ``dC/dA`` by Newtonian rule.

    Windward face (``n_in·Vhat > 0``): ``Cp = cp_max * (n_in·Vhat)^2``.
    Leeward face or shielded face: ``Cp = 0``.
    Pressure force acts along ``-n_out``.
    """
    if shielded:
        return np.zeros(3, dtype=float)

    Vhat = np.asarray(Vhat, dtype=float)
    n_out = np.asarray(n_out, dtype=float)
    n_in = -n_out

    windward_eq = _resolve_windward_equation(windward_eq)
    leeward_eq = _resolve_leeward_equation(leeward_eq)
    gamma_n = float(np.dot(Vhat, n_in))

    if gamma_n > 0.0:
        if windward_eq == "shield":
            return np.zeros(3, dtype=float)
        if windward_eq in {"newtonian", "modified_newtonian"}:
            cp = float(cp_max) * (gamma_n * gamma_n)
        else:
            if Mach is None or gamma is None:
                raise ValueError("Mach and gamma are required for windward_eq=tangent_wedge.")
            deltar = math.asin(max(-1.0, min(1.0, gamma_n)))
            cp = tangent_wedge_pressure_coefficient(
                Mach=float(Mach),
                gamma=float(gamma),
                deltar=deltar,
                cp_cap=float(cp_max),
            )
    else:
        if leeward_eq == "shield":
            return np.zeros(3, dtype=float)
        if leeward_eq == "newtonian_mirror":
            cp = float(cp_max) * (gamma_n * gamma_n)
        else:
            if Mach is None or gamma is None:
                raise ValueError("Mach and gamma are required for leeward_eq=prandtl_meyer.")
            deltar = math.asin(max(-1.0, min(1.0, gamma_n)))
            cp = prandtl_meyer_pressure_coefficient(Mach=float(Mach), gamma=float(gamma), deltar=deltar)

    return -(cp / float(Aref)) * n_out


def newtonian_dC_dA_vectors(
    Vhat: np.ndarray,
    n_out: np.ndarray,
    Aref: float,
    shielded: np.ndarray | bool = False,
    cp_max: float = 2.0,
    windward_eq: str = "newtonian",
    leeward_eq: str = "shield",
    Mach: float | None = None,
    gamma: float | None = None,
) -> np.ndarray:
    """Compute Newtonian ``dC/dA`` for multiple panels in one call.

    Args:
        Vhat: Freestream unit vector in STL axes, shape ``(3,)``.
        n_out: Outward panel normals in STL axes, shape ``(N, 3)``.
        Aref: Reference area [m^2] for non-dimensionalization.
        shielded: Shield mask. Scalar bool or bool array of shape ``(N,)``.
        cp_max: Newtonian maximum pressure coefficient.

    Returns:
        Force-coefficient density vectors in STL axes, shape ``(N, 3)``.
        Shielded rows are zeros.
    """
    Vhat = np.asarray(Vhat, dtype=float)
    n_out = np.asarray(n_out, dtype=float)
    if n_out.ndim != 2 or n_out.shape[1] != 3:
        raise ValueError("n_out must have shape (N, 3).")

    n_faces = int(n_out.shape[0])
    out = np.zeros((n_faces, 3), dtype=float)
    if n_faces == 0:
        return out

    if np.isscalar(shielded):
        shielded_arr = np.full(n_faces, bool(shielded), dtype=bool)
    else:
        shielded_arr = np.asarray(shielded, dtype=bool)
        if shielded_arr.shape != (n_faces,):
            raise ValueError("shielded must be scalar or shape (N,).")

    active = ~shielded_arr
    if not np.any(active):
        return out

    windward_eq = _resolve_windward_equation(windward_eq)
    leeward_eq = _resolve_leeward_equation(leeward_eq)
    n_in = -n_out[active]
    gamma_n = n_in @ Vhat
    windward = gamma_n > 0.0
    cp = np.zeros_like(gamma_n)
    if windward_eq in {"newtonian", "modified_newtonian"}:
        cp[windward] = float(cp_max) * np.square(gamma_n[windward])
    elif windward_eq == "tangent_wedge":
        if Mach is None or gamma is None:
            raise ValueError("Mach and gamma are required for windward_eq=tangent_wedge.")
        windward_idx = np.where(windward)[0]
        gamma_n_win = np.clip(gamma_n[windward_idx], -1.0, 1.0)
        for i, gn in zip(windward_idx, gamma_n_win):
            cp[i] = tangent_wedge_pressure_coefficient(
                Mach=float(Mach),
                gamma=float(gamma),
                deltar=float(math.asin(float(gn))),
                cp_cap=float(cp_max),
            )
    if leeward_eq == "newtonian_mirror":
        cp[~windward] = float(cp_max) * np.square(gamma_n[~windward])
    elif leeward_eq == "prandtl_meyer":
        if Mach is None or gamma is None:
            raise ValueError("Mach and gamma are required for leeward_eq=prandtl_meyer.")
        leeward_idx = np.where(~windward)[0]
        gamma_n_lee = np.clip(gamma_n[leeward_idx], -1.0, 1.0)
        for i, gn in zip(leeward_idx, gamma_n_lee):
            cp[i] = prandtl_meyer_pressure_coefficient(
                Mach=float(Mach),
                gamma=float(gamma),
                deltar=float(math.asin(float(gn))),
            )

    out_active = np.zeros_like(n_in)
    nonzero = np.abs(cp) > 0.0
    if np.any(nonzero):
        out_active[nonzero] = -(cp[nonzero, None] / float(Aref)) * n_out[active][nonzero]
    out[active] = out_active
    return out

def stl_to_body(v_stl: np.ndarray) -> np.ndarray:
    """Convert a vector from STL axes to body axes.

    Axis mapping is ``body = (-x_stl, +y_stl, -z_stl)``.
    """
    v = np.asarray(v_stl, dtype=float)
    return np.array([-v[0], v[1], -v[2]], dtype=float)


def rot_y(alpha_rad: float) -> np.ndarray:
    """Return right-handed rotation matrix about +Y by ``alpha_rad``.

    Args:
        alpha_rad: Rotation angle [rad].

    Returns:
        3x3 rotation matrix.
    """
    c = math.cos(alpha_rad)
    s = math.sin(alpha_rad)
    return np.array(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ],
        dtype=float,
    )
