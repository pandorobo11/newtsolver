from __future__ import annotations

"""Newtonian pressure model and coordinate transforms."""

import math

import numpy as np

ATTITUDE_INPUT_VALUES = {"beta_tan", "beta_sin", "bank"}

def _resolve_attitude_mode(attitude_input: str | None) -> str:
    """Return canonical attitude mode with default and validation."""
    mode = str(attitude_input or "").strip().lower() or "beta_tan"
    if mode not in ATTITUDE_INPUT_VALUES:
        raise ValueError(
            f"Invalid attitude_input: '{attitude_input}'. "
            "Expected one of: beta_tan, beta_sin, bank."
        )
    return mode


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

    gamma_n = float(np.dot(Vhat, n_in))
    if gamma_n <= 0.0:
        return np.zeros(3, dtype=float)

    cp = float(cp_max) * (gamma_n * gamma_n)
    return -(cp / float(Aref)) * n_out


def newtonian_dC_dA_vectors(
    Vhat: np.ndarray,
    n_out: np.ndarray,
    Aref: float,
    shielded: np.ndarray | bool = False,
    cp_max: float = 2.0,
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

    n_in = -n_out[active]
    gamma_n = n_in @ Vhat
    windward = gamma_n > 0.0
    if np.any(windward):
        cp = float(cp_max) * np.square(gamma_n[windward])
        out_active = np.zeros_like(n_in)
        out_active[windward] = -(cp[:, None] / float(Aref)) * n_out[active][windward]
        out[active] = out_active
    return out


def sentman_dC_dA_vector(
    Vhat: np.ndarray,
    n_out: np.ndarray,
    S: float,
    Ti: float,
    Tw: float,
    Aref: float,
    shielded: bool = False,
) -> np.ndarray:
    """Backward-compatible alias for the Newtonian panel model."""
    _ = (S, Ti, Tw)
    return newtonian_dC_dA_vector(Vhat=Vhat, n_out=n_out, Aref=Aref, shielded=shielded)


def sentman_dC_dA_vectors(
    Vhat: np.ndarray,
    n_out: np.ndarray,
    S: float,
    Ti: float,
    Tw: float,
    Aref: float,
    shielded: np.ndarray | bool = False,
) -> np.ndarray:
    """Backward-compatible alias for the Newtonian panel model."""
    _ = (S, Ti, Tw)
    return newtonian_dC_dA_vectors(Vhat=Vhat, n_out=n_out, Aref=Aref, shielded=shielded)


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
