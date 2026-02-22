from __future__ import annotations

"""Panel force-coefficient assembly from local pressure models."""

import math

import numpy as np

from .pressure_models import (
    prandtl_meyer_pressure_coefficient,
    prandtl_meyer_pressure_coefficients,
    tangent_wedge_pressure_coefficient,
)

WINDWARD_EQUATION_VALUES = {"newtonian", "modified_newtonian", "tangent_wedge"}
LEEWARD_EQUATION_VALUES = {"shield", "prandtl_meyer"}


def _resolve_windward_equation(value: str | None) -> str:
    """Normalize windward equation selector to canonical keyword."""
    eq = str(value or "").strip().lower() or "newtonian"
    if eq not in WINDWARD_EQUATION_VALUES:
        raise ValueError(
            f"Invalid windward_eq: '{value}'. "
            "Expected one of: newtonian, modified_newtonian, tangent_wedge."
        )
    return eq


def _resolve_leeward_equation(value: str | None) -> str:
    """Normalize leeward equation selector to canonical keyword."""
    eq = str(value or "").strip().lower() or "shield"
    if eq not in LEEWARD_EQUATION_VALUES:
        raise ValueError(
            f"Invalid leeward_eq: '{value}'. "
            "Expected one of: shield, prandtl_meyer."
        )
    return eq


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
    """Compute panel force-coefficient density vector ``dC/dA``.

    Windward faces use the configured windward equation.
    Leeward faces use either `shield` (`Cp=0`) or `prandtl_meyer`.
    Shielded faces are always zeroed.
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
    """Compute Newtonian ``dC/dA`` for multiple panels in one call."""
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
    if leeward_eq == "prandtl_meyer":
        if Mach is None or gamma is None:
            raise ValueError("Mach and gamma are required for leeward_eq=prandtl_meyer.")
        leeward_idx = np.where(~windward)[0]
        gamma_n_lee = np.clip(gamma_n[leeward_idx], -1.0, 1.0)
        deltar_lee = np.arcsin(gamma_n_lee)
        cp[leeward_idx] = prandtl_meyer_pressure_coefficients(
            Mach=float(Mach),
            gamma=float(gamma),
            deltar=deltar_lee,
        )

    out_active = np.zeros_like(n_in)
    nonzero = np.abs(cp) > 0.0
    if np.any(nonzero):
        out_active[nonzero] = -(cp[nonzero, None] / float(Aref)) * n_out[active][nonzero]
    out[active] = out_active
    return out
