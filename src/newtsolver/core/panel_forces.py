from __future__ import annotations

"""Panel force-coefficient assembly from local pressure models."""

import numpy as np

from .pressure_models import (
    prandtl_meyer_pressure_coefficient,
    tangent_cone_pressure_coefficient,
    tangent_wedge_pressure_coefficient,
)

WINDWARD_EQUATION_VALUES = {"newtonian", "modified_newtonian", "tangent_wedge", "tangent_cone"}
LEEWARD_EQUATION_VALUES = {"shield", "prandtl_meyer"}


def _resolve_windward_equation(value: str | None) -> str:
    """Normalize windward equation selector to canonical keyword."""
    eq = str(value or "").strip().lower() or "newtonian"
    if eq not in WINDWARD_EQUATION_VALUES:
        raise ValueError(
            f"Invalid windward_eq: '{value}'. "
            "Expected one of: newtonian, modified_newtonian, tangent_wedge, tangent_cone."
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


def panel_force_density(
    Vhat: np.ndarray,
    n_out: np.ndarray,
    Aref: float,
    shielded: np.ndarray | bool = False,
    face_stl_index: np.ndarray | None = None,
    cp_max: float = 2.0,
    windward_eq: str = "newtonian",
    leeward_eq: str = "shield",
    windward_eq_by_component: list[str] | tuple[str, ...] | None = None,
    leeward_eq_by_component: list[str] | tuple[str, ...] | None = None,
    Mach: float | None = None,
    gamma: float | None = None,
) -> np.ndarray:
    """Return per-panel force-coefficient density vectors.

    Args:
        Vhat: Freestream unit vector in STL axes, shape ``(3,)``.
        n_out: Outward panel normals in STL axes, shape ``(N, 3)``.
        Aref: Reference area used for coefficient normalization.
        shielded: Panel shielding mask (scalar or shape ``(N,)``).
        face_stl_index: Source STL component index per panel, shape ``(N,)``.
        cp_max: Windward ``Cp`` cap used by Newtonian-family models.
        windward_eq: Windward pressure model keyword.
        leeward_eq: Leeward pressure model keyword.
        windward_eq_by_component: Optional per-component windward models.
        leeward_eq_by_component: Optional per-component leeward models.
        Mach: Freestream Mach number (required by non-Newtonian models).
        gamma: Specific-heat ratio (required by non-Newtonian models).

    Returns:
        Array of shape ``(N, 3)`` with panel ``dC/dA`` vectors.
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

    windward_eq_single = _resolve_windward_equation(windward_eq)
    leeward_eq_single = _resolve_leeward_equation(leeward_eq)
    if face_stl_index is None:
        comp_idx_all = np.zeros(n_faces, dtype=np.int32)
        n_components = 1
    else:
        comp_idx_all = np.asarray(face_stl_index, dtype=np.int32)
        if comp_idx_all.shape != (n_faces,):
            raise ValueError("face_stl_index must have shape (N,).")
        if np.any(comp_idx_all < 0):
            raise ValueError("face_stl_index must be non-negative.")
        n_components = int(comp_idx_all.max()) + 1 if n_faces > 0 else 1
        if n_components <= 0:
            n_components = 1

    if windward_eq_by_component is None:
        windward_models = [windward_eq_single] * n_components
    else:
        if len(windward_eq_by_component) != n_components:
            raise ValueError("windward_eq_by_component length must match component count.")
        windward_models = [_resolve_windward_equation(eq) for eq in windward_eq_by_component]

    if leeward_eq_by_component is None:
        leeward_models = [leeward_eq_single] * n_components
    else:
        if len(leeward_eq_by_component) != n_components:
            raise ValueError("leeward_eq_by_component length must match component count.")
        leeward_models = [_resolve_leeward_equation(eq) for eq in leeward_eq_by_component]

    n_in = -n_out[active]
    gamma_n = n_in @ Vhat
    cp = np.zeros_like(gamma_n)
    active_idx = np.where(active)[0]
    comp_idx_active = comp_idx_all[active_idx]
    for comp_i in range(n_components):
        local = comp_idx_active == comp_i
        if not np.any(local):
            continue

        gamma_local = gamma_n[local]
        cp_local = np.zeros_like(gamma_local)
        windward_local = gamma_local > 0.0

        windward_model = windward_models[comp_i]
        if windward_model == "newtonian":
            cp_local[windward_local] = 2.0 * np.square(gamma_local[windward_local])
        elif windward_model == "modified_newtonian":
            cp_local[windward_local] = float(cp_max) * np.square(gamma_local[windward_local])
        elif windward_model == "tangent_wedge":
            if Mach is None or gamma is None:
                raise ValueError("Mach and gamma are required for windward_eq=tangent_wedge.")
            gamma_n_win = np.clip(gamma_local[windward_local], -1.0, 1.0)
            deltar_win = np.arcsin(gamma_n_win)
            cp_local[windward_local] = tangent_wedge_pressure_coefficient(
                Mach=float(Mach),
                gamma=float(gamma),
                deltar=deltar_win,
                cp_cap=float(cp_max),
            )
        else:
            if Mach is None or gamma is None:
                raise ValueError("Mach and gamma are required for windward_eq=tangent_cone.")
            gamma_n_win = np.clip(gamma_local[windward_local], -1.0, 1.0)
            deltar_win = np.arcsin(gamma_n_win)
            cp_local[windward_local] = tangent_cone_pressure_coefficient(
                Mach=float(Mach),
                gamma=float(gamma),
                deltar=deltar_win,
                cp_cap=float(cp_max),
            )

        leeward_model = leeward_models[comp_i]
        if leeward_model == "prandtl_meyer":
            if Mach is None or gamma is None:
                raise ValueError("Mach and gamma are required for leeward_eq=prandtl_meyer.")
            leeward_local = ~windward_local
            gamma_n_lee = np.clip(gamma_local[leeward_local], -1.0, 1.0)
            deltar_lee = np.arcsin(gamma_n_lee)
            cp_local[leeward_local] = prandtl_meyer_pressure_coefficient(
                Mach=float(Mach),
                gamma=float(gamma),
                deltar=deltar_lee,
            )

        cp[local] = cp_local

    out_active = np.zeros_like(n_in)
    nonzero = np.abs(cp) > 0.0
    if np.any(nonzero):
        out_active[nonzero] = -(cp[nonzero, None] / float(Aref)) * n_out[active][nonzero]
    out[active] = out_active
    return out
