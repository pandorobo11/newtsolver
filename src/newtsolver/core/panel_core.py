from __future__ import annotations

"""Compatibility import surface for core panel computations.

This module intentionally re-exports symbols from smaller submodules.
"""

from .attitude import (
    ATTITUDE_INPUT_VALUES,
    _resolve_attitude_mode,
    resolve_attitude_to_vhat,
    rot_y,
    stl_to_body,
)
from .panel_forces import (
    LEEWARD_EQUATION_VALUES,
    WINDWARD_EQUATION_VALUES,
    _resolve_leeward_equation,
    _resolve_windward_equation,
    newtonian_dC_dA_vector,
    newtonian_dC_dA_vectors,
)
from .pressure_models.modified_newtonian import modified_newtonian_cp_max
from .pressure_models.prandtl_meyer import (
    _inverse_prandtl_meyer,
    _prandtl_meyer_nu,
    prandtl_meyer_pressure_coefficient,
)
from .pressure_models.tangent_wedge import (
    _oblique_theta_from_beta,
    _real_cuberoot,
    _tangent_wedge_detach_limit,
    _weak_oblique_shock_beta,
    tangent_wedge_pressure_coefficient,
)

__all__ = [
    "ATTITUDE_INPUT_VALUES",
    "WINDWARD_EQUATION_VALUES",
    "LEEWARD_EQUATION_VALUES",
    "_resolve_attitude_mode",
    "_resolve_windward_equation",
    "_resolve_leeward_equation",
    "modified_newtonian_cp_max",
    "_oblique_theta_from_beta",
    "_real_cuberoot",
    "_tangent_wedge_detach_limit",
    "_weak_oblique_shock_beta",
    "tangent_wedge_pressure_coefficient",
    "_prandtl_meyer_nu",
    "_inverse_prandtl_meyer",
    "prandtl_meyer_pressure_coefficient",
    "resolve_attitude_to_vhat",
    "newtonian_dC_dA_vector",
    "newtonian_dC_dA_vectors",
    "stl_to_body",
    "rot_y",
]
