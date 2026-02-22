"""Pressure-model equations used by panel-force assembly."""

from .modified_newtonian import modified_newtonian_cp_max
from .prandtl_meyer import (
    _inverse_prandtl_meyer,
    _prandtl_meyer_nu,
    prandtl_meyer_pressure_coefficient,
    prandtl_meyer_pressure_coefficients,
)
from .tangent_wedge import (
    _oblique_theta_from_beta,
    _tangent_wedge_detach_limit,
    _weak_oblique_shock_beta,
    tangent_wedge_pressure_coefficient,
)

__all__ = [
    "modified_newtonian_cp_max",
    "_prandtl_meyer_nu",
    "_inverse_prandtl_meyer",
    "prandtl_meyer_pressure_coefficient",
    "prandtl_meyer_pressure_coefficients",
    "_oblique_theta_from_beta",
    "_tangent_wedge_detach_limit",
    "_weak_oblique_shock_beta",
    "tangent_wedge_pressure_coefficient",
]
