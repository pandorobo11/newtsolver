from __future__ import annotations

"""Modified-Newtonian pressure relations."""

import math


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
