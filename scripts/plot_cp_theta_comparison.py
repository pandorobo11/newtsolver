from __future__ import annotations

"""Plot Cp-theta comparison across windward pressure models."""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from newtsolver.core.panel_core import (
    _tangent_cone_detach_limit,
    _tangent_wedge_detach_limit,
    modified_newtonian_cp_max,
    tangent_cone_pressure_coefficient,
    tangent_wedge_pressure_coefficient,
)


def main() -> None:
    cases = [
        (2.0, 1.4),
        (6.0, 1.4),
        (10.0, 1.4),
        (6.0, 1.67),
    ]

    theta_deg = np.linspace(0.0, 90.0, 721)
    theta_rad = np.radians(theta_deg)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()

    legend_handles = None
    legend_labels = None
    for ax, (mach, gamma) in zip(axes, cases):
        cp_cap = modified_newtonian_cp_max(mach, gamma)
        theta_max_tw, _cp_crit_tw = _tangent_wedge_detach_limit(mach, gamma)
        theta_max_tc, _cp_crit_tc = _tangent_cone_detach_limit(mach, gamma)
        theta_max_tw_deg = math.degrees(theta_max_tw)
        theta_max_tc_deg = math.degrees(theta_max_tc)

        cp_newton = 2.0 * np.sin(theta_rad) ** 2
        cp_mn = cp_cap * np.sin(theta_rad) ** 2
        cp_tw = tangent_wedge_pressure_coefficient(
            Mach=mach,
            gamma=gamma,
            deltar=theta_rad,
            cp_cap=cp_cap,
        )
        cp_tc = tangent_cone_pressure_coefficient(
            Mach=mach,
            gamma=gamma,
            deltar=theta_rad,
            cp_cap=cp_cap,
        )

        line_n, = ax.plot(theta_deg, cp_newton, label="newtonian", lw=2.0, ls=":")
        line_mn, = ax.plot(theta_deg, cp_mn, label="modified_newtonian", lw=2.0, ls="--")
        line_tw, = ax.plot(theta_deg, cp_tw, label="tangent_wedge", lw=2.2)
        line_tc, = ax.plot(theta_deg, cp_tc, label="tangent_cone", lw=2.0, ls="-.")
        line_tw_th = ax.axvline(theta_max_tw_deg, color="tab:red", ls=":", lw=1.4, label="theta_max_tw")
        line_tc_th = ax.axvline(
            theta_max_tc_deg,
            color="tab:purple",
            ls="--",
            lw=1.2,
            alpha=0.9,
            label="theta_max_tc",
        )

        if legend_handles is None:
            legend_handles = [line_n, line_mn, line_tw, line_tc, line_tw_th, line_tc_th]
            legend_labels = [h.get_label() for h in legend_handles]

        ax.set_title(f"M={mach}, gamma={gamma}")
        ax.set_xlabel("theta [deg]")
        ax.set_ylabel("Cp")
        ax.grid(True, alpha=0.25)

    fig.suptitle("Cp vs Theta: Windward Model Comparison", y=0.98)
    fig.subplots_adjust(top=0.88, bottom=0.14, wspace=0.20, hspace=0.30)
    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=3,
        frameon=False,
    )

    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "cp_theta_windward_models.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    print(out_path)


if __name__ == "__main__":
    main()
