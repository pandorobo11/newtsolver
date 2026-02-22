from __future__ import annotations

"""Plot Cp-theta comparison for tangent wedge vs modified Newtonian."""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from newtsolver.core.panel_core import (
    _tangent_wedge_detach_limit,
    modified_newtonian_cp_max,
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

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes = axes.ravel()

    legend_handles = None
    legend_labels = None
    for ax, (mach, gamma) in zip(axes, cases):
        cp_cap = modified_newtonian_cp_max(mach, gamma)
        theta_max, _cp_crit = _tangent_wedge_detach_limit(mach, gamma)
        theta_max_deg = math.degrees(theta_max)

        cp_tw = np.array(
            [
                tangent_wedge_pressure_coefficient(
                    mach,
                    gamma,
                    float(theta),
                    cp_cap=cp_cap,
                )
                for theta in theta_rad
            ],
            dtype=float,
        )
        cp_mn = cp_cap * np.sin(theta_rad) ** 2

        line_tw, = ax.plot(theta_deg, cp_tw, label="tangent_wedge", lw=2.2)
        line_mn, = ax.plot(theta_deg, cp_mn, label="modified_newtonian", lw=2.0, ls="--")
        line_th = ax.axvline(theta_max_deg, color="tab:red", ls=":", lw=1.6, label="theta_max")
        ax.fill_between(
            theta_deg,
            0.0,
            max(float(cp_tw.max()), float(cp_mn.max())) * 1.02,
            where=theta_deg >= theta_max_deg,
            color="tab:red",
            alpha=0.08,
        )

        if legend_handles is None:
            legend_handles = [line_tw, line_mn, line_th]
            legend_labels = [h.get_label() for h in legend_handles]

        ax.set_title(f"M={mach}, gamma={gamma}")
        ax.set_xlabel("theta [deg]")
        ax.set_ylabel("Cp")
        ax.grid(True, alpha=0.25)

    fig.suptitle("Cp vs Theta: Tangent Wedge vs Modified Newtonian")
    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=3,
        frameon=False,
    )

    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "cp_theta_tangent_wedge_vs_modified_newtonian.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    print(out_path)


if __name__ == "__main__":
    main()
