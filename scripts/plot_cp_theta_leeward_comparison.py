from __future__ import annotations

"""Plot Cp-theta comparison across leeward pressure models."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from newtsolver.core.pressure_models import prandtl_meyer_pressure_coefficient


def main() -> None:
    cases = [
        (2.0, 1.4),
        (6.0, 1.4),
        (10.0, 1.4),
        (6.0, 1.67),
    ]

    theta_deg = np.linspace(0.0, 90.0, 721)
    theta_rad = np.radians(theta_deg)
    deltar_lee = -theta_rad

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()

    legend_handles = None
    legend_labels = None
    for ax, (mach, gamma) in zip(axes, cases):
        cp_shield = np.zeros_like(theta_rad)
        cp_pm = prandtl_meyer_pressure_coefficient(
            Mach=mach,
            gamma=gamma,
            deltar=deltar_lee,
        )
        cp_vac = np.full_like(theta_rad, -2.0 / (gamma * mach * mach))

        line_shield, = ax.plot(theta_deg, cp_shield, lw=2.0, ls="--", label="shield")
        line_pm, = ax.plot(theta_deg, cp_pm, lw=2.2, label="prandtl_meyer")
        line_vac, = ax.plot(theta_deg, cp_vac, lw=1.4, ls=":", label="vacuum_limit")

        if legend_handles is None:
            legend_handles = [line_shield, line_pm, line_vac]
            legend_labels = [h.get_label() for h in legend_handles]

        ax.set_title(f"M={mach}, gamma={gamma}")
        ax.set_xlabel("theta [deg]")
        ax.set_ylabel("Cp")
        ax.grid(True, alpha=0.25)

    fig.suptitle("Cp vs Theta: Leeward Model Comparison", y=0.98)
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
    out_path = out_dir / "cp_theta_leeward_models.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    print(out_path)


if __name__ == "__main__":
    main()
