from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import trimesh

from newtsolver.core.solver import run_case


def _newtonian_flat_plate_ca(alpha_rad: float, cp_max: float = 2.0) -> float:
    ca = math.cos(alpha_rad)
    if ca <= 0.0:
        return 0.0
    return cp_max * (ca * ca)


def _write_one_sided_plate(path: Path) -> float:
    vertices = np.array(
        [
            [0.0, -0.5, -0.5],
            [0.0, +0.5, -0.5],
            [0.0, +0.5, +0.5],
            [0.0, -0.5, +0.5],
        ],
        dtype=float,
    )
    faces = np.array(
        [
            [0, 2, 1],
            [0, 3, 2],
        ],
        dtype=np.int64,
    )
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.export(path)
    return float(mesh.area)


class TestFlatPlateVerification(unittest.TestCase):
    def test_run_case_matches_newtonian_flat_plate_formula(self):
        alpha_deg_values = (0.0, 10.0, 30.0, 60.0)
        tol = 1e-10

        with tempfile.TemporaryDirectory(prefix="newtsolver_flat_plate_") as td:
            stl_path = Path(td) / "one_sided_plate.stl"
            area_m2 = _write_one_sided_plate(stl_path)

            for alpha_deg in alpha_deg_values:
                alpha_rad = math.radians(alpha_deg)
                ca_ref = _newtonian_flat_plate_ca(alpha_rad)
                row = {
                    "case_id": f"flat_a{alpha_deg:g}",
                    "stl_path": str(stl_path),
                    "stl_scale_m_per_unit": 1.0,
                    "Mach": 10.0,
                    "gamma": 1.4,
                    "alpha_deg": alpha_deg,
                    "beta_or_bank_deg": 0.0,
                    "ref_x_m": 0.0,
                    "ref_y_m": 0.0,
                    "ref_z_m": 0.0,
                    "Aref_m2": area_m2,
                    "Lref_Cl_m": 1.0,
                    "Lref_Cm_m": 1.0,
                    "Lref_Cn_m": 1.0,
                    "shielding_on": 0,
                    "save_vtp_on": 0,
                    "save_npz_on": 0,
                    "out_dir": td,
                }
                result = run_case(row, lambda _msg: None)
                cn_err = abs(float(result["CN"]) - 0.0)
                ca_err = abs(float(result["CA"]) - ca_ref)
                with self.subTest(alpha_deg=alpha_deg):
                    self.assertLessEqual(cn_err, tol)
                    self.assertLessEqual(ca_err, tol)


if __name__ == "__main__":
    unittest.main()
