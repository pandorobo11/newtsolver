from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import pyvista as pv

from fmfsolver.core.solver import build_case_signature, run_case, run_cases
from fmfsolver.io.csv_out import write_results_csv


class TestSolverPipeline(unittest.TestCase):
    def test_build_case_signature_numeric_normalization(self):
        row_int = {
            "case_id": "sig_case",
            "stl_path": "samples/stl/cube.stl",
            "stl_scale_m_per_unit": 1,
            "alpha_deg": 5,
            "beta_deg": 0,
            "Tw_K": 300,
            "ref_x_m": 0,
            "ref_y_m": 0,
            "ref_z_m": 0,
            "Aref_m2": 1,
            "Lref_Cl_m": 1,
            "Lref_Cm_m": 1,
            "Lref_Cn_m": 1,
            "S": 5,
            "Ti_K": 300,
            "shielding_on": 0,
        }
        row_float = {
            **row_int,
            "stl_scale_m_per_unit": 1.0,
            "alpha_deg": 5.0,
            "Tw_K": 300.0,
            "Aref_m2": 1.0,
            "S": 5.0,
            "Ti_K": 300.0,
        }
        self.assertEqual(build_case_signature(row_int), build_case_signature(row_float))

    def test_run_case_mode_a_smoke(self):
        with tempfile.TemporaryDirectory(prefix="fmfsolver_test_") as td:
            row = {
                "case_id": "test_mode_a",
                "stl_path": "samples/stl/cube.stl",
                "stl_scale_m_per_unit": 1.0,
                "alpha_deg": 5.0,
                "beta_deg": 0.0,
                "Tw_K": 300.0,
                "ref_x_m": 0.0,
                "ref_y_m": 0.0,
                "ref_z_m": 0.0,
                "Aref_m2": 1.0,
                "Lref_Cl_m": 1.0,
                "Lref_Cm_m": 1.0,
                "Lref_Cn_m": 1.0,
                "S": 5.0,
                "Ti_K": 300.0,
                "shielding_on": 0,
                "save_vtp_on": 0,
                "save_npz_on": 0,
                "out_dir": td,
            }
            result = run_case(row, lambda _msg: None)

            self.assertEqual(result["mode"], "A")
            self.assertGreater(result["faces"], 0)
            self.assertEqual(result["vtp_path"], "")
            self.assertEqual(result["npz_path"], "")
            self.assertEqual(result["case_signature"], build_case_signature(row))
            self.assertTrue(str(result["solver_version"]).strip() != "")
            self.assertTrue(str(result["run_started_at_utc"]).endswith("Z"))
            self.assertTrue(str(result["run_finished_at_utc"]).endswith("Z"))
            self.assertGreaterEqual(float(result["run_elapsed_s"]), 0.0)
            for key in ("CA", "CY", "CN", "Cl", "Cm", "Cn", "CD", "CL"):
                self.assertTrue(math.isfinite(float(result[key])), key)
            self.assertTrue(Path(td).exists())

    def test_run_case_mode_b_smoke(self):
        with tempfile.TemporaryDirectory(prefix="fmfsolver_test_") as td:
            row = {
                "case_id": "test_mode_b",
                "stl_path": "samples/stl/plate.stl",
                "stl_scale_m_per_unit": 1.0,
                "alpha_deg": 5.0,
                "beta_deg": 0.0,
                "Tw_K": 300.0,
                "ref_x_m": 0.0,
                "ref_y_m": 0.0,
                "ref_z_m": 0.0,
                "Aref_m2": 1.0,
                "Lref_Cl_m": 1.0,
                "Lref_Cm_m": 1.0,
                "Lref_Cn_m": 1.0,
                "Mach": 25.0,
                "Altitude_km": 100.0,
                "shielding_on": 0,
                "save_vtp_on": 0,
                "save_npz_on": 0,
                "out_dir": td,
            }
            result = run_case(row, lambda _msg: None)

            self.assertEqual(result["mode"], "B")
            self.assertGreater(result["faces"], 0)
            self.assertEqual(result["vtp_path"], "")
            self.assertEqual(result["npz_path"], "")
            self.assertEqual(result["case_signature"], build_case_signature(row))
            self.assertTrue(str(result["solver_version"]).strip() != "")
            self.assertTrue(str(result["run_started_at_utc"]).endswith("Z"))
            self.assertTrue(str(result["run_finished_at_utc"]).endswith("Z"))
            self.assertGreaterEqual(float(result["run_elapsed_s"]), 0.0)
            for key in ("S", "Ti_K", "CA", "CY", "CN", "CD", "CL"):
                self.assertTrue(math.isfinite(float(result[key])), key)

    def test_run_cases_cancel_before_start(self):
        df = pd.DataFrame(
            [
                {
                    "case_id": "cancel_case",
                    "stl_path": "samples/stl/cube.stl",
                    "stl_scale_m_per_unit": 1.0,
                    "alpha_deg": 0.0,
                    "beta_deg": 0.0,
                    "Tw_K": 300.0,
                    "ref_x_m": 0.0,
                    "ref_y_m": 0.0,
                    "ref_z_m": 0.0,
                    "Aref_m2": 1.0,
                    "Lref_Cl_m": 1.0,
                    "Lref_Cm_m": 1.0,
                    "Lref_Cn_m": 1.0,
                    "S": 5.0,
                    "Ti_K": 300.0,
                    "shielding_on": 0,
                    "save_vtp_on": 0,
                    "save_npz_on": 0,
                    "out_dir": ".",
                }
            ]
        )
        with self.assertRaisesRegex(RuntimeError, "Canceled by user."):
            run_cases(df, lambda _msg: None, workers=1, cancel_cb=lambda: True)

    def test_run_cases_progress_callback(self):
        with tempfile.TemporaryDirectory(prefix="fmfsolver_test_") as td:
            df = pd.DataFrame(
                [
                    {
                        "case_id": "progress_case",
                        "stl_path": "samples/stl/cube.stl",
                        "stl_scale_m_per_unit": 1.0,
                        "alpha_deg": 0.0,
                        "beta_deg": 0.0,
                        "Tw_K": 300.0,
                        "ref_x_m": 0.0,
                        "ref_y_m": 0.0,
                        "ref_z_m": 0.0,
                        "Aref_m2": 1.0,
                        "Lref_Cl_m": 1.0,
                        "Lref_Cm_m": 1.0,
                        "Lref_Cn_m": 1.0,
                        "S": 5.0,
                        "Ti_K": 300.0,
                        "shielding_on": 0,
                        "save_vtp_on": 0,
                        "save_npz_on": 0,
                        "out_dir": td,
                    }
                ]
            )
            ticks: list[tuple[int, int]] = []
            _ = run_cases(
                df,
                lambda _msg: None,
                workers=1,
                progress_cb=lambda done, total: ticks.append((done, total)),
            )
            self.assertEqual(ticks, [(1, 1)])

    def test_run_cases_multi_stl_emits_total_and_component_rows(self):
        with tempfile.TemporaryDirectory(prefix="fmfsolver_test_") as td:
            df = pd.DataFrame(
                [
                    {
                        "case_id": "multi_case",
                        "stl_path": "samples/stl/cube.stl;samples/stl/plate_offset_x2.stl",
                        "stl_scale_m_per_unit": 1.0,
                        "alpha_deg": 5.0,
                        "beta_deg": 0.0,
                        "Tw_K": 300.0,
                        "ref_x_m": 0.0,
                        "ref_y_m": 0.0,
                        "ref_z_m": 0.0,
                        "Aref_m2": 1.0,
                        "Lref_Cl_m": 1.0,
                        "Lref_Cm_m": 1.0,
                        "Lref_Cn_m": 1.0,
                        "S": 5.0,
                        "Ti_K": 300.0,
                        "shielding_on": 0,
                        "save_vtp_on": 0,
                        "save_npz_on": 0,
                        "out_dir": td,
                    }
                ]
            )
            res = run_cases(df, lambda _msg: None, workers=1)
            total_rows = res[res["scope"] == "total"].reset_index(drop=True)
            component_rows = res[res["scope"] == "component"].reset_index(drop=True)

            self.assertEqual(len(total_rows), 1)
            self.assertEqual(len(component_rows), 2)
            self.assertEqual(set(component_rows["component_id"].astype(int).tolist()), {0, 1})
            self.assertEqual(int(total_rows.loc[0, "faces"]), int(component_rows["faces"].sum()))
            self.assertEqual(
                int(total_rows.loc[0, "shielded_faces"]),
                int(component_rows["shielded_faces"].sum()),
            )

            for coef in ("CA", "CY", "CN", "Cl", "Cm", "Cn", "CD", "CL"):
                self.assertAlmostEqual(
                    float(total_rows.loc[0, coef]),
                    float(component_rows[coef].sum()),
                    places=12,
                    msg=coef,
                )

    def test_write_results_csv_keeps_component_rows(self):
        with tempfile.TemporaryDirectory(prefix="fmfsolver_test_") as td:
            df = pd.DataFrame(
                [
                    {
                        "case_id": "multi_case",
                        "stl_path": "samples/stl/cube.stl;samples/stl/plate_offset_x2.stl",
                        "stl_scale_m_per_unit": 1.0,
                        "alpha_deg": 0.0,
                        "beta_deg": 0.0,
                        "Tw_K": 300.0,
                        "ref_x_m": 0.0,
                        "ref_y_m": 0.0,
                        "ref_z_m": 0.0,
                        "Aref_m2": 1.0,
                        "Lref_Cl_m": 1.0,
                        "Lref_Cm_m": 1.0,
                        "Lref_Cn_m": 1.0,
                        "S": 5.0,
                        "Ti_K": 300.0,
                        "shielding_on": 0,
                        "save_vtp_on": 0,
                        "save_npz_on": 0,
                        "out_dir": td,
                    }
                ]
            )
            res = run_cases(df, lambda _msg: None, workers=1)
            out_csv = Path(td) / "result.csv"
            write_results_csv(str(out_csv), df, res)
            out_df = pd.read_csv(out_csv)

            self.assertEqual(len(out_df), len(res))
            self.assertIn("scope", out_df.columns)
            self.assertEqual(int((out_df["scope"] == "total").sum()), 1)
            self.assertEqual(int((out_df["scope"] == "component").sum()), 2)

    def test_run_case_multi_stl_vtp_has_component_metadata(self):
        with tempfile.TemporaryDirectory(prefix="fmfsolver_test_") as td:
            row = {
                "case_id": "multi_case_vtp",
                "stl_path": "samples/stl/cube.stl;samples/stl/plate_offset_x2.stl",
                "stl_scale_m_per_unit": 1.0,
                "alpha_deg": 5.0,
                "beta_deg": 0.0,
                "Tw_K": 300.0,
                "ref_x_m": 0.0,
                "ref_y_m": 0.0,
                "ref_z_m": 0.0,
                "Aref_m2": 1.0,
                "Lref_Cl_m": 1.0,
                "Lref_Cm_m": 1.0,
                "Lref_Cn_m": 1.0,
                "S": 5.0,
                "Ti_K": 300.0,
                "shielding_on": 0,
                "save_vtp_on": 1,
                "save_npz_on": 0,
                "out_dir": td,
            }
            result = run_case(row, lambda _msg: None)
            poly = pv.read(result["vtp_path"])

            self.assertIn("stl_index", poly.cell_data)
            self.assertEqual(len(poly.cell_data["stl_index"]), int(result["faces"]))
            self.assertIn("stl_count", poly.field_data)
            self.assertEqual(int(poly.field_data["stl_count"][0]), 2)
            self.assertIn("stl_paths_json", poly.field_data)


if __name__ == "__main__":
    unittest.main()
