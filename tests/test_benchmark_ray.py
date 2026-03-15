from __future__ import annotations

import importlib.util
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd


_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "benchmark_solver.py"
_SPEC = importlib.util.spec_from_file_location("newtsolver_benchmark_solver", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
benchmark_solver = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(benchmark_solver)


class TestBenchmarkSolverDefaults(unittest.TestCase):
    def test_peak_rss_pair_returns_nan_when_resource_is_unavailable(self):
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "resource":
                raise ImportError("resource unavailable")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            rss_self_mib, rss_children_mib = benchmark_solver._peak_rss_pair_mib()

        self.assertTrue(math.isnan(rss_self_mib))
        self.assertTrue(math.isnan(rss_children_mib))

    def test_default_metrics_output_uses_input_file_directory(self):
        df_in = pd.DataFrame([{"case_id": "case_a", "stl_path": "samples/stl/cube.stl"}])

        def fake_run_cases(_df_run, logfn, workers):
            logfn("ignored")
            self.assertEqual(workers, 1)
            return pd.DataFrame(
                [{"case_id": "case_a", "scope": "total", "component_id": "", "run_elapsed_s": 0.1}]
            )

        with tempfile.TemporaryDirectory(prefix="newtsolver_benchmark_") as td:
            input_dir = Path(td) / "inputs"
            input_dir.mkdir()
            input_path = input_dir / "input.csv"
            input_path.write_text("placeholder\n", encoding="utf-8")

            with (
                patch.object(benchmark_solver, "read_cases", return_value=df_in),
                patch.object(benchmark_solver, "run_cases", side_effect=fake_run_cases),
                patch("builtins.print"),
            ):
                rc = benchmark_solver.main(["--input", str(input_path)])

            self.assertEqual(rc, 0)
            self.assertTrue((input_dir / "outputs" / "benchmark_metrics.csv").exists())

    def test_default_result_prefix_uses_input_file_directory(self):
        df_in = pd.DataFrame([{"case_id": "case_a", "stl_path": "samples/stl/cube.stl"}])
        captured: dict[str, Path] = {}

        def fake_run_cases(_df_run, logfn, workers):
            logfn("ignored")
            self.assertEqual(workers, 1)
            return pd.DataFrame(
                [{"case_id": "case_a", "scope": "total", "component_id": "", "run_elapsed_s": 0.1}]
            )

        def fake_write_results_csv(out_path, _df_in, _df_out):
            captured["out_path"] = Path(out_path).resolve()

        with tempfile.TemporaryDirectory(prefix="newtsolver_benchmark_") as td:
            input_dir = Path(td) / "inputs"
            input_dir.mkdir()
            input_path = input_dir / "input.csv"
            input_path.write_text("placeholder\n", encoding="utf-8")

            with (
                patch.object(benchmark_solver, "read_cases", return_value=df_in),
                patch.object(benchmark_solver, "run_cases", side_effect=fake_run_cases),
                patch.object(benchmark_solver, "write_results_csv", side_effect=fake_write_results_csv),
                patch("builtins.print"),
            ):
                rc = benchmark_solver.main(["--input", str(input_path), "--write-results"])

            self.assertEqual(rc, 0)
            self.assertEqual(
                captured["out_path"],
                (input_dir / "outputs" / "benchmark_result_01.csv").resolve(),
            )


if __name__ == "__main__":
    unittest.main()
