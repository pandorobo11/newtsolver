from __future__ import annotations

import math
import unittest

import numpy as np

from newtsolver.core.panel_core import (
    _inverse_prandtl_meyer,
    _prandtl_meyer_nu,
    _tangent_wedge_detach_limit,
    modified_newtonian_cp_max,
    newtonian_dC_dA_vector,
    newtonian_dC_dA_vectors,
    prandtl_meyer_pressure_coefficient,
    resolve_attitude_to_vhat,
    stl_to_body,
    tangent_wedge_pressure_coefficient,
)


class TestPanelCore(unittest.TestCase):
    def test_vhat_matches_equivalent_tan_form(self):
        angles = [(-70.0, -30.0), (-25.0, 10.0), (0.0, 0.0), (35.0, -15.0), (70.0, 30.0)]
        for alpha_deg, beta_deg in angles:
            v, _, _, mode = resolve_attitude_to_vhat(alpha_deg, beta_deg, "beta_tan")

            a = math.radians(alpha_deg)
            b = math.radians(beta_deg)
            v_ref = np.array([1.0, -math.tan(b), math.tan(a)], dtype=float)
            v_ref /= np.linalg.norm(v_ref)

            self.assertEqual(mode, "beta_tan")
            np.testing.assert_allclose(v, v_ref, rtol=0.0, atol=1e-12)
            self.assertAlmostEqual(float(np.linalg.norm(v)), 1.0, places=12)

    def test_newtonian_vector_is_zero_when_shielded(self):
        v = newtonian_dC_dA_vector(
            Vhat=np.array([1.0, 0.0, 0.0]),
            n_out=np.array([-1.0, 0.0, 0.0]),
            Aref=1.0,
            shielded=True,
        )
        np.testing.assert_allclose(v, np.zeros(3), rtol=0.0, atol=0.0)

    def test_newtonian_vector_is_zero_on_leeward_face(self):
        v = newtonian_dC_dA_vector(
            Vhat=np.array([1.0, 0.0, 0.0]),
            n_out=np.array([1.0, 0.0, 0.0]),
            Aref=1.0,
            shielded=False,
        )
        np.testing.assert_allclose(v, np.zeros(3), rtol=0.0, atol=0.0)

    def test_modified_newtonian_cp_max_is_finite(self):
        cp_max = modified_newtonian_cp_max(Mach=6.0, gamma=1.4)
        self.assertTrue(math.isfinite(cp_max))
        self.assertGreater(cp_max, 0.0)
        self.assertLess(cp_max, 2.0)

    def test_modified_newtonian_windward_uses_given_cp_max(self):
        cp_max = modified_newtonian_cp_max(Mach=6.0, gamma=1.4)
        v = newtonian_dC_dA_vector(
            Vhat=np.array([1.0, 0.0, 0.0]),
            n_out=np.array([-1.0, 0.0, 0.0]),
            Aref=1.0,
            shielded=False,
            cp_max=cp_max,
            windward_eq="modified_newtonian",
        )
        np.testing.assert_allclose(v, np.array([cp_max, 0.0, 0.0]), rtol=0.0, atol=1e-12)

    def test_tangent_wedge_pressure_coefficient_is_positive_and_bounded(self):
        cp_cap = modified_newtonian_cp_max(Mach=6.0, gamma=1.4)
        cp = tangent_wedge_pressure_coefficient(
            Mach=6.0,
            gamma=1.4,
            deltar=math.radians(10.0),
            cp_cap=cp_cap,
        )
        self.assertGreater(cp, 0.0)
        self.assertLess(cp, cp_cap)

    def test_tangent_wedge_detached_bridges_to_cp_cap(self):
        mach = 2.0
        gamma = 1.4
        theta = math.radians(24.0)  # above detach threshold for M=2, gamma=1.4
        cp_cap = modified_newtonian_cp_max(Mach=mach, gamma=gamma)
        theta_max, cp_crit = _tangent_wedge_detach_limit(mach, gamma)

        cp = tangent_wedge_pressure_coefficient(
            Mach=mach,
            gamma=gamma,
            deltar=theta,
            cp_cap=cp_cap,
        )
        self.assertGreater(cp, cp_crit)
        self.assertLess(cp, cp_cap)

        # At 90 deg, detached bridge should reach Cp_cap.
        cp_90 = tangent_wedge_pressure_coefficient(
            Mach=mach,
            gamma=gamma,
            deltar=math.radians(90.0),
            cp_cap=cp_cap,
        )
        self.assertAlmostEqual(cp_90, cp_cap, places=12)

        # No drop at detach boundary.
        cp_before = tangent_wedge_pressure_coefficient(
            Mach=mach,
            gamma=gamma,
            deltar=theta_max * (1.0 - 1e-6),
            cp_cap=cp_cap,
        )
        cp_after = tangent_wedge_pressure_coefficient(
            Mach=mach,
            gamma=gamma,
            deltar=theta_max * (1.0 + 1e-6),
            cp_cap=cp_cap,
        )
        self.assertGreaterEqual(cp_after + 1e-12, cp_before)

    def test_tangent_wedge_windward_generates_force(self):
        v = newtonian_dC_dA_vector(
            Vhat=np.array([1.0, 0.0, 0.0]),
            n_out=np.array([-1.0, 0.0, 0.0]),
            Aref=1.0,
            shielded=False,
            windward_eq="tangent_wedge",
            Mach=6.0,
            gamma=1.4,
            cp_max=modified_newtonian_cp_max(Mach=6.0, gamma=1.4),
        )
        self.assertGreater(float(v[0]), 0.0)
        self.assertLess(float(v[0]), 2.0)

    def test_removed_surface_equations_are_rejected(self):
        with self.assertRaises(ValueError):
            newtonian_dC_dA_vector(
                Vhat=np.array([1.0, 0.0, 0.0]),
                n_out=np.array([-1.0, 0.0, 0.0]),
                Aref=1.0,
                windward_eq="shield",
            )
        with self.assertRaises(ValueError):
            newtonian_dC_dA_vector(
                Vhat=np.array([1.0, 0.0, 0.0]),
                n_out=np.array([1.0, 0.0, 0.0]),
                Aref=1.0,
                leeward_eq="newtonian_mirror",
            )

    def test_prandtl_meyer_leeward_pressure_is_negative_and_bounded(self):
        cp = prandtl_meyer_pressure_coefficient(Mach=6.0, gamma=1.4, deltar=math.radians(-10.0))
        cp_vac = -2.0 / (1.4 * 6.0 * 6.0)
        self.assertLess(cp, 0.0)
        self.assertGreaterEqual(cp, cp_vac)

    def test_leeward_prandtl_meyer_generates_force(self):
        v = newtonian_dC_dA_vector(
            Vhat=np.array([1.0, 0.0, 0.0]),
            n_out=np.array([1.0, 0.0, 0.0]),
            Aref=1.0,
            shielded=False,
            leeward_eq="prandtl_meyer",
            Mach=6.0,
            gamma=1.4,
        )
        self.assertGreater(float(v[0]), 0.0)
        self.assertLess(float(v[0]), 2.0)

    def test_inverse_prandtl_meyer_regression_no_low_nu_oscillation(self):
        gamma = 1.67
        m_true = np.array([1.0571513513513513], dtype=float)
        nu = _prandtl_meyer_nu(m_true, gamma)
        m_est = _inverse_prandtl_meyer(nu, gamma)
        self.assertAlmostEqual(float(m_est[0]), float(m_true[0]), places=9)

    def test_stl_to_body_axis_mapping(self):
        v_stl = np.array([2.0, -3.0, 4.5], dtype=float)
        v_body = stl_to_body(v_stl)
        np.testing.assert_allclose(v_body, np.array([-2.0, -3.0, -4.5]), rtol=0.0, atol=0.0)

    def test_newtonian_vectors_matches_scalar(self):
        Vhat, _, _, _ = resolve_attitude_to_vhat(10.0, -5.0, "beta_tan")
        n_out = np.array(
            [
                [-1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, -1.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=float,
        )
        shielded = np.array([False, True, False, False], dtype=bool)
        vec = newtonian_dC_dA_vectors(
            Vhat=Vhat,
            n_out=n_out,
            Aref=2.0,
            shielded=shielded,
        )
        ref = np.vstack(
            [
                newtonian_dC_dA_vector(
                    Vhat=Vhat,
                    n_out=n_out[i],
                    Aref=2.0,
                    shielded=bool(shielded[i]),
                )
                for i in range(n_out.shape[0])
            ]
        )
        np.testing.assert_allclose(vec, ref, rtol=0.0, atol=1e-13)

    def test_resolve_attitude_beta_tan_matches_direct(self):
        v1, a_t, b_t, mode = resolve_attitude_to_vhat(10.0, -5.0, "beta_tan")
        v2, _, _, _ = resolve_attitude_to_vhat(10.0, -5.0, "")
        self.assertEqual(mode, "beta_tan")
        self.assertAlmostEqual(a_t, 10.0, places=12)
        self.assertAlmostEqual(b_t, -5.0, places=12)
        np.testing.assert_allclose(v1, v2, rtol=0.0, atol=1e-12)

    def test_resolve_attitude_bank_and_beta_sin(self):
        # Ground truth from bank definition.
        v_ref, _, _, _ = resolve_attitude_to_vhat(30.0, 25.0, "bank")

        # Build equivalent mixed (alpha_t, beta_s) input from the same vector.
        alpha_t = math.degrees(math.atan2(float(v_ref[2]), float(v_ref[0])))
        beta_s = math.degrees(math.asin(float(-v_ref[1])))
        v_from_beta_sin, _, _, mode = resolve_attitude_to_vhat(alpha_t, beta_s, "beta_sin")

        self.assertEqual(mode, "beta_sin")
        np.testing.assert_allclose(v_from_beta_sin, v_ref, rtol=0.0, atol=1e-12)

    def test_resolve_attitude_rejects_non_canonical_attitude_input(self):
        _, _, _, mode_default = resolve_attitude_to_vhat(0.0, 0.0, "")
        self.assertEqual(mode_default, "beta_tan")
        with self.assertRaises(ValueError):
            resolve_attitude_to_vhat(0.0, 0.0, "βsin定義")
        with self.assertRaises(ValueError):
            resolve_attitude_to_vhat(0.0, 0.0, "unknown_mode")


if __name__ == "__main__":
    unittest.main()
