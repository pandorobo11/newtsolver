from __future__ import annotations

import math
import unittest

import numpy as np

from fmfsolver.core.sentman_core import sentman_dC_dA_vector, stl_to_body, vhat_from_alpha_beta_stl


class TestSentmanCore(unittest.TestCase):
    def test_vhat_matches_equivalent_tan_form(self):
        angles = [(-70.0, -30.0), (-25.0, 10.0), (0.0, 0.0), (35.0, -15.0), (70.0, 30.0)]
        for alpha_deg, beta_deg in angles:
            v = vhat_from_alpha_beta_stl(alpha_deg, beta_deg)

            a = math.radians(alpha_deg)
            b = math.radians(beta_deg)
            v_ref = np.array([1.0, -math.tan(b), math.tan(a)], dtype=float)
            v_ref /= np.linalg.norm(v_ref)

            np.testing.assert_allclose(v, v_ref, rtol=0.0, atol=1e-12)
            self.assertAlmostEqual(float(np.linalg.norm(v)), 1.0, places=12)

    def test_sentman_vector_is_zero_when_shielded(self):
        v = sentman_dC_dA_vector(
            Vhat=np.array([1.0, 0.0, 0.0]),
            n_out=np.array([-1.0, 0.0, 0.0]),
            S=5.0,
            Ti=300.0,
            Tw=300.0,
            Aref=1.0,
            shielded=True,
        )
        np.testing.assert_allclose(v, np.zeros(3), rtol=0.0, atol=0.0)

    def test_stl_to_body_axis_mapping(self):
        v_stl = np.array([2.0, -3.0, 4.5], dtype=float)
        v_body = stl_to_body(v_stl)
        np.testing.assert_allclose(v_body, np.array([-2.0, -3.0, -4.5]), rtol=0.0, atol=0.0)


if __name__ == "__main__":
    unittest.main()
