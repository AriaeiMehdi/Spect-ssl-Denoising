from __future__ import annotations

import unittest

import numpy as np

from gated_spect.data.exposure import ExposureConfig, ExposureError, estimate_exposure


class ExposureTests(unittest.TestCase):
    def test_fallback_normalizes_per_view_and_flags_low_gate(self) -> None:
        gates, views, height, width = 4, 5, 4, 4
        gate_factor = np.array([1.0, 0.9, 1.1, 0.1])
        view_factor = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
        counts = np.zeros((gates, views, height, width), dtype=np.int32)
        for g in range(gates):
            for v in range(views):
                counts[g, v] = int(round(10 * gate_factor[g] * view_factor[v]))
        result = estimate_exposure(
            counts,
            np.full(views, 2.0),
            config=ExposureConfig(low_information_q=0.20),
        )
        self.assertEqual(result.source, "derived")
        np.testing.assert_allclose(result.frame_exposure_s.sum(axis=0), 2.0, rtol=1e-6)
        self.assertTrue(np.any(result.relative_exposure_q[3] < 0.20))
        self.assertTrue(result.stable_gates == (0, 1, 2))

    def test_vendor_exposure_is_not_renormalized(self) -> None:
        counts = np.ones((2, 3, 2, 2), dtype=np.int32)
        vendor = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = estimate_exposure(counts, [1.0, 1.0, 1.0], vendor_exposure_s=vendor)
        self.assertEqual(result.source, "vendor")
        np.testing.assert_array_equal(result.frame_exposure_s, vendor.astype(np.float32))

    def test_invalid_dwell_is_rejected(self) -> None:
        with self.assertRaises(ExposureError):
            estimate_exposure(np.ones((2, 2, 2, 2), dtype=np.int32), [1.0])


if __name__ == "__main__":
    unittest.main()
