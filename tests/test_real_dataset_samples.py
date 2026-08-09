from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from gated_spect.data.dicom_reader import read_nm_dicom
from gated_spect.types import ReaderConfig


DATA_ROOT = Path(r"D:\Dr.EntezarMahdi\3DRecon\datasets")
HAS_DATA = (DATA_ROOT / "Denoising dataset").exists() and (DATA_ROOT / "Pre-training dataset").exists()


@unittest.skipUnless(HAS_DATA, "workspace datasets are not mounted")
class RealDatasetSampleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = ReaderConfig(hash_key=b"local-test-only", institution_id="test")

    def test_denoising_is_explicitly_ungated_and_uses_nested_geometry(self) -> None:
        path = DATA_ROOT / "Denoising dataset" / "00001 10sec.dcm"
        sample = read_nm_dicom(path, self.config)
        self.assertEqual(sample.shape, (1, 32, 64, 64))
        self.assertEqual(sample.metadata["gating_mode"], "ungated")
        self.assertEqual(sample.metadata["radius_source"], "detector_sequence")
        self.assertTrue(np.all(sample.radius_valid))
        np.testing.assert_allclose(sample.view_dwell_s, np.full(32, 10.0, dtype=np.float32))
        self.assertEqual(sample.metadata["view_dwell_source"], "dicom_actual_frame_duration")
        self.assertEqual(sample.collimator_id, "LEHR")

    def test_pretraining_is_gated_with_valid_radius_sources(self) -> None:
        candidates = sorted((DATA_ROOT / "Pre-training dataset").glob("*.dcm"))
        path = candidates[0]
        sample = read_nm_dicom(path, self.config)
        self.assertEqual(sample.shape, (16, 32, 64, 64))
        self.assertEqual(sample.metadata["gating_mode"], "gated")
        self.assertEqual(sample.metadata["phase_source"], "slot_center")
        self.assertTrue(np.all(sample.radius_valid))
        self.assertTrue(np.isfinite(sample.energy_window_kev).all())
        self.assertAlmostEqual(float(np.median(sample.view_dwell_s)), 27.904, places=3)
        self.assertEqual(sample.metadata["view_dwell_source"], "dicom_actual_frame_duration")

        detector_radius_sample = read_nm_dicom(candidates[1], self.config)
        self.assertTrue(np.all(detector_radius_sample.radius_valid))
        self.assertEqual(detector_radius_sample.metadata["radius_source"], "detector_sequence")


if __name__ == "__main__":
    unittest.main()
