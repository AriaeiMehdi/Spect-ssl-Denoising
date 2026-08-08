from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

import numpy as np

from gated_spect.data.pairing import synthesize_count_split, verify_pair
from gated_spect.data.pair_manifest import discover_duration_pairs
from gated_spect.types import GatedSPECTSample, PairProvenance


def make_sample(sample_id: str, counts: np.ndarray, exposure: np.ndarray) -> GatedSPECTSample:
    g, v = counts.shape[:2]
    return GatedSPECTSample(
        sample_id=sample_id,
        counts=counts.astype(np.int32),
        frame_exposure_s=exposure.astype(np.float32),
        relative_exposure_q=np.ones((g, v), dtype=np.float32),
        valid_frames=np.ones((g, v), dtype=bool),
        angles_rad=np.linspace(0, np.pi, v, endpoint=False, dtype=np.float32),
        radii_mm=np.full(v, 180.0, dtype=np.float32),
        radius_valid=np.ones(v, dtype=bool),
        detector_spacing_mm=(6.4, 6.4),
        view_dwell_s=np.ones(v, dtype=np.float32),
        gate_phase_rad=2 * np.pi * (np.arange(g) + 0.5) / g,
        energy_window_kev=(126.45, 154.55),
        scanner_id="scanner",
        collimator_id="LEHR",
        stress_rest="rest",
        detector_stream="detector_0",
        patient_key="patient",
        study_key=sample_id,
        geometry_hash="geometry",
        exposure_source="synthetic",
        source_path=sample_id,
    )


class PairingTests(unittest.TestCase):
    def test_duration_pair_discovery_is_strict(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("p1 10sec.dcm", "p1 30sec.dcm", "p2 10sec.dcm"):
                (root / name).write_bytes(b"fixture")
            pairs, incomplete = discover_duration_pairs(root)
            self.assertEqual([(pair.pair_key) for pair in pairs], ["p1"])
            self.assertEqual(incomplete, ["p2"])

    def test_nested_pair_constructs_exact_complement(self) -> None:
        low = np.full((2, 3, 4, 4), 2, dtype=np.int32)
        high = low + 3
        low_e = np.ones((2, 3), dtype=np.float32)
        high_e = np.full((2, 3), 2.5, dtype=np.float32)
        result = verify_pair(
            make_sample("low", low, low_e),
            make_sample("high", high, high_e),
            PairProvenance.NESTED,
        )
        self.assertTrue(result.accepted)
        np.testing.assert_array_equal(result.target_counts, 3)
        np.testing.assert_allclose(result.target_exposure_s, 1.5)

    def test_synthetic_split_conserves_counts(self) -> None:
        high = np.full((2, 3, 5, 5), 20, dtype=np.int32)
        high_e = np.full((2, 3), 3.0, dtype=np.float32)
        low_e = np.full((2, 3), 1.0, dtype=np.float32)
        low, complement = synthesize_count_split(
            high, high_e, low_e, rng=np.random.default_rng(7)
        )
        np.testing.assert_array_equal(low.astype(np.int64) + complement.astype(np.int64), high)

    def test_nested_negative_difference_rejected(self) -> None:
        low = np.full((1, 2, 3, 3), 4, dtype=np.int32)
        high = np.full((1, 2, 3, 3), 3, dtype=np.int32)
        result = verify_pair(
            make_sample("low", low, np.ones((1, 2))),
            make_sample("high", high, np.full((1, 2), 2.0)),
            PairProvenance.NESTED,
        )
        self.assertFalse(result.accepted)


if __name__ == "__main__":
    unittest.main()
