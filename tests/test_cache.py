from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from gated_spect.data.cache import read_sample_cache, write_sample_cache
from tests.test_pairing import make_sample


class CacheTests(unittest.TestCase):
    def test_cache_round_trip_and_checksum_reuse(self) -> None:
        counts = np.arange(2 * 3 * 4 * 4, dtype=np.int32).reshape(2, 3, 4, 4)
        exposure = np.ones((2, 3), dtype=np.float32)
        sample = make_sample("sample-a", counts, exposure)
        with tempfile.TemporaryDirectory() as directory:
            path = write_sample_cache(sample, Path(directory))
            self.assertEqual(write_sample_cache(sample, Path(directory)), path)
            loaded = read_sample_cache(path, mmap_counts=False)
            np.testing.assert_array_equal(loaded.counts, sample.counts)
            np.testing.assert_allclose(loaded.frame_exposure_s, sample.frame_exposure_s)
            self.assertEqual(loaded.geometry_hash, sample.geometry_hash)


if __name__ == "__main__":
    unittest.main()
