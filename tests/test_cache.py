from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from gated_spect.data.cache import StaleCacheError, read_sample_cache, write_sample_cache
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

    def test_stale_cache_is_rejected_without_overwrite(self) -> None:
        counts = np.ones((1, 2, 4, 4), dtype=np.int32)
        sample = make_sample("sample-stale", counts, np.ones((1, 2), dtype=np.float32))
        with tempfile.TemporaryDirectory() as directory:
            path = write_sample_cache(sample, Path(directory))
            metadata_path = path / "metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata.pop("cache_schema_version")
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaises(StaleCacheError):
                read_sample_cache(path, mmap_counts=False)


if __name__ == "__main__":
    unittest.main()
