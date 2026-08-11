from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from gated_spect.data.cache import StaleCacheError, read_sample_cache, write_sample_cache
from tests.test_pairing import make_sample


class CacheTests(unittest.TestCase):
    @staticmethod
    def _provenance() -> dict[str, str]:
        return {
            "source_relative_path": "sample.dcm",
            "source_sha256": "a" * 64,
            "manifest_sha256": "b" * 64,
            "manifest_row_sha256": "c" * 64,
        }

    def test_cache_round_trip_and_checksum_reuse(self) -> None:
        counts = np.arange(2 * 3 * 4 * 4, dtype=np.int32).reshape(2, 3, 4, 4)
        exposure = np.ones((2, 3), dtype=np.float32)
        sample = make_sample("sample-a", counts, exposure)
        with tempfile.TemporaryDirectory() as directory:
            path = write_sample_cache(sample, Path(directory), **self._provenance())
            self.assertEqual(write_sample_cache(sample, Path(directory), **self._provenance()), path)
            loaded = read_sample_cache(path, mmap_counts=False)
            np.testing.assert_array_equal(loaded.counts, sample.counts)
            np.testing.assert_allclose(loaded.frame_exposure_s, sample.frame_exposure_s)
            self.assertEqual(loaded.geometry_hash, sample.geometry_hash)

    def test_schema3_cache_requires_provenance(self) -> None:
        sample = make_sample(
            "sample-missing-provenance",
            np.ones((1, 2, 4, 4), dtype=np.int32),
            np.ones((1, 2), dtype=np.float32),
        )
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                write_sample_cache(sample, Path(directory))

    def test_schema3_cache_read_rejects_invalid_provenance(self) -> None:
        sample = make_sample(
            "sample-invalid-provenance",
            np.ones((1, 2, 4, 4), dtype=np.int32),
            np.ones((1, 2), dtype=np.float32),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = write_sample_cache(sample, Path(directory), **self._provenance())
            metadata_path = path / "metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["source_relative_path"] = r"C:\absolute\source.dcm"
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaises(StaleCacheError):
                read_sample_cache(path, mmap_counts=False)

    def test_schema3_cache_read_requires_complete_checksum_bundle(self) -> None:
        sample = make_sample(
            "sample-incomplete-checksums",
            np.ones((1, 2, 4, 4), dtype=np.int32),
            np.ones((1, 2), dtype=np.float32),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = write_sample_cache(sample, Path(directory), **self._provenance())
            checksums_path = path / "checksums.json"
            checksums = json.loads(checksums_path.read_text(encoding="utf-8"))
            checksums.pop("counts.npy")
            checksums_path.write_text(json.dumps(checksums), encoding="utf-8")
            with self.assertRaises(StaleCacheError):
                read_sample_cache(path, mmap_counts=False)

    def test_stale_cache_is_rejected_without_overwrite(self) -> None:
        counts = np.ones((1, 2, 4, 4), dtype=np.int32)
        sample = make_sample("sample-stale", counts, np.ones((1, 2), dtype=np.float32))
        with tempfile.TemporaryDirectory() as directory:
            path = write_sample_cache(sample, Path(directory), **self._provenance())
            metadata_path = path / "metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata.pop("cache_schema_version")
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaises(StaleCacheError):
                read_sample_cache(path, mmap_counts=False)

    def test_schema2_cache_is_not_reinterpreted_as_schema3(self) -> None:
        sample = make_sample(
            "sample-schema2",
            np.ones((1, 2, 4, 4), dtype=np.int32),
            np.ones((1, 2), dtype=np.float32),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = write_sample_cache(sample, Path(directory), **self._provenance())
            metadata_path = path / "metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["cache_schema_version"] = 2
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaises(StaleCacheError):
                read_sample_cache(path, mmap_counts=False)

    def test_existing_cache_provenance_is_compared_to_candidate(self) -> None:
        sample = make_sample(
            "sample-provenance",
            np.ones((1, 2, 4, 4), dtype=np.int32),
            np.ones((1, 2), dtype=np.float32),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_sample_cache(
                sample,
                root,
                source_relative_path="a.dcm",
                source_sha256="a" * 64,
                manifest_sha256="b" * 64,
                manifest_row_sha256="c" * 64,
            )
            with self.assertRaises(StaleCacheError):
                write_sample_cache(
                    sample,
                    root,
                    source_relative_path="a.dcm",
                    source_sha256="d" * 64,
                    manifest_sha256="b" * 64,
                    manifest_row_sha256="c" * 64,
                )


if __name__ == "__main__":
    unittest.main()
