from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from gated_spect.data.dicom_reader import read_nm_dicom
from gated_spect.data.migration import build_exposure_migration_report
from gated_spect.types import ReaderConfig
from tests.test_dicom_reader import write_fixture


class MigrationTests(unittest.TestCase):
    def test_report_reproduces_legacy_duration_division_and_rejects_transfer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.dcm"
            write_fixture(source, actual_frame_duration_ms=10_000, views=32)
            sample = read_nm_dicom(source, ReaderConfig(hash_key=b"test-secret"))
            stats = {
                "count_log_mean": 0.0,
                "count_log_std": 1.0,
                "log_exposure_mean": 0.0,
                "log_exposure_std": 1.0,
                "radius_mean_mm": 0.0,
                "radius_std_mm": 1.0,
            }
            old_stats = root / "old.json"
            new_stats = root / "new.json"
            old_stats.write_text(json.dumps({"stats": stats}), encoding="utf-8")
            new_stats.write_text(json.dumps({"stats": stats}), encoding="utf-8")
            manifest = [
                {
                    "parse_status": "parsed",
                    "relative_path": source.name,
                    "sample_id": sample.sample_id,
                    "geometry_hash": sample.geometry_hash,
                }
            ]
            report = build_exposure_migration_report(
                (("dataset1", root, manifest),),
                config=ReaderConfig(hash_key=b"test-secret"),
                old_normalization=old_stats,
                new_normalization=new_stats,
            )
            self.assertEqual(report["rows"][0]["old_median_dwell_s"], 10.0 / 32.0)
            self.assertEqual(report["rows"][0]["new_median_dwell_s"], 10.0)
            self.assertEqual(report["rows"][0]["actual_frame_duration_ms"], 10_000.0)
            self.assertFalse(report["legacy_backbone_compatibility"])


if __name__ == "__main__":
    unittest.main()
