from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from gated_spect.data.dicom_reader import read_nm_dicom
from gated_spect.data.migration import build_exposure_migration_report
from gated_spect.training.checkpoint_contract import sha256_file
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
            self.assertEqual(report["migration_schema_version"], 2)
            self.assertFalse(report["artifact_hashes_complete"])

    def test_report_hashes_exact_artifact_bytes(self) -> None:
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
            old_norm = root / "old_norm.json"
            new_norm = root / "new_norm.json"
            old_norm.write_text(json.dumps({"stats": stats}), encoding="utf-8")
            new_norm.write_text(json.dumps({"stats": stats}), encoding="utf-8")
            old_d1 = root / "old_d1.jsonl"
            new_d1 = root / "new_d1.jsonl"
            old_d2 = root / "old_d2.jsonl"
            new_d2 = root / "new_d2.jsonl"
            for path in (old_d1, new_d1, old_d2, new_d2):
                path.write_text("{}\n", encoding="utf-8")
            old_splits = root / "old_splits.json"
            new_splits = root / "new_splits.json"
            split_payload = {"dataset1": {}, "dataset2": {}}
            old_splits.write_text(json.dumps(split_payload), encoding="utf-8")
            new_splits.write_text(json.dumps(split_payload), encoding="utf-8")
            checkpoint = root / "legacy.pt"
            checkpoint.write_bytes(b"checkpoint")
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
                old_normalization=old_norm,
                new_normalization=new_norm,
                new_manifests={"dataset1": manifest},
                old_splits=split_payload,
                new_splits=split_payload,
                legacy_checkpoint=checkpoint,
                old_dataset1_manifest=old_d1,
                new_dataset1_manifest=new_d1,
                old_dataset2_manifest=old_d2,
                new_dataset2_manifest=new_d2,
                old_splits_path=old_splits,
                new_splits_path=new_splits,
            )
            self.assertTrue(report["artifact_hashes_complete"])
            self.assertEqual(report["legacy_checkpoint_sha256"], sha256_file(checkpoint))
            # The synthetic normalization deliberately does not reproduce the
            # legacy exposure scaling, so the exact-artifact report must reject
            # transfer even though every hash is present.
            self.assertFalse(report["legacy_backbone_compatibility"])
            missing_sample_report = build_exposure_migration_report(
                (("dataset1", root, manifest),),
                config=ReaderConfig(hash_key=b"test-secret"),
                old_normalization=old_norm,
                new_normalization=new_norm,
                new_manifests={"dataset1": []},
                old_splits=split_payload,
                new_splits=split_payload,
                legacy_checkpoint=checkpoint,
                old_dataset1_manifest=old_d1,
                new_dataset1_manifest=new_d1,
                old_dataset2_manifest=old_d2,
                new_dataset2_manifest=new_d2,
                old_splits_path=old_splits,
                new_splits_path=new_splits,
            )
            self.assertFalse(missing_sample_report["geometry_inputs_unchanged"])
            self.assertFalse(missing_sample_report["legacy_backbone_compatibility"])


if __name__ == "__main__":
    unittest.main()
