from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from gated_spect.versioning import (
    ArtifactCollisionError,
    IncompatibleArtifactError,
    artifact_version_metadata,
    sha256_file,
    validate_manifest_source_bindings,
    validate_normalization_binding,
    validate_pair_source_bindings,
    validate_resume_contract,
    write_text_create_only,
)


class ArtifactBindingTests(unittest.TestCase):
    def test_create_only_writer_reuses_identical_bytes_and_rejects_collision(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.json"
            self.assertEqual(write_text_create_only(path, "one\n"), path)
            self.assertEqual(write_text_create_only(path, "one\n"), path)
            with self.assertRaises(ArtifactCollisionError):
                write_text_create_only(path, "two\n")
            self.assertEqual(path.read_text(encoding="utf-8"), "one\n")

    def _write_fixture(self, root: Path) -> tuple[Path, Path, Path]:
        manifest = root / "manifest.jsonl"
        splits = root / "splits.json"
        normalization = root / "normalization.json"
        manifest.write_text('{"row": 1}\n', encoding="utf-8")
        splits.write_text('{"dataset1": {"p": "pretrain"}}\n', encoding="utf-8")
        normalization.write_text(
            json.dumps(
                {
                    **artifact_version_metadata(),
                    "manifest_sha256": sha256_file(manifest),
                    "splits_sha256": sha256_file(splits),
                    "stats": {
                        "count_log_mean": 0.0,
                        "count_log_std": 1.0,
                        "log_exposure_mean": 0.0,
                        "log_exposure_std": 1.0,
                        "radius_mean_mm": 0.0,
                        "radius_std_mm": 1.0,
                    },
                }
            ),
            encoding="utf-8",
        )
        return normalization, manifest, splits

    def test_exact_manifest_and_split_bindings_are_required(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            normalization, manifest, splits = self._write_fixture(Path(directory))
            binding = validate_normalization_binding(
                normalization, manifest_path=manifest, splits_path=splits
            )
            self.assertEqual(binding["manifest_sha256"], sha256_file(manifest))
            self.assertEqual(binding["splits_sha256"], sha256_file(splits))
            manifest.write_text('{"row": 2}\n', encoding="utf-8")
            with self.assertRaises(IncompatibleArtifactError):
                validate_normalization_binding(
                    normalization, manifest_path=manifest, splits_path=splits
                )

    def test_missing_bindings_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            normalization, manifest, splits = self._write_fixture(Path(directory))
            payload = json.loads(normalization.read_text(encoding="utf-8"))
            payload.pop("manifest_sha256")
            normalization.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(IncompatibleArtifactError):
                validate_normalization_binding(
                    normalization, manifest_path=manifest, splits_path=splits
                )

    def test_source_replacement_fails_manifest_binding(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "sample.dcm"
            source.write_bytes(b"original")
            rows = [
                {
                    "relative_path": source.name,
                    "training_eligible": True,
                    "source_sha256": sha256_file(source),
                }
            ]
            self.assertEqual(validate_manifest_source_bindings(root, rows), 1)
            source.write_bytes(b"replacement")
            with self.assertRaises(IncompatibleArtifactError):
                validate_manifest_source_bindings(root, rows)

    def test_pair_source_replacement_fails_pair_binding(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            low = root / "low.dcm"
            high = root / "high.dcm"
            low.write_bytes(b"low")
            high.write_bytes(b"high")
            rows = [
                {
                    "accepted": True,
                    "low_relative_path": low.name,
                    "high_relative_path": high.name,
                    "low_source_sha256": sha256_file(low),
                    "high_source_sha256": sha256_file(high),
                }
            ]
            self.assertEqual(validate_pair_source_bindings(root, rows), 2)
            high.write_bytes(b"changed")
            with self.assertRaises(IncompatibleArtifactError):
                validate_pair_source_bindings(root, rows)

    def test_resume_contract_rejects_scientific_setting_changes(self) -> None:
        runtime = {
            "seed": 1,
            "epochs": 250,
            "accumulation_steps": 8,
            "learning_rate": 2.0e-4,
        }
        validate_resume_contract(dict(runtime), runtime)
        changed = dict(runtime)
        changed["seed"] = 2
        with self.assertRaises(IncompatibleArtifactError):
            validate_resume_contract(runtime, changed)
        changed = dict(runtime)
        changed["epochs"] = 100
        with self.assertRaises(IncompatibleArtifactError):
            validate_resume_contract(runtime, changed)


if __name__ == "__main__":
    unittest.main()
