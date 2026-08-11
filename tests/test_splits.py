from __future__ import annotations

import unittest
import json
import tempfile
from pathlib import Path

from gated_spect.data.manifest import write_locked_splits
from gated_spect.data.splits import SplitConfig, SplitError, make_dataset1_splits, make_dataset2_splits


def records(dataset_id: str, start: int, stop: int) -> list[dict[str, object]]:
    return [
        {
            "patient_key": f"p{index}",
            "dataset_id": dataset_id,
            "sample_id": f"{dataset_id}-{index}",
            "scanner_id": "scanner_a" if index % 2 else "scanner_b",
            "stress_rest": "stress" if index % 3 else "rest",
            "total_counts": 1000 + 100 * index,
            "training_eligible": True,
        }
        for index in range(start, stop)
    ]


class SplitTests(unittest.TestCase):
    def test_dataset2_split_is_deterministic_and_grouped(self) -> None:
        data = records("dataset2", 0, 80)
        first = make_dataset2_splits(data)
        second = make_dataset2_splits(data)
        self.assertEqual(first, second)
        counts = {name: list(first.values()).count(name) for name in ("train", "validation", "test")}
        self.assertEqual(sum(counts.values()), 80)
        self.assertTrue(all(value >= 10 for value in counts.values()))

    def test_dataset1_excludes_requested_patients(self) -> None:
        data = records("dataset1", 0, 40)
        result = make_dataset1_splits(
            data,
            excluded_patient_keys={"p0", "p1"},
            config=SplitConfig(dataset1_minimum_validation_patients=10),
        )
        self.assertNotIn("p0", result)
        self.assertNotIn("p1", result)
        self.assertEqual(len(result), 38)
        self.assertEqual(list(result.values()).count("validation"), 10)

    def test_small_dataset_is_rejected(self) -> None:
        with self.assertRaises(SplitError):
            make_dataset2_splits(records("dataset2", 0, 20))

    def test_locked_splits_record_exact_input_manifest_bindings(self) -> None:
        dataset1 = records("dataset1", 0, 40)
        dataset2 = records("dataset2", 0, 80)
        pairs = [
            {"accepted": True, "patient_key": row["patient_key"]}
            for row in dataset2
        ]
        with tempfile.TemporaryDirectory() as directory:
            output = write_locked_splits(
                dataset1,
                dataset2,
                pairs,
                output=Path(directory) / "splits.json",
                config=SplitConfig(seed=7),
                dataset1_manifest_sha256="d1",
                dataset2_manifest_sha256="d2",
                pair_manifest_sha256="pairs",
            )
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["dataset1_manifest_sha256"], "d1")
            self.assertEqual(payload["dataset2_manifest_sha256"], "d2")
            self.assertEqual(payload["pair_manifest_sha256"], "pairs")


if __name__ == "__main__":
    unittest.main()
