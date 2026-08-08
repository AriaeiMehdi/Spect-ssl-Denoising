from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from gated_spect.data.inventory import inventory_headers, write_jsonl
from gated_spect.types import ReaderConfig
from tests.test_dicom_reader import write_fixture


class InventoryTests(unittest.TestCase):
    def test_header_inventory_is_deidentified_and_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "dataset"
            root.mkdir()
            write_fixture(root / "sample.dcm")
            rows = inventory_headers(
                [root],
                dataset_id="dataset1",
                config=ReaderConfig(hash_key=b"test-secret", institution_id="site-a"),
            )
            self.assertEqual(len(rows), 1)
            self.assertNotIn("PatientID", rows[0])
            self.assertIn("patient_key", rows[0])
            self.assertNotIn(str(root), json.dumps(rows[0]))
            output = Path(directory) / "manifest.jsonl"
            write_jsonl(rows, output)
            self.assertEqual(len(output.read_text(encoding="utf-8").splitlines()), 1)


if __name__ == "__main__":
    unittest.main()
