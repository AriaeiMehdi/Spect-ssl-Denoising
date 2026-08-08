from __future__ import annotations

import os
from pathlib import Path
import unittest

import torch

from gated_spect.data import (
    AcquisitionDataset,
    NormalizationStats,
    PretrainingDataset,
    TwoViewPretrainingDataset,
    collate_acquisitions,
    collate_two_views,
    model_batch,
)
from gated_spect.types import ReaderConfig


DATA_ROOT = Path(r"D:\Dr.EntezarMahdi\3DRecon\datasets")
MANIFEST = Path(r"D:\Dr.EntezarMahdi\3DRecon\artifacts\manifests\dataset1_acquisition.jsonl")
HAS_DATA = DATA_ROOT.joinpath("Pre-training dataset").exists() and MANIFEST.exists()


@unittest.skipUnless(HAS_DATA, "workspace datasets are not mounted")
class TorchDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        key = os.environ.get("SPECT_HASH_KEY", "local-test-only").encode()
        self.dataset = AcquisitionDataset(
            DATA_ROOT / "Pre-training dataset",
            MANIFEST,
            config=ReaderConfig(hash_key=key, institution_id="local"),
            normalization=NormalizationStats(),
        )

    def test_pretraining_split_is_deterministic_and_conserves_counts(self) -> None:
        first_dataset = PretrainingDataset(self.dataset, global_seed=13, epoch=2)
        a = first_dataset[0]
        b = first_dataset[0]
        self.assertTrue(torch.equal(a["input_counts_raw"], b["input_counts_raw"]))
        self.assertTrue(torch.equal(a["target_counts"], b["target_counts"]))
        self.assertTrue(torch.equal(a["input_counts_raw"] + a["target_counts"], a["counts_raw"]))
        different_epoch = PretrainingDataset(self.dataset, global_seed=13, epoch=3)[0]
        self.assertFalse(torch.equal(a["input_counts_raw"], different_epoch["input_counts_raw"]))

    def test_collate_pads_and_model_batch_excludes_targets(self) -> None:
        item = PretrainingDataset(self.dataset, global_seed=3)[0]
        batch = collate_acquisitions([item])
        self.assertEqual(tuple(batch["counts_input"].shape), (1, 16, 32, 1, 64, 64))
        self.assertIn("target_counts", batch)
        selected = model_batch(batch)
        self.assertNotIn("target_counts", selected)
        self.assertNotIn("target_exposure_s", selected)
        self.assertEqual(tuple(selected["counts_input"].shape), (1, 16, 32, 1, 64, 64))

    def test_two_view_dataset_has_unmasked_teacher(self) -> None:
        dataset = TwoViewPretrainingDataset(self.dataset, global_seed=5, epoch=1)
        item = dataset[0]
        self.assertEqual(set(item), {"student_a", "student_b", "teacher"})
        self.assertTrue(item["teacher"]["visibility"].all())
        for name in ("student_a", "student_b", "teacher"):
            view = item[name]
            self.assertTrue(torch.equal(view["input_counts_raw"] + view["target_counts"], view["counts_raw"]))
        batch = collate_two_views([item])
        self.assertEqual(tuple(batch["teacher"]["counts_input"].shape), (1, 16, 32, 1, 64, 64))


if __name__ == "__main__":
    unittest.main()
