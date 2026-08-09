from __future__ import annotations

import argparse
import unittest

from scripts.run_matched_denoising import build_matched_command


class ExperimentToolTests(unittest.TestCase):
    def _args(self) -> argparse.Namespace:
        return argparse.Namespace(
            root="root",
            pairs="pairs.jsonl",
            splits="splits.json",
            normalization="normalization.json",
            pretrained_checkpoint="pretrained.pt",
            migration_report="migration.json",
            output_root="runs",
            pseudocount=0.5,
            epochs=2,
            batch_size=1,
            accumulation_steps=2,
            num_workers=0,
            frame_chunk_size=16,
            low_signal_threshold=5.0,
            calibration_ranges="0,1,2,3-5,6-10,11-20,>20",
            freeze_epochs=0,
            comparison_protocol="full_finetune",
        )

    def test_matched_builder_defaults_to_fair_full_finetuning(self) -> None:
        args = self._args()
        pretrained, pre_meta = build_matched_command(
            args, initialization="pretrained", seed=17
        )
        scratch, scratch_meta = build_matched_command(
            args, initialization="scratch", seed=17
        )
        self.assertIn("--freeze-epochs", pretrained)
        self.assertEqual(pretrained[pretrained.index("--freeze-epochs") + 1], "0")
        self.assertEqual(pre_meta["comparison_protocol"], "full_finetune")
        self.assertTrue(pre_meta["same_decoder_initialization_seed"])
        self.assertTrue(scratch_meta["same_decoder_initialization_seed"])
        self.assertFalse(pre_meta["scratch_flag"])
        self.assertTrue(scratch_meta["scratch_flag"])
        self.assertNotEqual(pretrained[-1], "--scratch")
        self.assertEqual(scratch[-1], "--scratch")


if __name__ == "__main__":
    unittest.main()
