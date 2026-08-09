from __future__ import annotations

import argparse
import unittest

import torch

from gated_spect.models import ProjectionDenoiser
from gated_spect.training.checkpoint_contract import state_dict_sha256
from scripts.analyze_denoising_runs import _pseudocount_summary
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

    def test_pseudocount_summary_preserves_expanded_patient_metrics(self) -> None:
        patient = {
            "model_deviance": 1.0,
            "naive_deviance": 2.0,
            "absolute_bias": 0.1,
            "per_view_bias": {"mean_absolute_bias": 0.2},
            "zero_input_bin_metrics": {"bias": 0.3},
            "projection_low_signal_mask": {"bias": 0.4},
            "low_count_strata": {"0": {"mae": 0.5}},
            "calibration_table": {"0": {"mean_prediction": 0.6}},
        }
        report = _pseudocount_summary(
            [
                {
                    "pseudocount": 0.5,
                    "seed": 17,
                    "patients": {"p1": patient},
                    "rows": [],
                }
            ],
            resamples=100,
        )
        result = report["results"][0]
        self.assertEqual(result["mean_absolute_per_view_bias"], 0.2)
        self.assertEqual(result["mean_zero_input_bin_bias"], 0.3)
        self.assertEqual(report["selection_rule"]["selection_performed"], False)

    def test_same_seed_fingerprints_decoder_but_not_transferred_backbone(self) -> None:
        torch.manual_seed(17)
        pretrained = ProjectionDenoiser()
        torch.manual_seed(17)
        scratch = ProjectionDenoiser()
        self.assertEqual(
            state_dict_sha256(pretrained.decoder), state_dict_sha256(scratch.decoder)
        )
        with torch.no_grad():
            next(scratch.backbone.parameters()).add_(1.0)
        self.assertNotEqual(
            state_dict_sha256(pretrained.backbone), state_dict_sha256(scratch.backbone)
        )


if __name__ == "__main__":
    unittest.main()
