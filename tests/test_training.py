import math
import tempfile
import unittest
from pathlib import Path

import torch
from gated_spect.versioning import IncompatibleArtifactError

from gated_spect.models import ProjectionDenoiser
from gated_spect.training import (
    denoising_step,
    evaluate_denoising_patients,
    evaluate_epoch,
    load_checkpoint,
    parameter_summary,
    pretraining_step,
    save_checkpoint,
    train_epoch,
    optimizer_step_succeeded,
)
from gated_spect.training.checkpoint_contract import validate_checkpoint_metadata


def _batch() -> dict[str, torch.Tensor]:
    b, g, v, h, w = 1, 2, 4, 8, 8
    mask = torch.zeros(b, g, v, h, w, dtype=torch.bool)
    mask[..., :4, :] = True
    visible = torch.zeros_like(mask)
    visible[..., 4:, :] = True
    return {
        "counts_input": torch.zeros(b, g, v, 1, h, w),
        "input_exposure_s": torch.ones(b, g, v),
        "visibility": torch.ones(b, g, v, h, w, dtype=torch.bool),
        "valid_frames": torch.ones(b, g, v, dtype=torch.bool),
        "angles_rad": torch.tensor([[0.0, math.pi / 2, math.pi, 3 * math.pi / 2]]),
        "radii_mm": torch.full((b, v), 8.0),
        "gate_phase_rad": torch.tensor([[0.0, math.pi]]),
        "split_fraction": torch.full((b, 1), 0.5),
        "scanner_index": torch.zeros(b, dtype=torch.long),
        "collimator_index": torch.zeros(b, dtype=torch.long),
        "stress_rest_index": torch.zeros(b, dtype=torch.long),
        "target_counts": torch.ones(b, g, v, h, w, dtype=torch.int64),
        "target_exposure_s": torch.ones(b, g, v),
        "target_weight": torch.ones(b, g, v),
        "masked_target": mask,
        "visible_target": visible,
    }


class TrainingStepTests(unittest.TestCase):
    def test_patient_evaluation_retains_identity_and_naive_baseline(self) -> None:
        from types import SimpleNamespace

        class IdentityRateModel(torch.nn.Module):
            baseline_residual = False

            def forward(self, **batch):
                return SimpleNamespace(rate=batch["counts_input"].squeeze(3))

        batch = {
            "counts_input": torch.tensor([[[[[[1.0, 2.0]]]]]]),
            "input_exposure_s": torch.tensor([[[2.0]]]),
            "target_counts": torch.tensor([[[[[3, 6]]]]]),
            "target_exposure_s": torch.tensor([[[3.0]]]),
            "target_weight": torch.ones(1, 1, 1),
            "counts_raw": torch.tensor([[[[[2, 4]]]]]),
            "meta": [{"patient_key": "p1", "pair_id": "pair1"}],
            "visibility": torch.ones(1, 1, 1, 1, 2, dtype=torch.bool),
            "valid_frames": torch.ones(1, 1, 1, dtype=torch.bool),
            "angles_rad": torch.zeros(1, 1),
            "radii_mm": torch.ones(1, 1),
            "gate_phase_rad": torch.zeros(1, 1),
            "split_fraction": torch.zeros(1, 1),
            "scanner_index": torch.zeros(1, dtype=torch.long),
            "collimator_index": torch.zeros(1, dtype=torch.long),
            "stress_rest_index": torch.zeros(1, dtype=torch.long),
        }
        rows = evaluate_denoising_patients(IdentityRateModel(), [batch], device="cpu")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["patient_key"], "p1")
        self.assertAlmostEqual(rows[0]["model_deviance"], 0.0, places=6)
        self.assertAlmostEqual(rows[0]["naive_deviance"], 0.0, places=6)

    def test_amp_scale_drop_marks_optimizer_step_skipped(self) -> None:
        class FakeScaler:
            def __init__(self) -> None:
                self.scale_value = 8.0

            def get_scale(self) -> float:
                return self.scale_value

            def step(self, optimizer: torch.optim.Optimizer) -> None:
                self.scale_value = 4.0

            def update(self) -> None:
                pass

        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.SGD([parameter], lr=1.0)
        parameter.grad = torch.tensor(1.0)
        success = optimizer_step_succeeded(
            optimizer, scaler=FakeScaler(), autocast_enabled=True
        )
        self.assertFalse(success)
        self.assertEqual(float(parameter), 1.0)

    def test_one_batch_pretraining_and_denoising_steps_are_finite(self) -> None:
        torch.manual_seed(31)
        model = ProjectionDenoiser()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-5)
        batch = _batch()
        pretraining_loss = pretraining_step(model, batch, optimizer=optimizer)
        self.assertTrue(torch.isfinite(pretraining_loss.total))
        self.assertTrue(torch.isfinite(pretraining_loss.deviance))
        denoising_loss = denoising_step(model, batch, optimizer=optimizer)
        self.assertTrue(torch.isfinite(denoising_loss.total))
        self.assertTrue(torch.isfinite(denoising_loss.deviance))

    def test_epoch_runner_and_checkpoint_round_trip(self) -> None:
        torch.manual_seed(32)
        model = ProjectionDenoiser()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-5)
        batch = _batch()
        metrics = train_epoch(
            model,
            [batch],
            optimizer,
            task="denoising",
            device="cpu",
            max_batches=1,
        )
        self.assertEqual(metrics.batches, 1)
        validation = evaluate_epoch(model, [batch], task="denoising", device="cpu", max_batches=1)
        self.assertEqual(validation.batches, 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            save_checkpoint(path, model, optimizer, epoch=4, metrics={"loss": metrics.loss}, metadata={"seed": 32})
            restored = ProjectionDenoiser()
            restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1.0e-5)
            payload = load_checkpoint(path, restored, restored_optimizer)
            self.assertEqual(payload["epoch"], 4)
            self.assertEqual(payload["metadata"]["seed"], 32)
            self.assertEqual(payload["metadata"]["scientific_semantics_version"], 2)
            for left, right in zip(model.parameters(), restored.parameters()):
                self.assertTrue(torch.equal(left, right))

    def test_legacy_checkpoint_is_rejected_by_versioned_loader(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.pt"
            model = ProjectionDenoiser()
            torch.save(
                {
                    "format_version": 1,
                    "epoch": 0,
                    "model": model.state_dict(),
                    "optimizer": None,
                    "metrics": {},
                    "metadata": {},
                    "extra_state": {},
                },
                path,
            )
            with self.assertRaises(IncompatibleArtifactError):
                load_checkpoint(path, ProjectionDenoiser())

    def test_legacy_backbone_transfer_requires_passing_migration_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "migration.json"
            report.write_text('{"legacy_backbone_compatibility": true}', encoding="utf-8")
            validate_checkpoint_metadata(
                {}, allow_legacy_backbone_transfer=True, migration_report=report
            )
            report.write_text('{"legacy_backbone_compatibility": false}', encoding="utf-8")
            with self.assertRaises(IncompatibleArtifactError):
                validate_checkpoint_metadata(
                    {}, allow_legacy_backbone_transfer=True, migration_report=report
                )

    def test_gradient_accumulation_steps_optimizer_once_for_two_batches(self) -> None:
        torch.manual_seed(33)
        model = ProjectionDenoiser()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-5)
        batch = _batch()
        train_epoch(
            model,
            [batch, batch],
            optimizer,
            task="denoising",
            device="cpu",
            accumulation_steps=2,
        )
        first_parameter = next(model.parameters())
        self.assertEqual(int(optimizer.state[first_parameter]["step"].item()), 1)

    def test_progress_callback_and_parameter_summary(self) -> None:
        torch.manual_seed(34)
        model = ProjectionDenoiser()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-5)
        events: list[dict[str, object]] = []
        metrics = train_epoch(
            model,
            [_batch(), _batch()],
            optimizer,
            task="pretraining",
            device="cpu",
            accumulation_steps=2,
            log_interval=1,
            epoch=3,
            progress_callback=events.append,
        )
        self.assertEqual(metrics.batches, 2)
        self.assertEqual(events[0]["event"], "batch")
        self.assertEqual(events[0]["phase"], "train")
        self.assertIn("rate_mean", events[0])
        self.assertIn("masked_fraction", events[0])
        self.assertIn("gradient_norm", events[0])
        self.assertEqual(events[-1]["event"], "phase_end")
        counts = parameter_summary(model)
        self.assertEqual(counts["total"], counts["trainable"])
        self.assertGreater(counts["total"], 0)


if __name__ == "__main__":
    unittest.main()
