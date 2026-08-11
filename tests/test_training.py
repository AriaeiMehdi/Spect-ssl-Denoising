import math
import json
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from gated_spect.versioning import IncompatibleArtifactError

from gated_spect.models import ProjectionDenoiser
from gated_spect.training import (
    capture_rng_state,
    denoising_step,
    evaluate_denoising_patients,
    evaluate_epoch,
    evaluate_scalar_rate_baseline,
    load_checkpoint,
    parameter_summary,
    pretraining_checkpoint_is_eligible,
    pretraining_step,
    save_checkpoint,
    restore_rng_state,
    train_epoch,
    optimizer_step_succeeded,
    validate_resume_state,
    update_post_warmup_patience,
)
from gated_spect.training.checkpoint_contract import sha256_file, validate_checkpoint_metadata
from gated_spect.training.steps import pretraining_objective


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
    def test_scalar_rate_baseline_uses_exposure_conversion(self) -> None:
        batch = _batch()
        batch["counts_raw"] = torch.full_like(batch["target_counts"], 2)
        batch["counts_input"] = batch["counts_raw"].unsqueeze(3)
        batch["input_exposure_s"] = torch.full_like(batch["target_exposure_s"], 2.0)
        batch["target_exposure_s"] = torch.full_like(batch["target_exposure_s"], 4.0)
        batch["target_counts"] = torch.full_like(batch["target_counts"], 4)
        baseline = evaluate_scalar_rate_baseline([batch], device="cpu")
        self.assertAlmostEqual(baseline, 0.0, places=6)

    def test_scalar_rate_baseline_uses_one_rate_from_thinned_visible_bins(self) -> None:
        batch = {
            "input_counts_raw": torch.tensor([[[[[1.0]], [[3.0]]]]]),
            "input_exposure_s": torch.ones(1, 1, 2),
            "visibility": torch.ones(1, 1, 2, 1, 1, dtype=torch.bool),
            "target_exposure_s": torch.ones(1, 1, 2),
            "target_counts": torch.tensor([[[[[2]], [[2]]]]]),
            "target_weight": torch.ones(1, 1, 2),
        }
        baseline = evaluate_scalar_rate_baseline([batch], device="cpu")
        self.assertAlmostEqual(baseline, 0.0, places=6)

    def test_scalar_rate_baseline_matches_student_masks_and_ignores_teacher(self) -> None:
        def view(input_values: list[float], target_values: list[int]) -> dict[str, torch.Tensor]:
            input_counts = torch.tensor(input_values, dtype=torch.float32).reshape(1, 1, 2, 1, 1)
            target_counts = torch.tensor(target_values, dtype=torch.int64).reshape(1, 1, 2, 1, 1)
            selected = torch.zeros_like(target_counts, dtype=torch.bool)
            selected[:, :, 0] = True
            return {
                "input_counts_raw": input_counts,
                "input_exposure_s": torch.ones(1, 1, 2),
                "visibility": torch.ones(1, 1, 2, 1, 1, dtype=torch.bool),
                "target_exposure_s": torch.ones(1, 1, 2),
                "target_counts": target_counts,
                "target_weight": torch.ones(1, 1, 2),
                "masked_target": selected,
                "visible_target": torch.zeros_like(selected),
            }

        # Each student has a different visible-input scalar rate and an exact
        # selected-bin target.  The unmasked teacher is intentionally poor;
        # the matched baseline must not use it.
        batch = {
            "student_a": view([0.0, 4.0], [2, 100]),
            "student_b": view([2.0, 6.0], [4, 100]),
            "teacher": {
                "input_counts_raw": torch.tensor([[[[[10.0]], [[10.0]]]]]),
                "input_exposure_s": torch.ones(1, 1, 2),
                "visibility": torch.ones(1, 1, 2, 1, 1, dtype=torch.bool),
                "target_exposure_s": torch.ones(1, 1, 2),
                "target_counts": torch.tensor([[[[[2]], [[2]]]]]),
                "target_weight": torch.ones(1, 1, 2),
            },
        }
        baseline = evaluate_scalar_rate_baseline([batch], device="cpu")
        self.assertAlmostEqual(baseline, 0.0, places=6)

    def test_pretraining_checkpoint_guardrail_rejects_collapse_and_baseline_regression(self) -> None:
        from gated_spect.training.runner import EpochMetrics

        healthy = EpochMetrics(
            loss=1.0,
            deviance=0.5,
            batches=1,
            health={"representation_std": 0.2},
        )
        self.assertTrue(
            pretraining_checkpoint_is_eligible(healthy, scalar_baseline_deviance=0.6)
        )
        collapsed = EpochMetrics(
            loss=1.0,
            deviance=0.5,
            batches=1,
            health={"representation_std": 0.001},
        )
        self.assertFalse(
            pretraining_checkpoint_is_eligible(collapsed, scalar_baseline_deviance=0.6)
        )
        self.assertFalse(
            pretraining_checkpoint_is_eligible(healthy, scalar_baseline_deviance=0.4)
        )

    def test_checkpoint_guardrail_uses_literal_masked_deviance(self) -> None:
        rate = torch.ones(1, 1, 2, 1, 1)
        target_counts = torch.tensor([[[[[1]], [[9]]]]])
        exposure = torch.ones(1, 1, 2)
        frame_weight = torch.ones(1, 1, 2)
        masked = torch.tensor([[[[[True]], [[False]]]]])
        visible = ~masked
        loss = pretraining_objective(
            rate, target_counts, exposure, frame_weight, masked, visible
        )
        self.assertAlmostEqual(float(loss.masked_deviance), 0.0, places=6)
        self.assertGreater(float(loss.deviance), 0.0)
        from gated_spect.training.runner import EpochMetrics

        metrics = EpochMetrics(
            loss=1.0,
            deviance=0.0,
            batches=1,
            health={"representation_std": 0.2},
            masked_deviance=0.8,
        )
        self.assertFalse(
            pretraining_checkpoint_is_eligible(metrics, scalar_baseline_deviance=0.5)
        )

    def test_early_stop_patience_starts_after_warmup(self) -> None:
        counter = 0
        for epoch in range(1, 11):
            counter = update_post_warmup_patience(
                counter, epoch_number=epoch, warmup_epochs=10, improved=False
            )
        self.assertEqual(counter, 0)
        counter = update_post_warmup_patience(
            counter, epoch_number=11, warmup_epochs=10, improved=False
        )
        self.assertEqual(counter, 1)

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
        self.assertIn("per_view_bias", rows[0])
        self.assertIn("mean_absolute_bias", rows[0]["per_view_bias"])
        self.assertIn("zero_input_bin_metrics", rows[0])
        self.assertIn("low_count_strata", rows[0])
        self.assertIn("projection_low_signal_mask", rows[0])
        self.assertIn("calibration_table", rows[0])

    def test_expanded_patient_metrics_are_not_aggregated_across_batch(self) -> None:
        from types import SimpleNamespace

        class IdentityRateModel(torch.nn.Module):
            baseline_residual = False

            def forward(self, **batch):
                return SimpleNamespace(rate=batch["counts_input"].squeeze(3))

        first = _batch()
        second = _batch()
        first["counts_input"].fill_(1.0)
        first["target_counts"].fill_(1.0)
        second["counts_input"].fill_(2.0)
        second["target_counts"].fill_(4.0)
        batch = {
            key: torch.cat([first[key], second[key]], dim=0)
            for key in first
            if key != "meta"
        }
        batch["meta"] = [
            {"patient_key": "p1", "pair_id": "pair1"},
            {"patient_key": "p2", "pair_id": "pair2"},
        ]
        rows = evaluate_denoising_patients(IdentityRateModel(), [batch], device="cpu")
        self.assertEqual(rows[0]["per_view_bias"]["mean_absolute_bias"], 0.0)
        self.assertAlmostEqual(
            rows[1]["per_view_bias"]["mean_absolute_bias"], 0.5, places=6
        )

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
            save_checkpoint(
                path,
                model,
                optimizer,
                epoch=4,
                metrics={"loss": metrics.loss},
                metadata={"seed": 32},
                extra_state={"resume_marker": 7},
            )
            restored = ProjectionDenoiser()
            restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1.0e-5)
            payload = load_checkpoint(path, restored, restored_optimizer)
            self.assertEqual(payload["epoch"], 4)
            self.assertEqual(payload["metadata"]["seed"], 32)
            self.assertEqual(payload["metadata"]["scientific_semantics_version"], 2)
            self.assertEqual(payload["extra_state"]["resume_marker"], 7)
            for left, right in zip(model.parameters(), restored.parameters()):
                self.assertTrue(torch.equal(left, right))

    def test_rng_state_can_be_captured_and_restored(self) -> None:
        import random
        import numpy as np

        random.seed(11)
        np.random.seed(11)
        torch.manual_seed(11)
        state = capture_rng_state()
        expected = (random.random(), float(np.random.rand()), float(torch.rand(())))
        random.random()
        np.random.rand()
        torch.rand(())
        restore_rng_state(state)
        actual = (random.random(), float(np.random.rand()), float(torch.rand(())))
        self.assertEqual(expected, actual)

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

    def test_checkpoint_does_not_overwrite_without_explicit_permission(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            model = ProjectionDenoiser()
            save_checkpoint(path, model, None, epoch=0, metrics={}, metadata={})
            with self.assertRaises(FileExistsError):
                save_checkpoint(path, model, None, epoch=1, metrics={}, metadata={})
            save_checkpoint(path, model, None, epoch=1, metrics={}, metadata={}, overwrite=True)

    def test_resume_rejects_missing_required_extra_module_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            model = ProjectionDenoiser()
            save_checkpoint(path, model, None, epoch=0, metrics={}, metadata={})
            with self.assertRaises(IncompatibleArtifactError):
                load_checkpoint(
                    path,
                    ProjectionDenoiser(),
                    extra_modules={"ema_teacher": ProjectionDenoiser()},
                )

    def test_resume_state_rejects_missing_optimizer_adjacent_state(self) -> None:
        base = {
            "scheduler": {},
            "scaler": {},
            "rng_state": {"python": (), "numpy": (), "torch": torch.zeros(1, dtype=torch.uint8)},
            "training_state": {
                "best_deviance": float("inf"),
                "best_epoch": None,
                "epochs_without_improvement": 0,
            },
            "loader_generators": {
                "train": torch.zeros(1, dtype=torch.uint8),
                "validation": torch.zeros(1, dtype=torch.uint8),
            },
        }
        validate_resume_state(base)
        for key in ("scheduler", "scaler", "rng_state", "training_state", "loader_generators"):
            incomplete = dict(base)
            incomplete.pop(key)
            with self.assertRaises(IncompatibleArtifactError):
                validate_resume_state(incomplete)
        incomplete_rng = dict(base)
        incomplete_rng["rng_state"] = {"torch": torch.zeros(1, dtype=torch.uint8)}
        with self.assertRaises(IncompatibleArtifactError):
            validate_resume_state(incomplete_rng)
        incomplete_training = dict(base)
        incomplete_training["training_state"] = {"best_deviance": 0.0}
        with self.assertRaises(IncompatibleArtifactError):
            validate_resume_state(incomplete_training)

    def test_resume_rejects_missing_optimizer_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            save_checkpoint(
                path,
                ProjectionDenoiser(),
                optimizer=None,
                epoch=0,
                metrics={},
                metadata={},
            )
            with self.assertRaises(IncompatibleArtifactError):
                load_checkpoint(
                    path,
                    ProjectionDenoiser(),
                    optimizer=torch.optim.AdamW(ProjectionDenoiser().parameters(), lr=1.0e-5),
                )

    def test_interrupted_resume_matches_uninterrupted_training_state(self) -> None:
        class Samples(Dataset[dict[str, torch.Tensor]]):
            def __len__(self) -> int:
                return 4

            def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
                target = torch.full((1, 2, 2, 2), float(index + 1))
                return {
                    "counts_input": torch.zeros(1, 2, 1, 2, 2),
                    "input_exposure_s": torch.ones(1, 2),
                    "visibility": torch.ones(1, 2, 2, 2, dtype=torch.bool),
                    "valid_frames": torch.ones(1, 2, dtype=torch.bool),
                    "angles_rad": torch.zeros(2),
                    "radii_mm": torch.ones(2),
                    "gate_phase_rad": torch.zeros(1),
                    "split_fraction": torch.zeros(1),
                    "scanner_index": torch.zeros((), dtype=torch.long),
                    "collimator_index": torch.zeros((), dtype=torch.long),
                    "stress_rest_index": torch.zeros((), dtype=torch.long),
                    "target_counts": target,
                    "target_exposure_s": torch.ones(1, 2),
                    "target_weight": torch.ones(1, 2),
                }

        class TinyRateModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.log_rate = torch.nn.Parameter(torch.tensor(0.0))

            def forward(self, counts_input: torch.Tensor, **_: torch.Tensor) -> SimpleNamespace:
                b, g, v, _, h, w = counts_input.shape
                return SimpleNamespace(rate=self.log_rate.exp().expand(b, g, v, h, w))

        def loader_pair(train_generator: torch.Generator, validation_generator: torch.Generator):
            data = Samples()
            return (
                DataLoader(data, batch_size=1, shuffle=True, generator=train_generator),
                DataLoader(data, batch_size=2, shuffle=False, generator=validation_generator),
            )

        def scheduler_for(optimizer: torch.optim.Optimizer):
            return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 0.9**step)

        torch.manual_seed(313)
        uninterrupted = TinyRateModel()
        uninterrupted_optimizer = torch.optim.AdamW(uninterrupted.parameters(), lr=1.0e-2)
        uninterrupted_scheduler = scheduler_for(uninterrupted_optimizer)
        uninterrupted_train_generator = torch.Generator(device="cpu").manual_seed(701)
        uninterrupted_validation_generator = torch.Generator(device="cpu").manual_seed(702)
        uninterrupted_train_loader, uninterrupted_validation_loader = loader_pair(
            uninterrupted_train_generator, uninterrupted_validation_generator
        )
        train_epoch(
            uninterrupted,
            uninterrupted_train_loader,
            uninterrupted_optimizer,
            task="denoising",
            device="cpu",
        )
        uninterrupted_scheduler.step()
        list(uninterrupted_validation_loader)
        uninterrupted_train_generator_state = uninterrupted_train_generator.get_state()
        uninterrupted_validation_generator_state = uninterrupted_validation_generator.get_state()
        train_epoch(
            uninterrupted,
            uninterrupted_train_loader,
            uninterrupted_optimizer,
            task="denoising",
            device="cpu",
        )
        uninterrupted_scheduler.step()

        torch.manual_seed(313)
        interrupted = TinyRateModel()
        interrupted_optimizer = torch.optim.AdamW(interrupted.parameters(), lr=1.0e-2)
        interrupted_scheduler = scheduler_for(interrupted_optimizer)
        interrupted_train_generator = torch.Generator(device="cpu").manual_seed(701)
        interrupted_validation_generator = torch.Generator(device="cpu").manual_seed(702)
        interrupted_train_loader, interrupted_validation_loader = loader_pair(
            interrupted_train_generator, interrupted_validation_generator
        )
        train_epoch(
            interrupted,
            interrupted_train_loader,
            interrupted_optimizer,
            task="denoising",
            device="cpu",
        )
        interrupted_scheduler.step()
        list(interrupted_validation_loader)
        extra_state = {
            "scheduler": interrupted_scheduler.state_dict(),
            "scaler": {},
            "rng_state": capture_rng_state(),
            "training_state": {
                "best_deviance": 1.0,
                "best_epoch": 0,
                "epochs_without_improvement": 0,
            },
            "loader_generators": {
                "train": interrupted_train_generator.get_state(),
                "validation": interrupted_validation_generator.get_state(),
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "latest.pt"
            save_checkpoint(
                path,
                interrupted,
                interrupted_optimizer,
                epoch=0,
                metrics={"validation_masked_deviance": 1.0},
                metadata={"training_contract": {"epochs": 2}},
                extra_state=extra_state,
            )
            resumed = TinyRateModel()
            resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=1.0e-2)
            resumed_scheduler = scheduler_for(resumed_optimizer)
            loaded = load_checkpoint(path, resumed, resumed_optimizer)
            validate_resume_state(loaded["extra_state"])
            resumed_scheduler.load_state_dict(loaded["extra_state"]["scheduler"])
            restore_rng_state(loaded["extra_state"]["rng_state"])
            resumed_train_generator = torch.Generator(device="cpu")
            resumed_validation_generator = torch.Generator(device="cpu")
            resumed_train_generator.set_state(
                loaded["extra_state"]["loader_generators"]["train"]
            )
            resumed_validation_generator.set_state(
                loaded["extra_state"]["loader_generators"]["validation"]
            )
            resumed_train_loader, resumed_validation_loader = loader_pair(
                resumed_train_generator, resumed_validation_generator
            )
            list(resumed_validation_loader)
            train_epoch(
                resumed,
                resumed_train_loader,
                resumed_optimizer,
                task="denoising",
                device="cpu",
            )
            resumed_scheduler.step()

        self.assertEqual(uninterrupted_train_generator_state.tolist(), extra_state["loader_generators"]["train"].tolist())
        self.assertEqual(
            uninterrupted_validation_generator_state.tolist(),
            extra_state["loader_generators"]["validation"].tolist(),
        )
        self.assertTrue(torch.equal(uninterrupted.log_rate.detach(), resumed.log_rate.detach()))
        self.assertEqual(uninterrupted_optimizer.state_dict(), resumed_optimizer.state_dict())
        self.assertEqual(uninterrupted_scheduler.state_dict(), resumed_scheduler.state_dict())

    def test_legacy_backbone_transfer_requires_passing_migration_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "migration.json"
            report.write_text('{"legacy_backbone_compatibility": true}', encoding="utf-8")
            with self.assertRaises(IncompatibleArtifactError):
                validate_checkpoint_metadata(
                    {},
                    allow_legacy_backbone_transfer=True,
                    migration_report=report,
                    legacy_checkpoint_sha256="unused",
                )

    def test_legacy_transfer_binds_exact_checkpoint_and_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "legacy.pt"
            checkpoint.write_bytes(b"legacy-checkpoint")
            artifact_names = (
                "old_dataset1.jsonl",
                "new_dataset1.jsonl",
                "old_dataset2.jsonl",
                "new_dataset2.jsonl",
                "old_splits.json",
                "new_splits.json",
                "old_norm.json",
                "new_norm.json",
            )
            artifacts = {}
            for name in artifact_names:
                path = root / name
                path.write_text(name, encoding="utf-8")
                artifacts[name] = sha256_file(path)
            report_payload = {
                "migration_schema_version": 2,
                "artifact_hashes_complete": True,
                "legacy_backbone_compatibility": True,
                "count_normalization_unchanged": True,
                "radius_normalization_unchanged": True,
                "element_counts_unchanged": True,
                "geometry_inputs_unchanged": True,
                "exact_sample_coverage": True,
                "finite_exposure_deltas": True,
                "migration_error_count": 0,
                "patient_split_migration": {"unchanged": True},
                "mean_abs_delta_z": 0.0,
                "p99_abs_delta_z": 0.0,
                "compatibility_thresholds": {
                    "mean_abs_delta_z_max_exclusive": 1.0e-3,
                    "p99_abs_delta_z_max_exclusive": 1.0e-2,
                },
                "legacy_checkpoint_sha256": sha256_file(checkpoint),
                "old_dataset1_manifest_sha256": artifacts["old_dataset1.jsonl"],
                "new_dataset1_manifest_sha256": artifacts["new_dataset1.jsonl"],
                "old_dataset2_manifest_sha256": artifacts["old_dataset2.jsonl"],
                "new_dataset2_manifest_sha256": artifacts["new_dataset2.jsonl"],
                "old_splits_sha256": artifacts["old_splits.json"],
                "new_splits_sha256": artifacts["new_splits.json"],
                "old_normalization_sha256": artifacts["old_norm.json"],
                "new_normalization_sha256": artifacts["new_norm.json"],
            }
            report = root / "migration.json"
            report.write_text(json.dumps(report_payload), encoding="utf-8")
            kwargs = {
                "normalization_sha256": artifacts["new_norm.json"],
                "manifest_sha256": artifacts["new_dataset2.jsonl"],
                "splits_sha256": artifacts["new_splits.json"],
                "allow_legacy_backbone_transfer": True,
                "migration_report": report,
                "legacy_checkpoint_sha256": sha256_file(checkpoint),
            }
            validate_checkpoint_metadata({}, **kwargs)
            wrong_thresholds = dict(report_payload)
            wrong_thresholds["compatibility_thresholds"] = {
                "mean_abs_delta_z_max_exclusive": 0.1,
                "p99_abs_delta_z_max_exclusive": 0.01,
            }
            report.write_text(json.dumps(wrong_thresholds), encoding="utf-8")
            with self.assertRaises(IncompatibleArtifactError):
                validate_checkpoint_metadata({}, **kwargs)
            report.write_text(json.dumps(report_payload), encoding="utf-8")
            large_delta = dict(report_payload)
            large_delta["mean_abs_delta_z"] = 1.0
            report.write_text(json.dumps(large_delta), encoding="utf-8")
            with self.assertRaises(IncompatibleArtifactError):
                validate_checkpoint_metadata({}, **kwargs)
            report.write_text(json.dumps(report_payload), encoding="utf-8")
            incomplete = dict(report_payload)
            incomplete.pop("radius_normalization_unchanged")
            report.write_text(json.dumps(incomplete), encoding="utf-8")
            with self.assertRaises(IncompatibleArtifactError):
                validate_checkpoint_metadata({}, **kwargs)
            report.write_text(json.dumps(report_payload), encoding="utf-8")
            for field, value in (
                ("legacy_checkpoint_sha256", "wrong-checkpoint"),
                ("new_normalization_sha256", "wrong-normalization"),
                ("new_splits_sha256", "wrong-splits"),
                ("new_dataset2_manifest_sha256", "wrong-manifest"),
            ):
                mutated = dict(report_payload)
                mutated[field] = value
                report.write_text(json.dumps(mutated), encoding="utf-8")
                with self.assertRaises(IncompatibleArtifactError):
                    validate_checkpoint_metadata({}, **kwargs)
            pair_manifest = root / "new_pairs.jsonl"
            pair_manifest.write_text("pair-manifest", encoding="utf-8")
            report_payload["new_dataset2_pairs_sha256"] = sha256_file(pair_manifest)
            report.write_text(json.dumps(report_payload), encoding="utf-8")
            pair_kwargs = dict(kwargs)
            pair_kwargs.update(
                {
                    "manifest_sha256": sha256_file(pair_manifest),
                    "pairs_sha256": sha256_file(pair_manifest),
                }
            )
            validate_checkpoint_metadata({}, **pair_kwargs)
            report_payload["new_dataset2_pairs_sha256"] = "wrong-pairs"
            report.write_text(json.dumps(report_payload), encoding="utf-8")
            with self.assertRaises(IncompatibleArtifactError):
                validate_checkpoint_metadata({}, **pair_kwargs)
            report.write_text(json.dumps(report_payload), encoding="utf-8")

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
