import math
import unittest
from unittest.mock import patch

import torch

from gated_spect.losses import (
    EMATeacher,
    GlobalConsistencyHead,
    RepresentationQueue,
    hidden_gate_consistency_loss,
    variance_floor_loss,
)
from gated_spect.models import ProjectionDenoiser
from gated_spect.training import train_two_view_epoch, two_view_pretraining_step
from gated_spect.training.consistency_steps import (
    TwoViewPretrainingLoss,
    _representation_health,
)
from gated_spect.training import runner as training_runner


def _view(*, hidden_gate: bool = False) -> dict[str, torch.Tensor]:
    b, g, v, h, w = 1, 2, 4, 8, 8
    visibility = torch.ones(b, g, v, h, w, dtype=torch.bool)
    if hidden_gate:
        visibility[:, 1] = False
    masked = torch.zeros_like(visibility)
    masked[..., :4, :] = True
    visible = torch.zeros_like(visibility)
    visible[..., 4:, :] = True
    return {
        "counts_input": torch.zeros(b, g, v, 1, h, w),
        "input_exposure_s": torch.ones(b, g, v),
        "visibility": visibility,
        "valid_frames": torch.ones(b, g, v, dtype=torch.bool),
        "relative_exposure_q": torch.ones(b, g, v),
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
        "masked_target": masked,
        "visible_target": visible,
    }


class ConsistencyTests(unittest.TestCase):
    def _run_fake_accumulation(self, batch_count: int, *, failed_first: bool = False):
        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(1.0))
                self.backbone = torch.nn.Linear(1, 1, bias=False)

        class FakeEMA:
            def __init__(self):
                self.updates = 0

            def update(self, _backbone):
                self.updates += 1

        model = TinyModel()
        head = GlobalConsistencyHead(dimension=1, queue_capacity=64)
        ema = FakeEMA()
        optimizer = torch.optim.SGD(
            list(model.parameters()) + list(head.parameters()), lr=0.1
        )
        batches = [
            {
                "step": torch.tensor(float(index)),
                "representation": torch.tensor([[float(2 * index)], [float(2 * index + 1)]]),
            }
            for index in range(batch_count)
        ]

        def fake_step(model, _ema, _head, batch, *, optimizer=None, amp=False, scaler=None):
            del optimizer, amp, scaler
            total = model.weight * (batch["step"] + 1.0)
            zero = total.detach() * 0.0
            return TwoViewPretrainingLoss(
                total=total,
                masked=zero,
                visible=zero,
                deviance=zero,
                latent=zero,
                gate=zero,
                variance=zero,
                queue_values=batch["representation"],
                health={},
            )

        step_results = iter([False, True]) if failed_first else None
        with patch.object(training_runner, "two_view_pretraining_step", fake_step):
            step_patch = patch.object(
                training_runner,
                "optimizer_step_succeeded",
                side_effect=lambda *args, **kwargs: next(step_results)
                if step_results is not None
                else (args[0].step() or True),
            )
            with step_patch:
                metrics = train_two_view_epoch(
                    model,
                    batches,
                    optimizer,
                    ema,
                    head,
                    device="cpu",
                    accumulation_steps=2,
                )
        return metrics, head, ema

    def test_accumulated_queue_inserts_each_window_once(self):
        metrics, head, ema = self._run_fake_accumulation(4)
        self.assertEqual(metrics.health["optimizer_steps"], 2.0)
        self.assertEqual(metrics.health["optimizer_step_skips"], 0.0)
        self.assertEqual(head.queue.num_entries, 8)
        torch.testing.assert_close(
            head.queue.values(), torch.arange(8, dtype=torch.float32).reshape(-1, 1)
        )
        self.assertEqual(ema.updates, 2)

    def test_partial_accumulated_queue_inserts_final_window_once(self):
        metrics, head, ema = self._run_fake_accumulation(5)
        self.assertEqual(metrics.health["optimizer_steps"], 3.0)
        self.assertEqual(head.queue.num_entries, 10)
        torch.testing.assert_close(
            head.queue.values(), torch.arange(10, dtype=torch.float32).reshape(-1, 1)
        )
        self.assertEqual(ema.updates, 3)

    def test_failed_accumulated_amp_window_discards_queue_and_ema(self):
        metrics, head, ema = self._run_fake_accumulation(4, failed_first=True)
        self.assertEqual(metrics.health["optimizer_steps"], 1.0)
        self.assertEqual(metrics.health["optimizer_step_skips"], 1.0)
        self.assertEqual(head.queue.num_entries, 4)
        torch.testing.assert_close(
            head.queue.values(), torch.arange(4, 8, dtype=torch.float32).reshape(-1, 1)
        )
        self.assertEqual(ema.updates, 1)

    def test_covariance_spectrum_rank_detects_duplicate_features(self) -> None:
        queue = RepresentationQueue(dimension=256, capacity=2048)
        values = torch.randn(64, 1).repeat(1, 256)
        health = _representation_health(values, queue)
        self.assertAlmostEqual(float(health["representation_effective_rank"]), 1.0, delta=1e-4)
        self.assertAlmostEqual(float(health["representation_participation_rank"]), 1.0, delta=1e-4)
        self.assertGreater(float(health["mean_absolute_offdiagonal_correlation"]), 0.99)

    def test_queue_default_supports_full_representation(self) -> None:
        self.assertEqual(RepresentationQueue(dimension=256).capacity, 2048)
        independent = torch.randn(512, 256)
        health = _representation_health(independent, RepresentationQueue(dimension=256))
        self.assertGreater(float(health["representation_effective_rank_fraction"]), 0.5)

    def test_correlation_quantile_promotes_half_precision(self) -> None:
        # CUDA autocast can hand diagnostics half-precision projected values;
        # the old implementation passed that dtype to torch.quantile and
        # crashed.  The corrected diagnostic remains finite.
        values = torch.randn(32, 16, dtype=torch.float16)
        health = _representation_health(values, RepresentationQueue(dimension=16))
        self.assertTrue(torch.isfinite(health["p95_absolute_offdiagonal_correlation"]))
        self.assertLessEqual(float(health["max_absolute_offdiagonal_correlation"]), 1.0)

    def test_covariance_loss_distinguishes_correlated_features(self) -> None:
        from gated_spect.losses import covariance_offdiagonal_loss

        independent = covariance_offdiagonal_loss(torch.randn(128, 8), minimum_samples=128)
        duplicate = covariance_offdiagonal_loss(
            torch.randn(128, 1).repeat(1, 8), minimum_samples=128
        )
        self.assertLess(float(independent), float(duplicate))

    def test_covariance_regularizer_is_queue_supported_and_warm_gated(self) -> None:
        from gated_spect.losses import covariance_offdiagonal_loss

        current = torch.randn(2, 8, requires_grad=True)
        history = torch.randn(510, 8, requires_grad=True)
        loss, ready = covariance_offdiagonal_loss(
            current, history, minimum_samples=512, return_ready=True
        )
        self.assertTrue(ready)
        self.assertGreater(float(loss), 0.0)
        loss.backward()
        self.assertIsNotNone(current.grad)
        self.assertIsNone(history.grad)
        cold, cold_ready = covariance_offdiagonal_loss(
            torch.randn(2, 8), torch.randn(10, 8), minimum_samples=512, return_ready=True
        )
        self.assertFalse(cold_ready)
        self.assertEqual(float(cold), 0.0)

    def test_covariance_regularizer_is_scale_invariant(self) -> None:
        from gated_spect.losses import covariance_offdiagonal_loss

        values = torch.randn(512, 8)
        baseline = covariance_offdiagonal_loss(values, minimum_samples=512)
        scaled_small = covariance_offdiagonal_loss(0.01 * values, minimum_samples=512)
        scaled_large = covariance_offdiagonal_loss(100.0 * values, minimum_samples=512)
        self.assertAlmostEqual(float(baseline), float(scaled_small), delta=1.0e-4)
        self.assertAlmostEqual(float(baseline), float(scaled_large), delta=1.0e-4)

    def test_global_head_executes_with_covariance_weight_enabled(self) -> None:
        head = GlobalConsistencyHead(
            dimension=4,
            queue_capacity=16,
            covariance_loss_weight=0.01,
            covariance_min_samples=2,
        )
        representation_a = torch.randn(3, 4, requires_grad=True)
        representation_b = torch.randn(3, 4, requires_grad=True)
        result = head(representation_a, representation_b)
        self.assertTrue(bool(result.covariance_ready))
        self.assertTrue(torch.isfinite(result.total))
        result.total.backward()
        self.assertIsNotNone(representation_a.grad)
        self.assertIsNotNone(representation_b.grad)

    def test_queue_and_variance_warmup(self) -> None:
        queue = RepresentationQueue(dimension=4, capacity=5)
        projected = torch.randn(2, 4)
        self.assertEqual(queue.num_entries, 0)
        self.assertEqual(float(variance_floor_loss(projected, queue)), 0.0)
        queue.enqueue(torch.randn(3, 4))
        self.assertEqual(queue.num_entries, 3)
        self.assertTrue(torch.isfinite(variance_floor_loss(projected, queue, minimum_entries=2)))
        queue.enqueue(torch.randn(4, 4))
        self.assertEqual(queue.num_entries, 5)

    def test_hidden_gate_loss_and_two_view_step(self) -> None:
        torch.manual_seed(41)
        student = torch.randn(1, 2, 4, 6, 2, 2)
        teacher = torch.randn_like(student)
        valid = torch.ones(1, 2, 4, dtype=torch.bool)
        hidden = torch.tensor([[False, True]])
        loss = hidden_gate_consistency_loss(student, teacher, valid, valid, torch.ones_like(valid, dtype=torch.float32), hidden)
        self.assertTrue(torch.isfinite(loss))

        model = ProjectionDenoiser()
        ema = EMATeacher(model.backbone)
        head = GlobalConsistencyHead()
        optimizer = torch.optim.AdamW(
            list(model.parameters()) + list(head.parameters()), lr=1.0e-5
        )
        batch = {
            "student_a": _view(hidden_gate=True),
            "student_b": _view(hidden_gate=False),
            "teacher": _view(hidden_gate=False),
        }
        result = two_view_pretraining_step(model, ema, head, batch, optimizer=optimizer)
        self.assertTrue(torch.isfinite(result.total))
        self.assertTrue(torch.isfinite(result.latent))
        self.assertEqual(head.queue.num_entries, 2)
        metrics = train_two_view_epoch(
            model,
            [batch],
            optimizer,
            ema,
            head,
            device="cpu",
            max_batches=1,
            log_interval=1,
        )
        self.assertIn("representation_effective_rank", metrics.health)
        self.assertIn("rate_mean", metrics.health)
        self.assertIn("gradient_norm_mean", metrics.health)


if __name__ == "__main__":
    unittest.main()
