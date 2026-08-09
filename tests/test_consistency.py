import math
import unittest

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
from gated_spect.training.consistency_steps import _representation_health


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

    def test_covariance_loss_distinguishes_correlated_features(self) -> None:
        from gated_spect.losses import covariance_offdiagonal_loss

        independent = covariance_offdiagonal_loss(torch.randn(128, 8))
        duplicate = covariance_offdiagonal_loss(torch.randn(128, 1).repeat(1, 8))
        self.assertLess(float(independent), float(duplicate))

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
