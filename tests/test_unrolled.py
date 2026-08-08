import math
import unittest

import torch

from gated_spect.models import ProjectionDenoiser, UnrolledOSEM
from gated_spect.physics import Geometry, IdealParallelProjector


class UnrolledTests(unittest.TestCase):
    def test_one_stage_has_positive_volume_and_gradients(self) -> None:
        torch.manual_seed(21)
        geometry = Geometry(
            angles_rad=tuple(i * math.pi / 2.0 for i in range(4)),
            detector_shape=(8, 8),
            volume_shape=(8, 8, 8),
            voxel_spacing_mm=(1.0, 1.0, 1.0),
            detector_spacing_mm=(1.0, 1.0),
        )
        projector = IdealParallelProjector(geometry)
        projection_model = ProjectionDenoiser()
        model = UnrolledOSEM(projector, projection_model, stages=1).eval()
        true_volume = torch.ones(1, 2, 8, 8, 8, dtype=torch.float32)
        low_exposure = torch.ones(1, 2, 4)
        low_counts = projector.forward(true_volume) * low_exposure[..., None, None]
        projection_batch = {
            "counts_input": torch.log1p(low_counts)[:, :, :, None],
            "input_exposure_s": low_exposure,
            "visibility": torch.ones(1, 2, 4, 8, 8, dtype=torch.bool),
            "valid_frames": torch.ones(1, 2, 4, dtype=torch.bool),
            "angles_rad": torch.tensor([[0.0, math.pi / 2, math.pi, 3 * math.pi / 2]]),
            "radii_mm": torch.full((1, 4), 8.0),
            "gate_phase_rad": torch.tensor([[0.0, math.pi]]),
            "split_fraction": torch.zeros(1, 1),
        }
        output = model(
            projection_batch=projection_batch,
            low_counts=low_counts,
            low_exposure_s=low_exposure,
            valid_frames=projection_batch["valid_frames"],
        )
        self.assertEqual(tuple(output.volume.shape), (1, 2, 8, 8, 8))
        self.assertEqual(len(output.stages), 1)
        self.assertTrue(torch.isfinite(output.volume).all())
        self.assertTrue(torch.all(output.volume > 0))
        output.volume.mean().backward()
        self.assertIsNotNone(model.regularizer.output.weight.grad)
        self.assertTrue(torch.isfinite(model.regularizer.output.weight.grad).all())


if __name__ == "__main__":
    unittest.main()
