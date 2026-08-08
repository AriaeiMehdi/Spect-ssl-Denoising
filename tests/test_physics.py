import math
import unittest

import torch

from gated_spect.physics import (
    Geometry,
    IdealParallelProjector,
    mlem_reconstruct,
    osem_reconstruct,
    poisson_nll,
    subset_view_indices,
)


def _projector() -> IdealParallelProjector:
    geometry = Geometry(
        angles_rad=tuple(i * math.pi / 4.0 for i in range(4)),
        detector_shape=(5, 7),
        volume_shape=(5, 7, 7),
        voxel_spacing_mm=(1.0, 1.0, 1.0),
        detector_spacing_mm=(1.0, 1.0),
    )
    return IdealParallelProjector(geometry)


class PhysicsTests(unittest.TestCase):
    def test_exact_transpose_adjointness_float64(self) -> None:
        torch.manual_seed(11)
        projector = _projector()
        x = torch.rand(2, 1, 5, 7, 7, dtype=torch.float64)
        y = torch.rand(2, 1, 4, 5, 7, dtype=torch.float64)
        lhs = (projector.forward(x) * y).sum()
        rhs = (x * projector.adjoint(y)).sum()
        relative_error = (lhs - rhs).abs() / lhs.abs().clamp_min(1.0e-12)
        self.assertLess(float(relative_error), 1.0e-10)

    def test_forward_and_adjoint_shapes_and_finiteness(self) -> None:
        projector = _projector()
        x = torch.ones(1, 2, 5, 7, 7)
        projections = projector.forward(x)
        self.assertEqual(tuple(projections.shape), (1, 2, 4, 5, 7))
        self.assertTrue(torch.isfinite(projections).all())
        self.assertTrue(torch.all(projections >= 0))
        self.assertEqual(tuple(projector.adjoint(projections).shape), tuple(x.shape))

    def test_subset_schedule_is_interleaved_and_complete(self) -> None:
        subsets = subset_view_indices(10, 4)
        self.assertEqual([subset.tolist() for subset in subsets], [[0, 4, 8], [1, 5, 9], [2, 6], [3, 7]])
        self.assertEqual(sorted(i for subset in subsets for i in subset.tolist()), list(range(10)))

    def test_mlem_nll_decreases_and_osem_is_positive(self) -> None:
        torch.manual_seed(12)
        projector = _projector()
        true_volume = torch.full((1, 1, 5, 7, 7), 0.5)
        true_volume[:, :, 2, 3, 4] = 4.0
        exposure = torch.ones(1, 1, 4, dtype=torch.float64) * 2.0
        noiseless = projector.forward(true_volume.double())
        counts = exposure[..., None, None] * noiseless

        initial = mlem_reconstruct(projector, counts, exposure, iterations=1)
        final = mlem_reconstruct(projector, counts, exposure, iterations=10)
        initial_nll = poisson_nll(exposure[..., None, None] * projector.forward(initial), counts)
        final_nll = poisson_nll(exposure[..., None, None] * projector.forward(final), counts)
        self.assertLessEqual(float(final_nll), float(initial_nll) + 1.0e-8)

        reconstruction = osem_reconstruct(projector, counts, exposure, iterations=2, num_subsets=2)
        self.assertTrue(torch.isfinite(reconstruction).all())
        self.assertTrue(torch.all(reconstruction > 0))


if __name__ == "__main__":
    unittest.main()
