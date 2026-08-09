import unittest

import torch

from gated_spect.models import (
    ProjectionBackbone,
    ProjectionDenoiser,
    pool_global_bottleneck,
    standardized_log_exposure,
)


def _batch(batch_size: int = 1, gates: int = 2, views: int = 3) -> dict[str, torch.Tensor]:
    h = w = 64
    return {
        "counts_input": torch.zeros(batch_size, gates, views, 1, h, w),
        "input_exposure_s": torch.ones(batch_size, gates, views),
        "visibility": torch.ones(batch_size, gates, views, h, w, dtype=torch.bool),
        "valid_frames": torch.ones(batch_size, gates, views, dtype=torch.bool),
        "angles_rad": torch.linspace(0.0, 1.0, views).repeat(batch_size, 1),
        "radii_mm": torch.full((batch_size, views), 220.0),
        "gate_phase_rad": torch.linspace(0.0, 1.0, gates).repeat(batch_size, 1),
        "split_fraction": torch.zeros(batch_size, 1),
    }


class ModelContractTests(unittest.TestCase):
    def test_exposure_conditioning_is_clipped_and_shared_transform_is_finite(self) -> None:
        exposure = torch.tensor([1.0e-20, 1.0, 1.0e20])
        transformed = standardized_log_exposure(exposure, 0.0, 1.0)
        self.assertTrue(torch.all(transformed <= 8.0))
        self.assertTrue(torch.all(transformed >= -8.0))
        self.assertTrue(torch.isfinite(transformed).all())
        self.assertGreaterEqual(float(standardized_log_exposure(exposure, 0.0, 1.0).numel()), 1)

    def test_global_pool_is_mean_over_valid_spatial_tokens(self) -> None:
        values = torch.arange(1.0, 1.0 + 2 * 1 * 2 * 1 * 2 * 2).reshape(2, 1, 2, 1, 2, 2)
        valid = torch.tensor([[[True, False]], [[True, True]]])
        pooled = pool_global_bottleneck(values, valid)
        expected = torch.tensor([[2.5], [12.5]])
        torch.testing.assert_close(pooled, expected)

    def test_shapes_and_positive_rates(self) -> None:
        torch.manual_seed(3)
        model = ProjectionDenoiser().eval()
        with torch.no_grad():
            output = model(**_batch(batch_size=2, gates=3, views=4))
        self.assertEqual(tuple(output.rate.shape), (2, 3, 4, 64, 64))
        self.assertEqual(tuple(output.conditioning.shape), (2, 3, 4, 8, 64, 64))
        self.assertEqual(tuple(output.global_representation.shape), (2, 256))
        self.assertIsNotNone(output.exposure_clip_fraction)
        self.assertTrue(torch.isfinite(output.rate).all())
        self.assertTrue(torch.all(output.rate > 0))

    def test_baseline_residual_mode_requires_raw_counts_and_is_positive(self) -> None:
        torch.manual_seed(31)
        model = ProjectionDenoiser(baseline_residual=True).eval()
        batch = _batch()
        counts_raw = torch.full((1, 2, 3, 64, 64), 10.0)
        with torch.no_grad():
            output = model(**batch, counts_raw=counts_raw)
        self.assertEqual(tuple(output.rate.shape), (1, 2, 3, 64, 64))
        self.assertTrue(torch.isfinite(output.rate).all())
        self.assertTrue(torch.all(output.rate > 0))
        with self.assertRaises(ValueError):
            model(**batch)

    def test_parameter_count_is_in_specification_range(self) -> None:
        model = ProjectionDenoiser()
        parameters = sum(parameter.numel() for parameter in model.parameters())
        self.assertGreaterEqual(parameters, 1_000_000)
        self.assertLessEqual(parameters, 5_000_000)

    def test_padded_gate_and_view_do_not_change_valid_output(self) -> None:
        torch.manual_seed(4)
        model = ProjectionDenoiser().eval()
        base = _batch(gates=2, views=3)
        padded = _batch(gates=3, views=4)
        padded["angles_rad"][:, :3] = base["angles_rad"]
        padded["gate_phase_rad"][:, :2] = base["gate_phase_rad"]
        padded["valid_frames"][:, 2, :] = False
        padded["valid_frames"][:, :, 3] = False
        padded["visibility"][:, 2, :] = False
        padded["visibility"][:, :, 3] = False
        with torch.no_grad():
            base_output = model(**base)
            padded_output = model(**padded)
        self.assertTrue(torch.allclose(base_output.rate, padded_output.rate[:, :2, :3], atol=2e-5, rtol=2e-5))
        self.assertTrue(
            torch.allclose(
                base_output.global_representation,
                padded_output.global_representation,
                atol=2e-5,
                rtol=2e-5,
            )
        )

    def test_gradients_reach_frame_angular_temporal_and_decoder(self) -> None:
        torch.manual_seed(5)
        model = ProjectionDenoiser()
        output = model(**_batch())
        loss = output.rate.mean() + output.global_representation.square().mean()
        loss.backward()
        for name in (
            "backbone.stem.weight",
            "backbone.angular1.attention.in_proj_weight",
            "backbone.temporal1.attention.in_proj_weight",
            "decoder.rate_head.weight",
        ):
            gradient = dict(model.named_parameters())[name].grad
            self.assertIsNotNone(gradient, name)
            self.assertTrue(torch.isfinite(gradient).all(), name)

    def test_checkpointed_frame_chunks_preserve_eval_output_and_gradients(self) -> None:
        torch.manual_seed(6)
        reference = ProjectionDenoiser().eval()
        chunked = ProjectionDenoiser(backbone=ProjectionBackbone(frame_chunk_size=2))
        chunked.load_state_dict(reference.state_dict())
        batch = _batch(gates=2, views=3)
        with torch.no_grad():
            reference_output = reference(**batch)
            chunked_output = chunked.eval()(**batch)
        self.assertTrue(torch.allclose(reference_output.rate, chunked_output.rate, atol=1e-6, rtol=1e-6))
        chunked.train()
        output = chunked(**batch)
        (output.rate.mean() + output.global_representation.square().mean()).backward()
        self.assertIsNotNone(chunked.backbone.stem.weight.grad)
        self.assertTrue(torch.isfinite(chunked.backbone.stem.weight.grad).all())

    def test_invalid_detector_shape_is_rejected(self) -> None:
        model = ProjectionDenoiser()
        batch = _batch()
        batch["counts_input"] = torch.zeros(1, 2, 3, 1, 62, 64)
        with self.assertRaises(ValueError):
            model(**batch)


if __name__ == "__main__":
    unittest.main()
