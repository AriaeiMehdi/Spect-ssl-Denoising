import unittest

import torch

from gated_spect.losses import (
    binomial_count_split,
    count_split_seed,
    make_mask_plan,
    poisson_deviance,
    weighted_poisson_nll,
)


class LossTests(unittest.TestCase):
    def test_binomial_split_conserves_counts_and_reproduces(self) -> None:
        counts = torch.tensor([[0, 1, 10, 1000]], dtype=torch.int64)
        seed = count_split_seed(17, 3, "sample-a", 2)
        first_a, second_a = binomial_count_split(counts, 0.6, seed=seed)
        first_b, second_b = binomial_count_split(counts, 0.6, seed=seed)
        self.assertTrue(torch.equal(first_a, first_b))
        self.assertTrue(torch.equal(second_a, second_b))
        self.assertTrue(torch.equal(first_a + second_a, counts))
        self.assertTrue(torch.all(first_a >= 0))
        self.assertTrue(torch.all(second_a >= 0))

    def test_augmentation_seed_is_worker_independent_and_stream_specific(self) -> None:
        worker_zero = count_split_seed(17, 3, "sample-a", "student_a")
        worker_two = count_split_seed(17, 3, "sample-a", "student_a")
        self.assertEqual(worker_zero, worker_two)
        self.assertNotEqual(worker_zero, count_split_seed(17, 3, "sample-a", "student_b"))

    def test_binomial_samples_have_reasonable_mean(self) -> None:
        counts = torch.full((2000,), 100, dtype=torch.int64)
        samples = torch.stack(
            [binomial_count_split(counts, 0.4, seed=seed)[0].float() for seed in range(20)], dim=0
        )
        self.assertAlmostEqual(float(samples.mean()), 40.0, delta=0.6)
        self.assertGreater(float(samples.var(unbiased=False)), 20.0)

    def test_mask_plan_is_deterministic_and_preserves_invalid_frames(self) -> None:
        valid = torch.ones(4, 8, dtype=torch.bool)
        valid[3, 7] = False
        q = torch.ones(4, 8)
        first = make_mask_plan(valid, q, height=16, width=16, seed=9)
        second = make_mask_plan(valid, q, height=16, width=16, seed=9)
        self.assertEqual(first.task, second.task)
        self.assertTrue(torch.equal(first.visibility, second.visibility))
        self.assertTrue(torch.equal(first.masked_target, second.masked_target))
        self.assertTrue(torch.equal(first.visible_target, second.visible_target))
        self.assertFalse(first.visibility[3, 7].any())
        self.assertTrue(torch.equal(first.masked_target, valid[:, :, None, None] & ~first.visibility))

    def test_weighted_loss_is_patient_balanced_and_deviance_zero_at_mean(self) -> None:
        mean = torch.ones(2, 1, 2, 2, 2)
        counts = torch.ones_like(mean)
        frame_weight = torch.ones(2, 1, 2)
        self.assertAlmostEqual(float(poisson_deviance(mean, counts)), 0.0, places=7)
        # Frame weights apply to every detector bin, so the reduction is a
        # per-bin mean rather than a sum over H*W pixels per frame.
        self.assertAlmostEqual(
            float(weighted_poisson_nll(mean, counts, frame_weight)),
            1.0,
            places=7,
        )

    def test_weighted_loss_denominator_counts_selected_bins(self) -> None:
        mean = torch.full((1, 1, 1, 2, 2), 2.0)
        counts = torch.zeros_like(mean)
        frame_weight = torch.ones(1, 1, 1)
        expected = 2.0 - 0.0  # constant-free Poisson NLL for y=0, per bin
        self.assertAlmostEqual(
            float(weighted_poisson_nll(mean, counts, frame_weight)),
            expected,
            places=7,
        )


if __name__ == "__main__":
    unittest.main()
