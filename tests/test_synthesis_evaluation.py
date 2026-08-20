"""Tests for held-out synthetic-cycle quality metrics."""
import unittest

import numpy as np

from src.steps.synthesis_evaluation_step import SynthesisEvaluationStep


class SynthesisEvaluationTest(unittest.TestCase):
    def test_shape_resampling_is_offset_and_scale_invariant(self):
        base = np.sin(np.linspace(0, 4 * np.pi, 100))
        left = SynthesisEvaluationStep._resample_shape(base, 64)
        right = SynthesisEvaluationStep._resample_shape(base * 20 + 300, 64)
        np.testing.assert_allclose(left, right, atol=1e-10)

    def test_distribution_metrics_are_zero_for_identical_samples(self):
        row = SynthesisEvaluationStep._distribution_row(
            1, 2, "energy_wh", [1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        self.assertEqual(row["wasserstein"], 0.0)
        self.assertEqual(row["normalized_wasserstein"], 0.0)
        self.assertEqual(row["ks_statistic"], 0.0)

    def test_missing_generated_group_is_explicit(self):
        row = SynthesisEvaluationStep._distribution_row(
            0, 0, "duration_seconds", [10.0, 20.0], [])
        self.assertEqual(row["real_count"], 2)
        self.assertEqual(row["generated_count"], 0)
        self.assertIsNone(row["wasserstein"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
