import unittest

import numpy as np

from src.steps.nilm_dataset_step import NilmDatasetStep


class NilmDatasetStepTests(unittest.TestCase):
    def test_resample_interval_rejects_long_gap(self):
        timestamp = np.array([0, 6, 12, 60, 66], dtype=np.int64)
        power = np.arange(5, dtype=np.float32)
        payload, reason = NilmDatasetStep._resample_interval(
            timestamp, power, power, 0, 66, period=6, max_gap=30)
        self.assertIsNone(payload)
        self.assertEqual(reason, "gap_exceeds_30s")

    def test_resample_interval_returns_uniform_float32_pair(self):
        timestamp = np.array([1, 7, 13, 19], dtype=np.int64)
        mains = np.array([100, 110, 120, 130], dtype=np.float32)
        appliance = np.array([0, 10, 20, 30], dtype=np.float32)
        payload, reason = NilmDatasetStep._resample_interval(
            timestamp, mains, appliance, 1, 19, period=6, max_gap=30)
        self.assertIsNone(reason)
        self.assertEqual(payload["timestamp"].tolist(), [6, 12, 18])
        self.assertEqual(payload["mains"].dtype, np.float32)
        self.assertTrue(np.allclose(payload["appliance"], [8.333333, 18.333333, 28.333333]))

    def test_stratified_select_covers_groups_before_repeating(self):
        records = [
            {"class_id": 0, "mode_id": 0, "id": value} for value in range(4)
        ] + [
            {"class_id": 1, "mode_id": 0, "id": value} for value in range(4, 8)
        ]
        selected = NilmDatasetStep._stratified_select(
            records, 2, np.random.default_rng(42))
        self.assertEqual(
            {(row["class_id"], row["mode_id"]) for row in selected},
            {(0, 0), (1, 0)},
        )


if __name__ == "__main__":
    unittest.main()
