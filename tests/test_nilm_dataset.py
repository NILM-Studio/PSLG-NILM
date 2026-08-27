import os
import tempfile
import unittest

import numpy as np

from src.steps.nilm_dataset_step import NilmDatasetStep
from src.steps.nilm_continuous_dataset_step import NilmContinuousDatasetStep
from src.generation.primitive_library import Primitive


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

    def test_stratified_order_produces_nested_budget_prefixes(self):
        records = [
            {"class_id": class_id, "mode_id": 0, "activity_id": str(index)}
            for index, class_id in enumerate([0, 0, 0, 1, 1, 1])
        ]
        ordered = NilmDatasetStep._stratified_order(
            records, np.random.default_rng(42))
        one = {row["activity_id"] for row in ordered[:2]}
        two = {row["activity_id"] for row in ordered[:4]}
        self.assertTrue(one < two)
        self.assertEqual({row["class_id"] for row in ordered[:2]}, {0, 1})

    def test_real_ratios_are_sorted_and_deduplicated(self):
        step = NilmDatasetStep(
            "kmeans_k2_merged", "pair.csv",
            real_ratios=(0.2, 0.01, 0.1, 0.01))
        self.assertEqual(step.real_ratios, [0.01, 0.1, 0.2])

    def test_traditional_augmentation_preserves_background_and_off_state(self):
        mains = np.array([100, 300, 500], dtype=np.float32)
        appliance = np.array([0, 100, 300], dtype=np.float32)
        augmented_mains, augmented_appliance, parameters = (
            NilmDatasetStep._traditional_augment(
                mains, appliance, np.random.default_rng(2),
                (1.0, 1.0), noise_ratio=0.0, active_threshold=10))
        self.assertTrue(np.array_equal(augmented_appliance, appliance))
        self.assertTrue(np.array_equal(augmented_mains, mains))
        self.assertEqual(parameters["scale"], 1.0)
        self.assertEqual(augmented_appliance[0], 0.0)

    def test_continuous_chunks_do_not_bridge_long_gap(self):
        step = NilmContinuousDatasetStep(
            "kmeans_k2_merged", "pair.csv", sample_period_seconds=6,
            max_gap_seconds=30, min_off_samples=2, max_chunk_samples=100)
        timestamp = np.asarray([0, 6, 12, 100, 106, 112], dtype=np.int64)
        mains = np.asarray([10, 11, 12, 20, 21, 22], dtype=np.float32)
        appliance = np.zeros(6, dtype=np.float32)
        chunks = step._uniform_chunks(
            timestamp, mains, appliance, None, None)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0]["timestamp"].tolist(), [0, 6, 12])
        self.assertEqual(chunks[1]["timestamp"].tolist(), [102, 108])

    def test_off_chunks_exclude_active_samples(self):
        step = NilmContinuousDatasetStep(
            "kmeans_k2_merged", "pair.csv", active_threshold_watts=10,
            min_off_samples=2, max_chunk_samples=100)
        chunk = {
            "timestamp": np.arange(7),
            "mains": np.arange(7, dtype=np.float32),
            "appliance": np.asarray([0, 1, 20, 30, 0, 0, 0], dtype=np.float32),
        }
        off = step._off_chunks([chunk])
        self.assertEqual([len(row["timestamp"]) for row in off], [2, 3])
        self.assertTrue(all(np.max(row["appliance"]) <= 10 for row in off))

    def test_off_pool_is_repeated_without_adding_unique_backgrounds(self):
        records = [
            {"path": "off_a.npz", "length_samples": 4},
            {"path": "off_b.npz", "length_samples": 6},
        ]
        selected = NilmContinuousDatasetStep._repeat_off(records, 24)
        self.assertEqual(
            [row["path"] for row in selected],
            ["off_a.npz", "off_b.npz", "off_a.npz", "off_b.npz", "off_a.npz"],
        )
        self.assertEqual(sum(row["length_samples"] for row in selected), 24)
        self.assertEqual({row["path"] for row in selected},
                         {"off_a.npz", "off_b.npz"})

    def test_budget_synthesis_never_uses_out_of_budget_primitives(self):
        step = NilmDatasetStep(
            "kmeans_k2_merged", "pair.csv", synthesis_scope="budget_local")
        real_subset = [
            {"activity_id": "1", "class_id": 0, "mode_id": 0},
            {"activity_id": "2", "class_id": 0, "mode_id": 0},
        ]
        catalog = {"activities": {
            "1": {"blocks": [
                {"state_label": 0, "length_samples": 3},
                {"state_label": 1, "length_samples": 3},
            ]},
            "2": {"blocks": [
                {"state_label": 0, "length_samples": 3},
                {"state_label": 1, "length_samples": 3},
            ]},
        }}
        primitives = [
            Primitive(0, 0, 1, 0, np.asarray([1, 2, 3], dtype=np.float32)),
            Primitive(1, 1, 1, 3, np.asarray([10, 11, 12], dtype=np.float32)),
            Primitive(2, 0, 2, 0, np.asarray([2, 3, 4], dtype=np.float32)),
            Primitive(3, 1, 2, 3, np.asarray([11, 12, 13], dtype=np.float32)),
            Primitive(4, 0, 99, 0, np.asarray([100, 100, 100], dtype=np.float32)),
        ]
        payload = {
            "mains": np.full(6, 50, dtype=np.float32),
            "appliance": np.asarray([1, 2, 3, 10, 11, 12], dtype=np.float32),
        }
        real_by_activity = {
            "1": {"payload": payload}, "2": {"payload": payload},
        }
        with tempfile.TemporaryDirectory() as directory:
            generated = step._generate_budget_cycles(
                directory, "01pct", real_subset, real_by_activity,
                catalog, primitives)
            self.assertEqual(len(generated), 2)
            for row in generated:
                self.assertEqual(set(row["budget_activity_ids"]), {1, 2})
                self.assertTrue(
                    set(row["primitive_source_activity_ids"]).issubset({1, 2}))
                with np.load(os.path.join(directory, row["file"])) as output:
                    self.assertTrue(np.all(output["mains"] >= output["appliance"]))


if __name__ == "__main__":
    unittest.main()
