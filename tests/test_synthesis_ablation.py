"""Tests for synthesis-ablation aggregation and pairing checks."""
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.summarize_synthesis_ablation import (
    SUMMARY_METRICS, evaluation_dir, experiment_tag, paired_deltas, summarize)


class SynthesisAblationTest(unittest.TestCase):
    def test_experiment_tags_are_stable(self):
        self.assertEqual(experiment_tag("independent", 42), "independent_seed42")
        self.assertEqual(
            experiment_tag("cycle_neighbors", 44, 10),
            "cycle_neighbors_k10_seed44")

    def test_paired_deltas_require_identical_duration_metric(self):
        left = pd.DataFrame([{
            "class_id": 0, "mode_id": 0, "metric": "duration_seconds",
            "normalized_wasserstein": 0.1,
        }])
        right = left.copy()
        result = paired_deltas(left, right, seed=42, neighbors=5)
        self.assertEqual(result["normalized_wasserstein_delta"].iloc[0], 0.0)
        right.loc[0, "normalized_wasserstein"] = 0.2
        with self.assertRaises(ValueError):
            paired_deltas(left, right, seed=42, neighbors=5)

    def test_summarize_writes_run_and_pair_tables(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            for method, neighbors in (("independent", None),
                                      ("cycle_neighbors", 5)):
                tag = experiment_tag(method, 42, neighbors)
                directory = evaluation_dir(root, "kmeans_k4_merged", tag)
                directory.mkdir(parents=True)
                with open(directory / "quality_summary.json", "w") as f:
                    json.dump({metric: 0.1 for metric in SUMMARY_METRICS}, f)
                pd.DataFrame([{
                    "class_id": 0, "mode_id": 0,
                    "metric": "duration_seconds",
                    "normalized_wasserstein": 0.1,
                }]).to_csv(directory / "distribution_metrics.csv", index=False)
            output = Path(tmp) / "output"
            result = summarize(
                root, "kmeans_k4_merged", [5], [42], output)
            self.assertEqual(result["report"]["completed_experiments"], 2)
            self.assertEqual(result["report"]["paired_comparisons"], 1)
            self.assertTrue((output / "ablation_aggregate.csv").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
