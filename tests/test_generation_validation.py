"""Fixtures for validation of inherited run artifacts, without real UK-DALE data."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.audit_budget_dataset import dataset_directories
from scripts.validate_generation_run import coverage, inspect_pair, upstream_report


class CoverageTests(unittest.TestCase):
    def test_complete_half_open_spans(self):
        result = coverage(10, [(4, 10), (0, 4)])
        self.assertEqual(result["covered_samples"], 10)
        self.assertEqual(result["gap_samples"], 0)
        self.assertEqual(result["overlap_samples"], 0)
        self.assertEqual(result["invalid_intervals"], 0)

    def test_gaps_and_overlaps_are_separate(self):
        gap = coverage(10, [(0, 4), (6, 10)])
        self.assertEqual(gap["gap_samples"], 2)
        self.assertEqual(gap["overlap_samples"], 0)
        overlap = coverage(10, [(0, 6), (4, 10)])
        self.assertEqual(overlap["gap_samples"], 0)
        self.assertEqual(overlap["overlap_samples"], 2)

    def test_out_of_bounds_spans_are_flagged_even_when_clipped_coverage_is_complete(self):
        result = coverage(10, [(-1, 5), (5, 12)])
        self.assertEqual(result["covered_samples"], 10)
        self.assertEqual(result["invalid_intervals"], 2)


class PairInspectionTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "pair.npz"

    def write_pair(self, **overrides):
        payload = {
            "timestamp": np.array([100, 106, 112], dtype=np.float64),
            "mains": np.array([100, 120, 140], dtype=np.float32),
            "appliance": np.array([0, 20, 40], dtype=np.float32),
        }
        payload.update(overrides)
        np.savez_compressed(self.path, **payload)

    def test_valid_grid_and_energy_use_declared_sample_period(self):
        self.write_pair()
        result = inspect_pair(self.path, 6)
        self.assertEqual(result["samples"], 3)
        self.assertEqual(result["on_samples"], 2)
        self.assertAlmostEqual(result["appliance_energy_wh"], 0.1)

    def test_nonpositive_or_nonfinite_period_is_rejected_for_single_sample(self):
        self.write_pair(timestamp=np.array([100]), mains=np.array([100]),
                        appliance=np.array([20]))
        for period in (0, -6, float("nan"), float("inf")):
            with self.subTest(period=period), self.assertRaises(ValueError):
                inspect_pair(self.path, period)

    def test_nonfinite_values_in_every_required_array_are_rejected(self):
        for key in ("timestamp", "mains", "appliance"):
            for invalid in (float("nan"), float("inf")):
                with self.subTest(key=key, invalid=invalid):
                    self.write_pair(**{key: np.array([100, invalid, 112])})
                    with self.assertRaises(ValueError):
                        inspect_pair(self.path, 6)

    def test_irregular_or_duplicate_time_grid_is_rejected(self):
        for timestamps in ([100, 106, 113], [100, 100, 106], [112, 106, 100]):
            with self.subTest(timestamps=timestamps):
                self.write_pair(timestamp=np.array(timestamps))
                with self.assertRaises(ValueError):
                    inspect_pair(self.path, 6)


class UpstreamInterfaceTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.sources = self.root / "inherited_sources"
        self.cluster = self.root / "custom_cluster_snapshot"
        self.sources.mkdir()
        self.cluster.mkdir()
        pd.DataFrame({
            "timestamp": np.arange(100, 136, 6),
            "power": [0, 100, 200, 150, 50, 0],
        }).to_csv(self.sources / "cycle_0.csv", index=False)
        self.tag = "kmeans_k2_merged"
        self.labels = np.array([1, 2], dtype=np.int64)
        self.indices = np.array([[0, 0, 1], [0, 3, 2]], dtype=np.int64)
        self.lengths = np.array([3, 3], dtype=np.int64)
        self.sequences = {"0": [
            {"block_id": 0, "start": 0, "end": 3, "length_samples": 3,
             "length_seconds": 18, "state_label": 1, "n_segments": 1},
            {"block_id": 1, "start": 3, "end": 6, "length_samples": 3,
             "length_seconds": 18, "state_label": 2, "n_segments": 1},
        ]}
        artifacts = {
            key: str((self.cluster / name).relative_to(self.root))
            for key, name in (("labels", "labels.npy"), ("indices", "indices.npy"),
                              ("seq_len", "lengths.npy"),
                              ("state_sequences", "states.json"))
        }
        self.manifest = {"steps": {
            "extract_active_data": {"artifacts": {
                "segments_dir": str(self.sources.relative_to(self.root))}},
            "time_clustering": {"results": {self.tag: {"artifacts": artifacts}}},
        }}

    def report(self):
        np.save(self.cluster / "labels.npy", self.labels)
        np.save(self.cluster / "indices.npy", self.indices)
        np.save(self.cluster / "lengths.npy", self.lengths)
        (self.cluster / "states.json").write_text(json.dumps(self.sequences))
        (self.root / "run_manifest.json").write_text(json.dumps(self.manifest))
        return upstream_report(self.root, self.tag)

    def test_complete_inherited_fixture_passes(self):
        result = self.report()
        self.assertTrue(result["upstream_interface_passed"], result["errors"])
        self.assertEqual(result["counts"]["activities"], 1)
        self.assertEqual(result["counts"]["source_samples"], 6)
        self.assertEqual(result["upstream_representation_holdout"], "not_verified")

    def test_primitive_coverage_gap_fails(self):
        self.lengths[0] = 2
        result = self.report()
        self.assertFalse(result["upstream_interface_passed"])
        self.assertTrue(result["errors"])

    def test_primitive_coverage_overlap_fails(self):
        self.lengths[0] = 4
        result = self.report()
        self.assertFalse(result["upstream_interface_passed"])
        self.assertTrue(result["errors"])

    def test_equal_total_state_length_does_not_hide_bad_start_end_or_label(self):
        for key, value in (("start", 1), ("end", 4), ("state_label", 999)):
            with self.subTest(key=key):
                block = self.sequences["0"][0]
                original = block[key]
                block[key] = value
                try:
                    result = self.report()
                    self.assertFalse(result["upstream_interface_passed"], result)
                    self.assertTrue(result["errors"])
                finally:
                    block[key] = original

    def test_unknown_sequence_activity_id_is_not_silently_ignored(self):
        self.sequences["99"] = [dict(self.sequences["0"][0])]
        result = self.report()
        self.assertFalse(result["upstream_interface_passed"])
        self.assertTrue(result["errors"])


class ManifestDatasetLocationTests(unittest.TestCase):
    def test_relative_and_absolute_manifest_paths_determine_dataset_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "log" / "inherited_run"
            run.mkdir(parents=True)
            dataset = root / "actual_budget_snapshot"
            continuous = root / "different_continuous_snapshot"
            dataset.mkdir()
            continuous.mkdir()
            # These names deliberately do not match any historical hard-coded folder.
            manifest = {"steps": {
                "nilm_dataset": {"artifacts": {
                    "dataset_manifest": "../../actual_budget_snapshot/nilm_dataset_manifest.json"}},
                "nilm_continuous_dataset": {"artifacts": {
                    "dataset_manifest": str(continuous / "nilm_dataset_manifest.json")}},
            }}
            (run / "run_manifest.json").write_text(json.dumps(manifest))
            self.assertEqual(dataset_directories(run), (dataset.resolve(), continuous.resolve()))


if __name__ == "__main__":
    unittest.main()
