"""M3 tests: all-candidate-k clustering + scan demotion.

Uses small synthetic blobs (no TF); sklearn is required (it is a real runtime
dependency of the clustering step).
"""
import json
import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.framework.step import Step
from src.framework.workflow import Workflow
from src.steps.time_clustering_step import TimeClusteringStep

CFG = {"paths": {"cache_dir": ".cache"}}


def make_blobs():
    """30 samples in 3 tight 4-D blobs (10 each)."""
    rng = np.random.RandomState(0)
    centers = np.array([[0, 0, 0, 0], [10, 10, 10, 10], [20, 20, 0, 0]], dtype=np.float64)
    return np.vstack([c + rng.randn(10, 4) * 0.01 for c in centers])


class FeatureStub(Step):
    step_type = "stub_features"

    def __init__(self, features, lengths=None, indices=None, csv_dir=None):
        super().__init__()
        self._f, self._l, self._i, self._csv = features, lengths, indices, csv_dir

    def run(self, context):
        context["data"]["features"] = self._f
        if self._l is not None:
            context["data"]["lengths"] = self._l
        if self._i is not None:
            context["data"]["indices"] = self._i
        if self._csv:
            context["input_root"] = self._csv
            # record like extract_active_data would, so standalone reruns
            # (fresh workflow, same run-id) can resolve it from the manifest
            context["manifest"].add_step(
                "extract_active_data", "stub", "stub",
                {"segments_dir": self.rel(context, self._csv)})
        return context


class ChdirCase(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self._cwd = os.getcwd()
        os.chdir(self._td.name)

    def tearDown(self):
        os.chdir(self._cwd)
        self._td.cleanup()


class TestKMeansAllK(ChdirCase):
    def test_every_candidate_k_gets_a_tagged_result(self):
        wf = Workflow("r1", "fridge", CFG)
        wf.add(FeatureStub(make_blobs()))
        wf.add(TimeClusteringStep(cluster_method="kmeans", n_clusters=[2, 3]))
        wf.run()

        m = wf.manifest
        self.assertEqual(m.cluster_tags(), ["kmeans_k2", "kmeans_k3"])
        for tag, k in (("kmeans_k2", 2), ("kmeans_k3", 3)):
            labels = np.load(m.cluster_artifact_path(tag, "labels"))
            self.assertEqual(len(set(labels.tolist())), k)
            with open(m.cluster_artifact_path(tag, "metrics")) as f:
                metrics = json.load(f)
            self.assertIsNotNone(metrics["silhouette_score"])
            self.assertEqual(metrics["n_clusters_requested"], k)
            # shared artifacts resolvable per tag
            feats = np.load(m.cluster_artifact_path(tag, "feature_matrix"))
            self.assertEqual(feats.shape, (30, 4))

        # k=3 on clean blobs: essentially perfect silhouette
        with open(m.cluster_artifact_path("kmeans_k3", "metrics")) as f:
            self.assertGreater(json.load(f)["silhouette_score"], 0.8)

        # no redundant copies / no figures in the step dir
        step_dir = os.path.join("runs", "r1", "TimeClustering_kmeans_on_detsec_on_clasp")
        top = sorted(os.listdir(step_dir))
        self.assertEqual(top, ["feature_matrix.npy", "kept_rows.npy", "kmeans_k2",
                               "kmeans_k3", "seq_len.npy"])
        result_files = sorted(os.listdir(os.path.join(step_dir, "kmeans_k3")))
        self.assertIn("cluster_labels.npy", result_files)
        self.assertIn("metrics.json", result_files)
        self.assertFalse(any(f.endswith(".png") for f in result_files))
        self.assertFalse(any(f.startswith("Cluster_") for f in result_files))
        self.assertNotIn("org_data.npy", result_files)

    def test_nan_rows_dropped_and_tracked(self):
        feats = make_blobs()
        feats[0] = np.nan
        wf = Workflow("rnan", "fridge", CFG)
        wf.add(FeatureStub(feats))
        wf.add(TimeClusteringStep(cluster_method="kmeans", n_clusters=[3]))
        wf.run()
        m = wf.manifest
        labels = np.load(m.cluster_artifact_path("kmeans_k3", "labels"))
        self.assertEqual(len(labels), 29)
        kept = np.load(m.cluster_artifact_path("kmeans_k3", "kept_rows"))
        self.assertEqual(kept.tolist(), list(range(1, 30)))


class TestScanDemotion(ChdirCase):
    def test_scan_is_diagnostic_only(self):
        wf = Workflow("rscan", "fridge", CFG)
        wf.add(FeatureStub(make_blobs()))
        wf.add(TimeClusteringStep(cluster_method="kmeans-scan", n_clusters=[2, 3, 4]))
        wf.run()
        m = wf.manifest

        self.assertEqual(m.cluster_tags(), [])  # no results registered
        scan_path = os.path.join("runs", "rscan",
                                 "TimeClustering_kmeans-scan_on_detsec_on_clasp",
                                 "kmeans_scan.json")
        with open(scan_path) as f:
            scan = json.load(f)
        self.assertEqual([r["n_clusters"] for r in scan["records"]], [2, 3, 4])
        self.assertEqual(scan["recommended_n_clusters"], 3)  # clean 3-blob data
        self.assertIn("diagnostic", scan["selection_rule"])


class TestDbscan(ChdirCase):
    def test_single_tag(self):
        wf = Workflow("rdb", "fridge", CFG)
        wf.add(FeatureStub(make_blobs()))
        wf.add(TimeClusteringStep(cluster_method="dbscan",
                                  dbscan_eps=0.5, dbscan_min_pts=2))
        wf.run()
        self.assertEqual(wf.manifest.cluster_tags(), ["dbscan"])
        with open(wf.manifest.cluster_artifact_path("dbscan", "metrics")) as f:
            metrics = json.load(f)
        self.assertEqual(metrics["n_clusters"], 3)
        self.assertEqual(metrics["n_noise"], 0)



if __name__ == "__main__":
    unittest.main(verbosity=2)
