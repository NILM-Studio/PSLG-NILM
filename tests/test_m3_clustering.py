"""M3 tests: all-candidate-k clustering + scan demotion + few-shot by tag.

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
from src.steps.few_shot_cluster_extract_step import FewShotClusterExtractStep

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
        step_dir = os.path.join("log", "r1", "TimeClustering_kmeans_on_detsec_on_clasp")
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
        scan_path = os.path.join("log", "rscan",
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


class TestFewShot(ChdirCase):
    def _build_scenario(self):
        """20 segments over 2 CSVs; clusters sized 9/9/2 -> the 2-sample cluster
        is the few-shot candidate (adjacent samples => not artifact-like)."""
        csv_dir = os.path.abspath("input_segments")
        os.makedirs(csv_dir)
        for fid in range(2):
            pd.DataFrame({"time": pd.date_range("2024-01-01", periods=100, freq="min"),
                          "power": np.ones(100) * (fid + 1)}).to_csv(
                os.path.join(csv_dir, f"file{fid}.csv"), index=False)

        rng = np.random.RandomState(1)
        feats = np.vstack([
            rng.randn(9, 4) * 0.01,                        # cluster A (file 0)
            [10, 10, 10, 10] + rng.randn(8, 4) * 0.01,     # cluster B (f0 tail + f1 head)
            [5, 20, 5, 20] + rng.randn(2, 4) * 0.01,       # cluster C (small, f1 tail)
            [10, 10, 10, 10] + rng.randn(1, 4) * 0.01,     # one more B -> 9/9/2
        ])
        indices = np.array([[0, s * 10] for s in range(10)] +
                           [[1, s * 10] for s in range(10)], dtype=np.int64)
        lengths = np.full((20, 1), 10, dtype=np.int64)
        return csv_dir, feats, indices, lengths

    def test_fewshot_consumes_tag_and_exports(self):
        csv_dir, feats, indices, lengths = self._build_scenario()
        wf = Workflow("rfew", "fridge", CFG)
        wf.add(FeatureStub(feats, lengths, indices, csv_dir))
        wf.add(TimeClusteringStep(cluster_method="kmeans", n_clusters=[3]))
        wf.add(FewShotClusterExtractStep(cluster_tag=None))  # single tag -> auto
        wf.run()

        m = wf.manifest
        step = m.data["steps"]["few_shot_cluster_extract"]
        self.assertEqual(step["extra"]["cluster_tag"], "kmeans_k3")
        self.assertEqual(step["extra"]["n_exported_segments"], 2)
        self.assertEqual(len(step["extra"]["true_few_shot_clusters"]), 1)
        self.assertEqual(step["extra"]["artifact_like_clusters"], [])

        with open(m.artifact_path("few_shot_cluster_extract", "summary")) as f:
            summary = json.load(f)
        cid = summary["true_few_shot_clusters"][0]
        self.assertEqual(summary["cluster_reports"][str(cid)]["cluster_size"], 2)

        with open(m.artifact_path("few_shot_cluster_extract", "export_manifest")) as f:
            exports = json.load(f)
        self.assertEqual(len(exports), 2)
        for e in exports:
            self.assertTrue(os.path.exists(e["export_path"]))
            self.assertEqual(e["length"], 10)

    def test_fewshot_standalone_via_manifest(self):
        csv_dir, feats, indices, lengths = self._build_scenario()
        Workflow("rfew2", "fridge", CFG).add(
            FeatureStub(feats, lengths, indices, csv_dir)).add(
            TimeClusteringStep(cluster_method="kmeans", n_clusters=[3])).run()

        # brand-new workflow, same run-id, ONLY the few-shot step
        wf2 = Workflow("rfew2", "fridge", CFG)
        wf2.add(FewShotClusterExtractStep(cluster_tag="kmeans_k3"))
        wf2.run()
        self.assertEqual(
            wf2.manifest.data["steps"]["few_shot_cluster_extract"]["extra"]["n_exported_segments"], 2)

    def test_ambiguous_tags_require_explicit_cluster_tag(self):
        csv_dir, feats, indices, lengths = self._build_scenario()
        wf = Workflow("ramb", "fridge", CFG)
        wf.add(FeatureStub(feats, lengths, indices, csv_dir))
        wf.add(TimeClusteringStep(cluster_method="kmeans", n_clusters=[2, 3]))
        wf.add(FewShotClusterExtractStep(cluster_tag=None))
        with self.assertRaises(ValueError):
            wf.run()


if __name__ == "__main__":
    unittest.main(verbosity=2)
