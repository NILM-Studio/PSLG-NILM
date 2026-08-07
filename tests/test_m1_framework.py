"""Lightweight smoke test for the M1 framework pieces.

Uses only stdlib + stub Steps (no TF/stumpy/sklearn), so it runs anywhere.
Covers: RunManifest round-trip & path resolution, the linear Workflow, and
main.py's CLI parsing helpers.
"""
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.framework.run_manifest import RunManifest
from src.framework.step import Step
from src.framework.workflow import Workflow
import main as mainmod


class StubStep(Step):
    """Records two artifacts and hands a value to the next step via context."""
    step_type = "stub_a"

    def run(self, context):
        log_dir = self.log_dir(context)
        a_path = os.path.join(log_dir, "a.txt")
        with open(a_path, "w") as f:
            f.write("hello")
        context["data"]["a_value"] = 42
        self.record(context, artifacts={"a": self.rel(context, a_path)})
        return context


class StubDownstream(Step):
    """Reads the upstream value from context, or from the manifest if absent."""
    step_type = "stub_b"

    def run(self, context):
        val = context["data"].get("a_value")
        if val is None:  # simulate a standalone rerun resolving via manifest
            p = self.resolve(context, "stub_a", "a")
            with open(p) as f:
                val = len(f.read())
        context["data"]["b_value"] = val
        return context


class TestManifest(unittest.TestCase):
    def test_add_and_resolve_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "run_manifest.json")
            m = RunManifest(path, run_id="r1", appliance="fridge")
            m.set_variant(segment_method="clasp")
            m.add_step("time_segmentation", "clasp", "TimeSegmentation_clasp",
                       {"X": "TimeSegmentation_clasp/X.npy"})
            m.save()

            loaded = RunManifest.load_or_create(path)
            self.assertEqual(loaded.artifact_path("time_segmentation", "X"),
                             os.path.join(td, "TimeSegmentation_clasp", "X.npy"))
            self.assertIsNone(loaded.artifact_path("time_segmentation", "missing"))
            self.assertEqual(loaded.data["variants"]["segment_method"], "clasp")

    def test_cluster_results(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "run_manifest.json")
            m = RunManifest(path, run_id="r1")
            m.add_cluster_result("kmeans_k3", "TimeClustering_kmeans_k3",
                                 {"labels": "TimeClustering_kmeans_k3/cluster_labels.npy"})
            m.add_cluster_result("kmeans_k4", "TimeClustering_kmeans_k4",
                                 {"labels": "TimeClustering_kmeans_k4/cluster_labels.npy"})
            self.assertEqual(m.cluster_tags(), ["kmeans_k3", "kmeans_k4"])
            got = m.cluster_artifact_path("kmeans_k4", "labels")
            self.assertEqual(got, os.path.normpath(os.path.join(
                td, "TimeClustering_kmeans_k4", "cluster_labels.npy")))


class TestWorkflow(unittest.TestCase):
    def test_linear_run_and_manifest(self):
        with tempfile.TemporaryDirectory() as td:
            cwd = os.getcwd()
            os.chdir(td)
            try:
                wf = Workflow("testrun", "fridge", {"paths": {"cache_dir": ".cache"}})
                wf.add(StubStep()).add(StubDownstream())
                ctx = wf.run()
                self.assertEqual(ctx["data"]["b_value"], 42)
                mpath = os.path.join(td, "log", "testrun", "run_manifest.json")
                self.assertTrue(os.path.exists(mpath))
                with open(mpath) as f:
                    data = json.load(f)
                self.assertIn("stub_a", data["steps"])
            finally:
                os.chdir(cwd)

    def test_standalone_rerun_resolves_via_manifest(self):
        """Re-running only the downstream step finds upstream artifacts on disk."""
        with tempfile.TemporaryDirectory() as td:
            cwd = os.getcwd()
            os.chdir(td)
            try:
                Workflow("r", "app", {}).add(StubStep()).run()
                # now a NEW workflow for the same run_id, only the downstream step
                wf2 = Workflow("r", "app", {})
                wf2.add(StubDownstream())
                ctx = wf2.run()
                self.assertEqual(ctx["data"]["b_value"], 5)  # len("hello")
            finally:
                os.chdir(cwd)


class TestCLIParsing(unittest.TestCase):
    def test_parse_steps(self):
        self.assertEqual(mainmod.parse_steps("all", mainmod.IMPLEMENTED_STEPS),
                         mainmod.IMPLEMENTED_STEPS)
        self.assertEqual(mainmod.parse_steps("segment,feature", mainmod.IMPLEMENTED_STEPS),
                         ["segment", "feature"])
        self.assertEqual(mainmod.parse_steps("feature,segment", mainmod.IMPLEMENTED_STEPS),
                         ["segment", "feature"])  # canonical order enforced
        with self.assertRaises(ValueError):
            mainmod.parse_steps("bogus", mainmod.IMPLEMENTED_STEPS)

    def test_parse_int_list(self):
        self.assertEqual(mainmod.parse_int_list("3,4,5"), [3, 4, 5])
        self.assertIsNone(mainmod.parse_int_list(""))
        self.assertIsNone(mainmod.parse_int_list(None))

    def test_resolve_selection_defaults(self):
        args = argparse_ns(segment_method=None, feature_model=None)
        sel = mainmod.resolve_selection(args, {"run": {"appliance": "fridge"},
                                               "paths": {"raw_series": "input/x.csv"}})
        self.assertEqual(sel["appliance"], "fridge")
        self.assertEqual(sel["segment_method"], "clasp")
        self.assertEqual(sel["feature_model"], "detsec")
        self.assertEqual(sel["n_clusters"], [3, 4, 5])
        self.assertEqual(sel["raw_series"], "input/x.csv")

    def test_resolve_selection_cli_overrides(self):
        args = argparse_ns(segment_method="fluss", feature_model="bilstm_ae",
                           appliance="kettle", n_clusters="2,3", run_id="r9")
        sel = mainmod.resolve_selection(args, {})
        self.assertEqual(sel["appliance"], "kettle")
        self.assertEqual(sel["segment_method"], "fluss")
        self.assertEqual(sel["feature_model"], "bilstm_ae")
        self.assertEqual(sel["n_clusters"], [2, 3])
        self.assertEqual(sel["run_id"], "r9")


def argparse_ns(**over):
    base = dict(segment_method=None, feature_model=None, cluster_method=None,
                n_clusters=None, cluster_tag=None, appliance=None, run_id=None,
                raw_series=None)
    base.update(over)

    class NS:  # minimal stand-in for argparse.Namespace
        pass
    ns = NS()
    for k, v in base.items():
        setattr(ns, k, v)
    return ns


if __name__ == "__main__":
    unittest.main(verbosity=2)
