"""Tests for real-primitive cycle synthesis and state-order sampling."""
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
from src.generation.transition_model import StateTransitionModel
from src.steps.primitive_synthesis_step import PrimitiveSynthesisStep


class SynthesisUpstreamStub(Step):
    step_type = "synthesis_upstream_stub"

    def run(self, context):
        root = context["log_root"]
        segments = os.path.join(root, "activities")
        os.makedirs(segments)
        for fid, offset in enumerate((0.0, 100.0)):
            pd.DataFrame({
                "timestamp": fid * 100 + np.arange(24),
                "power": offset + np.arange(24, dtype=np.float64),
            }).to_csv(os.path.join(segments, f"activity_{fid}.csv"), index=False)
        context["manifest"].add_step(
            "extract_active_data", "stub", "activities",
            {"segments_dir": self.rel(context, segments)})

        # Three primitives per activity, already labelled by a merged result.
        labels = np.array([0, 1, 0, 1, 0, 1], dtype=np.int64)
        indices = np.array([[0, 0], [0, 8], [0, 16],
                            [1, 0], [1, 8], [1, 16]], dtype=np.int64)
        lengths = np.full(6, 8, dtype=np.int64)
        result_dir = os.path.join(root, "merged", "kmeans_k2_merged")
        os.makedirs(result_dir)
        artifacts = {}
        for key, value, name in (
                ("labels", labels, "cluster_labels.npy"),
                ("indices", np.column_stack((indices, labels)), "indices.npy"),
                ("seq_len", lengths, "seq_len.npy")):
            path = os.path.join(result_dir, name)
            np.save(path, value)
            artifacts[key] = self.rel(context, path)

        sequences = {
            "0": [{"state_label": 0, "length_samples": 8},
                  {"state_label": 1, "length_samples": 8},
                  {"state_label": 0, "length_samples": 8}],
            "1": [{"state_label": 1, "length_samples": 8},
                  {"state_label": 0, "length_samples": 8},
                  {"state_label": 1, "length_samples": 8}],
        }
        sequence_path = os.path.join(result_dir, "state_sequences.json")
        with open(sequence_path, "w", encoding="utf-8") as f:
            json.dump(sequences, f)
        artifacts["state_sequences"] = self.rel(context, sequence_path)
        context["manifest"].add_cluster_result(
            "kmeans_k2_merged", "merged/kmeans_k2_merged", artifacts)
        return context


class TransitionModelTest(unittest.TestCase):
    def test_markov_sampling_is_bounded_and_reproducible(self):
        sequences = [
            [{"state_label": 0, "length_samples": 5},
             {"state_label": 1, "length_samples": 7}],
            [{"state_label": 1, "length_samples": 6},
             {"state_label": 0, "length_samples": 8}],
        ]
        model = StateTransitionModel(sequences)
        a = model.sample_markov(np.random.default_rng(3), min_blocks=2, max_blocks=5)
        b = model.sample_markov(np.random.default_rng(3), min_blocks=2, max_blocks=5)
        self.assertEqual(a, b)
        self.assertGreaterEqual(len(a), 2)
        self.assertLessEqual(len(a), 5)
        self.assertTrue(all(length > 0 for _, length in a))


class PrimitiveSynthesisStepTest(unittest.TestCase):
    def test_real_resampling_emits_traceable_cycles(self):
        with tempfile.TemporaryDirectory() as tmp:
            cwd = os.getcwd()
            os.chdir(tmp)
            try:
                wf = Workflow("synth", "washing_machine", {"paths": {"cache_dir": ".cache"}})
                wf.add(SynthesisUpstreamStub())
                wf.add(PrimitiveSynthesisStep(
                    cluster_tag="kmeans_k2_merged", n_cycles=4,
                    random_seed=9, sequence_method="empirical", fs=1.0))
                wf.run()

                manifest = wf.manifest
                cycles_dir = manifest.artifact_path("primitive_synthesis", "cycles_dir")
                cycle_files = sorted(f for f in os.listdir(cycles_dir) if f.endswith(".csv"))
                self.assertEqual(len(cycle_files), 4)
                frame = pd.read_csv(os.path.join(cycles_dir, cycle_files[0]))
                self.assertEqual(list(frame.columns),
                                 ["sample_index", "time_seconds", "power",
                                  "state_label", "block_id"])
                self.assertEqual(len(frame), 24)
                self.assertTrue(set(frame["state_label"]).issubset({0, 1}))

                path = manifest.artifact_path("primitive_synthesis", "synthesis_manifest")
                with open(path, encoding="utf-8") as f:
                    records = json.load(f)
                self.assertEqual(len(records), 4)
                self.assertTrue(records[0]["blocks"][0]["sources"])
                self.assertGreater(records[0]["energy_wh"], 0.0)

                library_path = manifest.artifact_path("primitive_synthesis", "library_summary")
                with open(library_path, encoding="utf-8") as f:
                    summary = json.load(f)
                self.assertEqual(summary["0"]["count"], 3)
                self.assertEqual(summary["1"]["count"], 3)
                step = manifest.data["steps"]["primitive_synthesis"]
                self.assertIn("real_resample_empirical_on_kmeans_k2_merged",
                              step["subdir"])
            finally:
                os.chdir(cwd)


if __name__ == "__main__":
    unittest.main(verbosity=2)
