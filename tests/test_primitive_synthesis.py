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
from src.generation.cycle_patterns import CyclePatternCatalog, CyclePatternClassifier
from src.generation.cycle_validation import (discover_metric_modes,
                                             infer_cycle_grammar,
                                             robust_z_scores)
from src.generation.primitive_library import Primitive, PrimitiveLibrary, RealPrimitiveSampler
from src.generation.transition_model import StateTransitionModel
from src.steps.cycle_classification_step import CycleClassificationStep
from src.steps.cycle_split_step import CycleSplitStep
from src.steps.cycle_validation_step import CycleValidationStep
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


class CyclePatternClassifierTest(unittest.TestCase):
    @staticmethod
    def _blocks(labels):
        return [{"state_label": label, "length_samples": 10} for label in labels]

    def test_frequent_patterns_form_classes_and_distant_sequence_is_outlier(self):
        sequences = {
            "0": self._blocks([1, 0, 2, 1]),
            "1": self._blocks([1, 0, 2, 1]),
            "2": self._blocks([1, 0, 2, 1]),
            "3": self._blocks([1, 0, 1]),
            "4": self._blocks([1, 0, 1]),
            "5": self._blocks([3]),
        }
        result = CyclePatternClassifier(
            min_support=2, rare_max_distance=0.34).fit(sequences)
        self.assertEqual(result["n_classes"], 2)
        self.assertEqual(result["activities"]["5"]["class_id"], -1)
        self.assertEqual(result["n_outliers"], 1)

    def test_continuity_sampler_removes_artificial_join_jump(self):
        library = PrimitiveLibrary([
            Primitive(0, 0, 0, 0, np.array([0.0, 1.0], dtype=np.float32)),
            Primitive(1, 0, 1, 0, np.array([10.0, 11.0], dtype=np.float32)),
        ])
        sampler = RealPrimitiveSampler(
            library, candidate_pool=2, within_state_smooth_samples=2,
            boundary_smooth_samples=2)
        power, provenance = sampler.sample_block(
            0, 4, np.random.default_rng(2), initial_power=5.0)
        self.assertAlmostEqual(float(power[0]), 5.0)
        self.assertEqual(provenance[0]["join_jump_after"], 0.0)
        self.assertEqual(provenance[1]["join_jump_after"], 0.0)

    def test_cycle_grammar_is_inferred_from_supported_class_patterns(self):
        classes = [
            {"support": 80, "representative_signature": [1, 0, 2, 1]},
            {"support": 60, "representative_signature": [3, 0, 2, 1]},
            {"support": 10, "representative_signature": [3, 1, 3]},
        ]
        grammar = infer_cycle_grammar(classes, min_class_support=30,
                                      core_state_min_prevalence=0.8,
                                      terminal_state_min_prevalence=0.7)
        self.assertEqual(grammar["required_core_states"], [0, 1, 2])
        self.assertEqual(grammar["allowed_terminal_states"], [1])

    def test_robust_z_scores_flag_single_extreme_value(self):
        scores = robust_z_scores([10, 11, 10, 9, 200])
        self.assertTrue(np.all(scores[:4] < 3.5))
        self.assertGreater(scores[-1], 3.5)

    def test_metric_modes_preserve_supported_short_and_long_programs(self):
        rng = np.random.default_rng(7)
        short = np.column_stack((
            rng.normal(3900, 80, 40), rng.normal(500, 20, 40),
            rng.normal(450, 15, 40), rng.normal(2300, 10, 40)))
        long = np.column_stack((
            rng.normal(7800, 120, 40), rng.normal(1300, 30, 40),
            rng.normal(600, 20, 40), rng.normal(2320, 10, 40)))
        labels, diagnostics = discover_metric_modes(
            np.vstack((short, long)), max_modes=3, min_mode_support=15,
            bic_min_gain=10.0, random_state=3)
        self.assertEqual(diagnostics["selected_modes"], 2)
        self.assertEqual(set(labels[:40]), {0})
        self.assertEqual(set(labels[40:]), {1})


class PrimitiveSynthesisStepTest(unittest.TestCase):
    def test_cycle_split_counts_keep_training_member(self):
        self.assertEqual(CycleSplitStep._counts(1, 0.7, 0.1, 0.2), (1, 0, 0))
        self.assertEqual(CycleSplitStep._counts(2, 0.7, 0.1, 0.2), (1, 0, 1))
        self.assertEqual(CycleSplitStep._counts(11, 0.7, 0.1, 0.2), (8, 1, 2))

    def test_catalog_audit_uses_collapsed_canonical_signature(self):
        catalog = CyclePatternCatalog({
            "classes": [{
                "class_id": 0,
                "representative_signature": [1, 0, 1],
                "support": 1,
                "member_ids": ["0"],
            }],
            "activities": {"0": {
                "validation_mode_id": 2,
                "blocks": [
                    {"state_label": 1, "length_samples": 3},
                    {"state_label": 1, "length_samples": 4},
                    {"state_label": 0, "length_samples": 5},
                    {"state_label": 1, "length_samples": 6},
                ],
            }},
        })
        audit = PrimitiveSynthesisStep._audit_catalog(catalog, [0])
        self.assertEqual(
            audit["classes"]["0"]["representative_signature"], [1, 0, 1])
        self.assertEqual(audit["classes"]["0"]["modes"], {"2": 1})

    def test_real_resampling_emits_traceable_cycles(self):
        with tempfile.TemporaryDirectory() as tmp:
            cwd = os.getcwd()
            os.chdir(tmp)
            try:
                wf = Workflow("synth", "washing_machine", {"paths": {"cache_dir": ".cache"}})
                wf.add(SynthesisUpstreamStub())
                wf.add(CycleClassificationStep(
                    cluster_tag="kmeans_k2_merged", min_support=1))
                wf.add(CycleValidationStep(
                    cluster_tag="kmeans_k2_merged", fs=1.0,
                    min_class_support=1, min_signature_purity=0.0,
                    min_valid_member_ratio=0.0,
                    terminal_state_min_prevalence=0.5,
                    min_duration_seconds=0.0,
                    boundary_absolute_watts=200.0,
                    robust_z_threshold=float("inf")))
                wf.add(CycleSplitStep(
                    cluster_tag="kmeans_k2_merged", train_ratio=1.0,
                    validation_ratio=0.0, test_ratio=0.0))
                wf.add(PrimitiveSynthesisStep(
                    cluster_tag="kmeans_k2_merged", n_cycles=4,
                    random_seed=9, sequence_method="empirical", fs=1.0,
                    class_sampling="balanced_pairs",
                    require_cycle_split=True))
                wf.run()

                manifest = wf.manifest
                whitelist = manifest.artifact_path("cycle_validation", "whitelist")
                with open(whitelist, encoding="utf-8") as f:
                    self.assertEqual(json.load(f)["valid_class_ids"], [0, 1])
                mode_summary = manifest.artifact_path(
                    "cycle_validation", "mode_summary")
                self.assertTrue(os.path.exists(mode_summary))
                split_summary = manifest.artifact_path("cycle_split", "summary")
                with open(split_summary, encoding="utf-8") as f:
                    self.assertEqual(
                        json.load(f)["counts"],
                        {"train": 2, "validation": 0, "test": 0})
                cycles_dir = manifest.artifact_path("primitive_synthesis", "cycles_dir")
                cycle_files = sorted(f for f in os.listdir(cycles_dir) if f.endswith(".csv"))
                self.assertEqual(len(cycle_files), 4)
                frame = pd.read_csv(os.path.join(cycles_dir, cycle_files[0]))
                self.assertEqual(list(frame.columns),
                                 ["sample_index", "time_seconds", "power",
                                  "state_label", "block_id", "cycle_class",
                                  "cycle_mode",
                                  "source_activity_id"])
                self.assertEqual(len(frame), 24)
                self.assertTrue(set(frame["state_label"]).issubset({0, 1}))

                path = manifest.artifact_path("primitive_synthesis", "synthesis_manifest")
                with open(path, encoding="utf-8") as f:
                    records = json.load(f)
                self.assertEqual(len(records), 4)
                self.assertEqual(
                    {(row["cycle_class"], row["cycle_mode"])
                     for row in records}, {(0, 0), (1, 0)})
                self.assertEqual(
                    [row["cycle_class"] for row in records], [0, 1, 0, 1])
                self.assertTrue(records[0]["blocks"][0]["sources"])
                self.assertGreater(records[0]["energy_wh"], 0.0)

                library_path = manifest.artifact_path("primitive_synthesis", "library_summary")
                with open(library_path, encoding="utf-8") as f:
                    summary = json.load(f)
                self.assertEqual(summary["class_0_mode_0"]["0"]["count"], 2)
                self.assertEqual(summary["class_0_mode_0"]["1"]["count"], 1)
                self.assertEqual(summary["class_1_mode_0"]["0"]["count"], 1)
                self.assertEqual(summary["class_1_mode_0"]["1"]["count"], 2)
                continuity_path = manifest.artifact_path(
                    "primitive_synthesis", "continuity_metrics")
                with open(continuity_path, encoding="utf-8") as f:
                    continuity = json.load(f)
                self.assertEqual(
                    continuity["state_boundary"]["after"]["max"], 0.0)
                step = manifest.data["steps"]["primitive_synthesis"]
                self.assertIn(
                    "real_resample_empirical_all_train_split_on_kmeans_k2_merged",
                              step["subdir"])
                audit_path = manifest.artifact_path(
                    "primitive_synthesis", "input_audit")
                with open(audit_path, encoding="utf-8") as f:
                    audit = json.load(f)
                self.assertEqual(audit["valid_class_ids"], [0, 1])
                self.assertEqual(
                    audit["classes"]["0"]["representative_signature"],
                    [0, 1, 0])
            finally:
                os.chdir(cwd)


if __name__ == "__main__":
    unittest.main(verbosity=2)
