import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.framework.run_manifest import RunManifest
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
            "1": {"payload": payload},
            "2": {"payload": {**payload, "appliance": np.asarray(
                [2, 3, 4, 11, 12, 13], dtype=np.float32)}},
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


class ContinuousDatasetIdentityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run_root = self.root / "log" / "run"
        self.cycle_root = self.run_root / "cycles"
        self.cycle_root.mkdir(parents=True)
        self.pair = self.root / "pair.csv"
        pd.DataFrame({"timestamp": np.arange(0, 180, 6),
                      "mains": np.full(30, 100),
                      "appliance": np.zeros(30)}).to_csv(self.pair, index=False)
        self.holdout = self.run_root / "assignments.csv"
        self.holdout.write_text(
            "activity_id,start_timestamp,end_timestamp,split\n"
            "0,0,12,train\n1,60,72,validation\n2,120,132,test\n", encoding="utf-8")
        summary = self.run_root / "holdout.json"
        summary.write_text('{"structure_fit_scope":"train_only"}', encoding="utf-8")
        for name in ("real.npz", "traditional.npz", "generated.npz"):
            np.savez(self.cycle_root / name, timestamp=np.array([0, 6, 12]),
                     mains=np.array([100, 120, 100]), appliance=np.array([0, 20, 0]))
        self.cycle = {
            "synthesis_scope": "budget_local", "nested_real_subsets": True,
            "budget_conditioning": {"method": "independent", "random_seed": 42},
            "mains_power_type": "active", "appliance_power_type": "active",
            "measurement_compatible_for_additive_synthesis": True,
            "measurement_audit": {"output_path": str(self.pair)},
            "budget_leakage_check": {"upstream_state_representation_scope": "not_verified"},
            "experiments": {
                "01pct": {
                    "real_ratio": 0.01, "selected_real_count": 1,
                    "selected_traditional_count": 1, "selected_generated_count": 1,
                    "synthesis_scope": "budget_local", "synthesis_fit_count": 1,
                    "synthesis_fit_activity_ids": ["0"],
                    "A_real_only": ["real.npz"],
                    "B_real_plus_traditional": ["real.npz", "traditional.npz"],
                    "C_real_plus_generated": ["real.npz", "generated.npz"],
                },
                "full": {"D_full_real": ["real.npz"]},
            },
        }
        self.cycle_path = self.cycle_root / "nilm_dataset_manifest.json"
        self._save_cycle()
        manifest = RunManifest(str(self.run_root / "run_manifest.json"))
        manifest.add_step("temporal_holdout", "stub", ".", {
            "assignments": "assignments.csv", "summary": "holdout.json"})
        manifest.add_step("cycle_split", "stub", ".", {},
                          extra={"structure_fit_scope": "train_only"})
        manifest.add_step("nilm_dataset", "stub", "cycles", {
            "dataset_manifest": "cycles/nilm_dataset_manifest.json"})
        self.context = {"manifest": manifest, "log_root": str(self.run_root)}

    def _save_cycle(self):
        self.cycle_path.write_text(json.dumps(self.cycle), encoding="utf-8")

    def _build(self, **overrides):
        options = {"min_off_samples": 2, "max_chunk_samples": 100,
                   "random_seed": 42, **overrides}
        step = NilmContinuousDatasetStep("kmeans_k2_merged", str(self.pair), **options)
        step.run(self.context)
        path = Path(self.context["manifest"].artifact_path(
            "nilm_continuous_dataset", "dataset_manifest"))
        return path, json.loads(path.read_text(encoding="utf-8"))

    def test_same_inputs_reuse_directory_and_copy_provenance(self):
        first, manifest = self._build()
        second, repeated = self._build()
        self.assertEqual(first, second)
        self.assertEqual(manifest, repeated)
        self.assertEqual(manifest["source_cycle_manifest_sha256"],
                         hashlib.sha256(self.cycle_path.read_bytes()).hexdigest())
        self.assertEqual(manifest["budget_conditioning"], self.cycle["budget_conditioning"])
        self.assertEqual(manifest["measurement_compatibility"],
                         {key: self.cycle[key] for key in (
                             "mains_power_type", "appliance_power_type",
                             "measurement_compatible_for_additive_synthesis")})
        self.assertEqual(manifest["measurement_audit"], self.cycle["measurement_audit"])
        self.assertEqual(manifest["upstream_state_representation_scope"], "not_verified")

    def test_methods_seeds_and_continuous_configuration_keep_separate_outputs(self):
        baseline_path, _ = self._build()
        baseline_bytes = baseline_path.read_bytes()
        self.cycle["budget_conditioning"]["method"] = "cycle_neighbors"
        self._save_cycle()
        conditioned_path, _ = self._build()
        self.assertNotEqual(conditioned_path, baseline_path)
        self.assertEqual(baseline_path.read_bytes(), baseline_bytes)
        changed_paths = {conditioned_path}
        for change in ({"random_seed": 7}, {"off_to_real_sample_ratio": 2},
                       {"max_chunk_samples": 20}, {"sample_period_seconds": 3},
                       {"max_gap_seconds": 30}, {"min_off_samples": 3},
                       {"active_threshold_watts": 5}):
            with self.subTest(change=change):
                path, _ = self._build(**change)
                self.assertNotIn(path, changed_paths)
                changed_paths.add(path)
        self.assertEqual(baseline_path.read_bytes(), baseline_bytes)

    def test_changed_actual_waveforms_or_holdout_change_directory(self):
        first, _ = self._build()
        np.savez(self.cycle_root / "generated.npz", timestamp=np.array([0, 6, 12]),
                 mains=np.array([100, 140, 100]), appliance=np.array([0, 40, 0]))
        changed_cycle, _ = self._build()
        self.assertNotEqual(first, changed_cycle)
        frame = pd.read_csv(self.pair)
        frame.loc[0, "mains"] = 125
        frame.to_csv(self.pair, index=False)
        changed_series, _ = self._build()
        self.assertNotEqual(changed_cycle, changed_series)
        self.holdout.write_text(self.holdout.read_text().replace("1,60,72", "1,66,72"),
                                encoding="utf-8")
        changed_holdout, _ = self._build()
        self.assertNotEqual(changed_series, changed_holdout)

    def test_missing_input_fails_before_creating_output_directory(self):
        (self.cycle_root / "generated.npz").unlink()
        with self.assertRaises(FileNotFoundError):
            self._build()
        self.assertEqual(list(self.run_root.glob("nilm_continuous_dataset_*")), [])

    def test_bounded_cohort_cannot_silently_extend_continuous_periods(self):
        for bound in ({"end_timestamp": 170}, {"start_timestamp": 0},
                      {"end": "1970-01-01T00:02:50Z"}):
            with self.subTest(bound=bound):
                (self.run_root / "holdout.json").write_text(json.dumps({
                    "structure_fit_scope": "train_only", "cohort": bound}), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "bounded temporal cohorts"):
                    self._build()
                self.assertEqual(list(self.run_root.glob("nilm_continuous_dataset_*")), [])

    def test_unbounded_cohort_metadata_remains_compatible(self):
        (self.run_root / "holdout.json").write_text(json.dumps({
            "structure_fit_scope": "train_only", "cohort": {
                "start": None, "end": None, "start_timestamp": None, "end_timestamp": None,
            }}), encoding="utf-8")
        path, _ = self._build()
        self.assertTrue(path.is_file())

    def test_unverified_structure_is_not_relabelled_train_only(self):
        self.context["manifest"].get_step("cycle_split")["extra"] = {}
        _, manifest = self._build(require_train_only_structure=False)
        self.assertEqual(manifest["structure_fit_scope"], "unknown")


class ContinuousSlurmResolutionTests(unittest.TestCase):
    def test_manifest_resolution_and_explicit_override(self):
        script = Path(__file__).resolve().parents[1] / "slurm/run_ukdale_seq2point_continuous.sh"
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            run_root = project / "log" / "test_run"
            discovered = run_root / "continuous_inputs_123"
            explicit = project / "explicit_dataset"
            for dataset in (discovered, explicit):
                dataset.mkdir(parents=True)
                (dataset / "nilm_dataset_manifest.json").write_text("{}", encoding="utf-8")
            (run_root / "run_manifest.json").write_text(json.dumps({"steps": {
                "nilm_continuous_dataset": {"artifacts": {
                    "dataset_manifest": "continuous_inputs_123/nilm_dataset_manifest.json"}},
            }}), encoding="utf-8")
            (project / "slurm").mkdir()
            (project / "slurm/run_ukdale_seq2point.sh").write_text(
                'printf "%s\\n" "$DATASET_DIR" "$OUTPUT_ROOT"\n', encoding="utf-8")
            env = {key: value for key, value in os.environ.items()
                   if key not in ("DATASET_DIR", "OUTPUT_ROOT")}
            env.update(PROJECT_DIR=str(project), RUN_ID="test_run", PYTHON_BIN=sys.executable)
            result = subprocess.run(["bash", str(script)], env=env,
                                    check=True, text=True, capture_output=True)
            self.assertEqual(result.stdout.splitlines(), [str(discovered.resolve()),
                str(run_root / "nilm_seq2point" / discovered.name)])
            env["DATASET_DIR"] = "explicit_dataset"
            (run_root / "run_manifest.json").unlink()
            result = subprocess.run(["bash", str(script)], env=env,
                                    check=True, text=True, capture_output=True)
            self.assertEqual(result.stdout.splitlines(), [str(explicit.resolve()),
                str(run_root / "nilm_seq2point" / explicit.name)])


if __name__ == "__main__":
    unittest.main()
