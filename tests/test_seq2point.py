import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from scripts.train_nilm_seq2point import chunk_energy_metrics, experiment_specs, train_one
from src.nilm import experiment_identity
from src.nilm.seq2point import CycleWindowCorpus, regression_metrics


class Seq2PointTests(unittest.TestCase):
    def test_cycle_window_corpus_pads_and_targets_midpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.savez(root / "cycle.npz",
                     mains=np.array([100, 200, 300], dtype=np.float32),
                     appliance=np.array([0, 20, 40], dtype=np.float32))
            corpus = CycleWindowCorpus(
                root, ["cycle.npz"], window_length=3, stride=1,
                mains_scale=1000, appliance_scale=100)
            x, y = corpus.batch(np.array([0, 1, 2]))
            self.assertEqual(x.shape, (3, 3, 1))
            self.assertTrue(np.allclose(x[0, :, 0], [0, 0.1, 0.2]))
            self.assertTrue(np.allclose(y[:, 0], [0, 0.2, 0.4]))

    def test_metrics_have_expected_perfect_values(self):
        target = np.array([0, 30, 50], dtype=np.float32)
        metrics = regression_metrics(target, target, on_threshold=20)
        self.assertEqual(metrics["mae_watts"], 0.0)
        self.assertEqual(metrics["sae"], 0.0)
        self.assertEqual(metrics["nde"], 0.0)
        self.assertEqual(metrics["f1"], 1.0)
        self.assertEqual(metrics["false_positive_rate"], 0.0)
        self.assertEqual(metrics["prediction_mean_watts"],
                         metrics["target_mean_watts"])
        self.assertEqual(metrics["prediction_on_fraction"],
                         metrics["target_on_fraction"])

    def test_chunk_energy_metrics_are_zero_for_perfect_prediction(self):
        values = np.asarray([0, 10, 20, 30], dtype=np.float32)
        metrics = chunk_energy_metrics(values, values, [2, 2])
        self.assertEqual(metrics["mean_active_chunk_energy_relative_error"], 0.0)

    def test_zero_energy_chunks_report_false_power_without_dividing_by_zero(self):
        target = np.asarray([0, 0, 10, 20], dtype=np.float32)
        prediction = np.asarray([5, 7, 10, 20], dtype=np.float32)
        metrics = chunk_energy_metrics(target, prediction, [2, 2])
        self.assertEqual(metrics["active_energy_chunk_count"], 1)
        self.assertEqual(metrics["zero_energy_chunk_count"], 1)
        self.assertEqual(
            metrics["mean_prediction_watts_on_zero_energy_chunks"], 6.0)
        self.assertEqual(metrics["mean_active_chunk_energy_relative_error"], 0.0)

    def test_experiment_matrix_contains_ten_runs(self):
        manifest = {"experiments": {
            ratio: {
                "real_ratio": int(ratio[:2]) / 100,
                "A_real_only": ["a"],
                "B_real_plus_traditional": ["b"],
                "C_real_plus_generated": ["c"],
            } for ratio in ("05pct", "10pct", "20pct")
        }}
        manifest["experiments"]["full"] = {"D_full_real": ["d"]}
        specs = experiment_specs(manifest, "all")
        self.assertEqual(len(specs), 10)
        self.assertEqual(specs[-1][:2], ("full", "D"))

    def test_experiment_matrix_discovers_five_ratios(self):
        manifest = {"experiments": {
            ratio: {
                "real_ratio": value,
                "A_real_only": ["a"],
                "B_real_plus_traditional": ["b"],
                "C_real_plus_generated": ["c"],
            } for ratio, value in (("01pct", 0.01), ("02pct", 0.02),
                                   ("05pct", 0.05), ("10pct", 0.1),
                                   ("20pct", 0.2))
        }}
        manifest["experiments"]["full"] = {"D_full_real": ["d"]}
        specs = experiment_specs(manifest, "all")
        self.assertEqual(len(specs), 16)
        self.assertEqual([row[0] for row in specs[:6]],
                         ["01pct"] * 3 + ["02pct"] * 3)


class ExperimentIdentityTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.files = [f"{split}.npz" for split in ("train", "validation", "test")]
        for filename in self.files:
            self._write_pair(filename, 20)
        self.manifest = {"experiments": {
            "01pct": {"A_real_only": ["train.npz"], "real_ratio": 0.01},
            "full": {"validation": ["validation.npz"], "test": ["test.npz"]},
        }}
        self._write_manifest()
        self.args = SimpleNamespace(
            seed=42, epochs=30, patience=5, batch_size=64, window_length=599,
            train_stride=5, validation_stride=5, test_stride=1,
            learning_rate=0.001, dropout=0.0, mains_scale=10000.0,
            appliance_scale=4000.0, on_threshold=20.0,
            output_root=str(self.root / "results"), force=False,
        )
        self.output = Path(self.args.output_root) / "01pct_A_seed42"
        self.output.mkdir(parents=True)

    def _write_pair(self, name, power):
        np.savez(self.root / name, timestamp=np.array([0, 6]),
                 mains=np.array([100, 200]), appliance=np.array([0, power]))

    def _write_manifest(self):
        (self.root / "nilm_dataset_manifest.json").write_text(
            json.dumps(self.manifest), encoding="utf-8")

    def _identity(self, files=None):
        return experiment_identity.build_experiment_identity(
            self.args, self.root, self.manifest, "01pct", "A",
            ["train.npz"] if files is None else files)

    def _save_completed(self, identity):
        metrics = {"mae_watts": 1.0, "experiment_fingerprint": identity["sha256"]}
        (self.output / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
        (self.output / "experiment_identity.json").write_text(
            json.dumps(identity), encoding="utf-8")
        return metrics

    def test_verified_cache_returns_before_tensorflow_or_corpus_loading(self):
        identity = self._identity()
        self.assertEqual(self._identity(), identity)
        metrics = self._save_completed(identity)
        with mock.patch.dict(sys.modules, {"tensorflow": None}), mock.patch(
                "scripts.train_nilm_seq2point.CycleWindowCorpus",
                side_effect=AssertionError("cached result must not load corpus")):
            self.assertEqual(train_one(
                self.args, self.root, self.manifest, "01pct", "A", ["train.npz"]),
                metrics)

    def test_replacing_any_split_file_invalidates_cache_even_with_same_mtime(self):
        for filename in self.files:
            with self.subTest(filename=filename):
                self._save_completed(self._identity())
                previous = (self.root / filename).stat()
                self._write_pair(filename, 99)
                os.utime(self.root / filename,
                         ns=(previous.st_atime_ns, previous.st_mtime_ns))
                with mock.patch.dict(sys.modules, {"tensorflow": None}):
                    with self.assertRaisesRegex(ValueError, "refusing stale result.*--output-root"):
                        train_one(self.args, self.root, self.manifest,
                                  "01pct", "A", ["train.npz"])

    def test_manifest_parameters_and_repeated_references_change_identity(self):
        original = self._identity()
        self._save_completed(original)
        self.manifest["sample_period_seconds"] = 6
        self._write_manifest()
        changed = self._identity()
        with self.assertRaisesRegex(ValueError, "refusing stale"):
            experiment_identity.reusable_metrics(self.output, changed)
        for name, value in (("epochs", 3), ("test_stride", 5),
                            ("mains_scale", 5000.0), ("learning_rate", 0.0001)):
            with self.subTest(parameter=name):
                baseline = self._identity()
                previous = getattr(self.args, name)
                setattr(self.args, name, value)
                self.assertNotEqual(self._identity()["sha256"], baseline["sha256"])
                setattr(self.args, name, previous)
        self.assertNotEqual(self._identity()["sha256"],
                            self._identity(["train.npz", "train.npz"])["sha256"])

    def test_changed_model_source_rejects_cache(self):
        code = self.root / "model_source.py"
        code.write_text("architecture = 1\n", encoding="utf-8")
        with mock.patch.object(experiment_identity, "CODE_FILES", (str(code),)):
            self._save_completed(self._identity())
            code.write_text("architecture = 2\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "refusing stale"):
                experiment_identity.reusable_metrics(self.output, self._identity())

    def test_legacy_result_requires_explicit_retrain_or_new_output(self):
        (self.output / "metrics.json").write_text('{"mae_watts": 1}', encoding="utf-8")
        identity = self._identity()
        with self.assertRaisesRegex(ValueError, "missing or invalid.*--force"):
            experiment_identity.reusable_metrics(self.output, identity)
        self.assertIsNone(experiment_identity.reusable_metrics(self.output, identity, force=True))

    def test_manifest_changed_since_selection_is_rejected(self):
        (self.root / "nilm_dataset_manifest.json").write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "manifest changed after loading"):
            self._identity()


if __name__ == "__main__":
    unittest.main()
