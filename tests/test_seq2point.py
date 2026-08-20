import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.train_nilm_seq2point import experiment_specs
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

    def test_experiment_matrix_contains_ten_runs(self):
        manifest = {"experiments": {
            ratio: {
                "A_real_only": ["a"],
                "B_real_plus_traditional": ["b"],
                "C_real_plus_generated": ["c"],
            } for ratio in ("05pct", "10pct", "20pct")
        }}
        manifest["experiments"]["full"] = {"D_full_real": ["d"]}
        specs = experiment_specs(manifest, "all")
        self.assertEqual(len(specs), 10)
        self.assertEqual(specs[-1][:2], ("full", "D"))


if __name__ == "__main__":
    unittest.main()
