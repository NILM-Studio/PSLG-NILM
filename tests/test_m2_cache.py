"""M2 tests: content-addressed feature-extract cache.

Covers: key stability/sensitivity, store/load roundtrip, and end-to-end
behavior — two Workflow runs with DIFFERENT run-ids must train only once and
record cache_hit correctly in the manifest. Uses a stubbed compute so no
TF/sklearn is needed.
"""
import json
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.framework import feature_cache
from src.framework.workflow import Workflow
from src.steps.feature_extract_step import FeatureExtractStep


X0 = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
CFG = {"latent_dim": 16, "epochs": 50, "batch_size": 32,
       "learning_rate": 0.0001, "patience": 5}


class TestKey(unittest.TestCase):
    def test_stable(self):
        k1 = feature_cache.compute_key(X0, None, "detsec", CFG)
        k2 = feature_cache.compute_key(X0.copy(), None, "detsec", dict(CFG))
        self.assertEqual(k1, k2)
        self.assertEqual(len(k1), 64)  # sha256 hex

    def test_sensitive_to_inputs(self):
        base = feature_cache.compute_key(X0, None, "detsec", CFG)
        cases = {
            "hyperparam": dict(x=X0, lengths=None, model="detsec", cfg={**CFG, "epochs": 51}),
            "model": dict(x=X0, lengths=None, model="bilstm_ae", cfg=CFG),
            "X content": dict(x=X0 + 1.0, lengths=None, model="detsec", cfg=CFG),
            "X shape": dict(x=X0.reshape(4, 3, 2), lengths=None, model="detsec", cfg=CFG),
            "lengths": dict(x=X0, lengths=np.array([3, 3]), model="detsec", cfg=CFG),
        }
        for name, kw in cases.items():
            with self.subTest(case=name):
                self.assertNotEqual(
                    base, feature_cache.compute_key(kw["x"], kw["lengths"], kw["model"], kw["cfg"]))

    def test_upstream_choices_are_covered_transitively(self):
        """Requirement: the feature-extract index must account for the upstream
        choices (dataset, segmentation method). Those influence features ONLY
        through the tensor X/lengths — which is hashed byte-for-byte — so any
        upstream change that matters changes the key automatically."""
        key = feature_cache.compute_key(X0, None, "detsec", CFG)

        # different dataset (e.g. fridge vs kettle) -> different raw series
        # -> different segmentation tensor -> different key
        x_other_appliance = X0 * 2.0
        self.assertNotEqual(key, feature_cache.compute_key(
            x_other_appliance, None, "detsec", CFG))

        # different segmentation method (clasp vs fluss) -> different tensor
        # -> different key
        x_other_segment = X0.copy()
        x_other_segment[0, 0, 0] += 1e-3
        self.assertNotEqual(key, feature_cache.compute_key(
            x_other_segment, None, "detsec", CFG))

        # different segmentation HYPERPARAMS (window_size etc.) -> different
        # tensor shape/content -> different key
        self.assertNotEqual(key, feature_cache.compute_key(
            X0.reshape(2, 4, 3), None, "detsec", CFG))

    def test_identical_tensor_shares_cache_regardless_of_provenance(self):
        """And the converse, by design: if two upstream paths produce a
        byte-identical tensor, the features would be identical, so they MUST
        share one cache entry. Provenance lives in meta.json, not the key."""
        k1 = feature_cache.compute_key(X0, None, "detsec", CFG)
        k2 = feature_cache.compute_key(X0.copy(), None, "detsec", CFG)
        self.assertEqual(k1, k2)

    def test_code_change_invalidates(self):
        k1 = feature_cache.compute_key(X0, None, "detsec", CFG, code_id="aaa")
        k2 = feature_cache.compute_key(X0, None, "detsec", CFG, code_id="bbb")
        self.assertNotEqual(k1, k2)

    def test_real_model_files_fingerprint(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        fp = feature_cache.file_fingerprint(
            os.path.join(root, "models", "feature_extract", "detsec_model.py"))
        self.assertEqual(len(fp), 64)  # found and hashed, not "missing:..."


class TestStoreLoad(unittest.TestCase):
    def test_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            feats = np.random.RandomState(0).randn(5, 16).astype(np.float32)
            hist = {"loss": [1.0, 0.5], "val_loss": [1.1, 0.6]}
            key = feature_cache.compute_key(X0, None, "detsec", CFG)
            feature_cache.store(td, key, feats, hist, meta={"model": "detsec"})

            got = feature_cache.load(td, key)
            self.assertIsNotNone(got)
            got_feats, got_hist = got
            np.testing.assert_array_equal(got_feats, feats)
            self.assertEqual(got_hist, hist)
            # meta is provenance for humans; present but unused for lookup
            with open(os.path.join(td, "features", key, "meta.json")) as f:
                self.assertEqual(json.load(f)["model"], "detsec")

    def test_miss(self):
        with tempfile.TemporaryDirectory() as td:
            self.assertIsNone(feature_cache.load(td, "0" * 64))


class StubFeatureStep(FeatureExtractStep):
    """Fixed input + call-counting compute; no TF needed."""
    calls = 0

    def _load_input(self, context):
        return X0, None

    def _compute_features(self, np_data, lengths):
        type(self).calls += 1
        return np_data.mean(axis=2), {"loss": [1.0, 0.5]}


class TestStepCaching(unittest.TestCase):
    def test_second_run_hits_cache_across_run_ids(self):
        StubFeatureStep.calls = 0
        with tempfile.TemporaryDirectory() as td:
            cwd = os.getcwd()
            os.chdir(td)
            try:
                cfg = {"paths": {"cache_dir": ".cache"},
                       "feature_extract": {}}

                wf1 = Workflow("run_a", "fridge", cfg)
                wf1.add(StubFeatureStep(model_name="detsec", segment_method="clasp"))
                wf1.run()
                self.assertEqual(StubFeatureStep.calls, 1)
                extra1 = wf1.manifest.data["steps"]["feature_extract"]["extra"]
                self.assertFalse(extra1["cache_hit"])
                self.assertTrue(extra1["cache_key"])

                # A completely separate run (new run-id, new manifest) with the
                # SAME inputs must reuse the cache — that's the whole point.
                wf2 = Workflow("run_b", "fridge", cfg)
                wf2.add(StubFeatureStep(model_name="detsec", segment_method="clasp"))
                ctx2 = wf2.run()
                self.assertEqual(StubFeatureStep.calls, 1)  # no re-train
                extra2 = wf2.manifest.data["steps"]["feature_extract"]["extra"]
                self.assertTrue(extra2["cache_hit"])
                self.assertEqual(extra1["cache_key"], extra2["cache_key"])

                # Each run still gets its own features.npy + manifest path.
                p2 = wf2.manifest.artifact_path("feature_extract", "features")
                self.assertTrue(os.path.exists(p2))
                np.testing.assert_array_equal(
                    np.load(p2), ctx2["data"]["features"])

                # Changing a hyperparameter must miss and re-train.
                wf3 = Workflow("run_c", "fridge", cfg)
                wf3.add(StubFeatureStep(model_name="detsec", segment_method="clasp", epochs=51))
                wf3.run()
                self.assertEqual(StubFeatureStep.calls, 2)
                extra3 = wf3.manifest.data["steps"]["feature_extract"]["extra"]
                self.assertFalse(extra3["cache_hit"])
                self.assertNotEqual(extra1["cache_key"], extra3["cache_key"])
            finally:
                os.chdir(cwd)

    def test_cache_disabled_always_trains(self):
        StubFeatureStep.calls = 0
        with tempfile.TemporaryDirectory() as td:
            cwd = os.getcwd()
            os.chdir(td)
            try:
                cfg = {"paths": {"cache_dir": ".cache"}, "feature_extract": {}}
                for rid in ("r1", "r2"):
                    wf = Workflow(rid, "fridge", cfg)
                    wf.add(StubFeatureStep(model_name="detsec", segment_method="clasp",
                                           cache_enabled=False))
                    wf.run()
                self.assertEqual(StubFeatureStep.calls, 2)
            finally:
                os.chdir(cwd)


if __name__ == "__main__":
    unittest.main(verbosity=2)
