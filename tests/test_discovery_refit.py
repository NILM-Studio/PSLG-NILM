import unittest
import tempfile
import numpy as np


class DiscoveryRefitTests(unittest.TestCase):
    def test_internal_validation_then_fresh_full_source_fit(self):
        from models.feature_extract.detsec_pc import detsec_pc
        x = np.random.RandomState(4).uniform(0,10,(10,16,4)).astype(np.float32)
        with tempfile.TemporaryDirectory() as out:
            z,h = detsec_pc(x,dict(lengths=np.full(10,16),epochs=2,patience=2,batch_size=4,
                norm_mode='minmax',artifact_dir=out,random_state=0,embed_dim=4,
                validation_spec=dict(train_ids=list(range(7)),val_ids=[7,8,9]),
                tf_schedule='linear',tf_ratio=0.))
            self.assertTrue(np.isfinite(z).all())
            self.assertTrue(h['refit_all_source'])
            self.assertEqual(h['schedule_epochs'],2)
            self.assertIn('validation_z_only_loss',h['internal_validation'])


if __name__ == '__main__':
    unittest.main()
