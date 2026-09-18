import unittest
import numpy as np
from models.feature_extract.discovery_validation import temporal_split, training_minmax
from nilm_experiments.nilm_lab.activity_metrics import score_record
from nilm_experiments.nilm_lab.data import Record


class PrimitiveProtocolTests(unittest.TestCase):
    def test_temporal_activity_split_and_gap(self):
        aa = [dict(csv_idx=i, record_id='r', context_start=a, context_end_exclusive=b)
              for i, (a,b) in enumerate([(10,20),(7900,8100),(9500,9900)])]
        indices = np.array([[0,0],[0,3],[1,0],[2,0],[2,4]])
        split = temporal_split(indices, aa, [dict(id='r',start_timestamp=0,end_timestamp_exclusive=10000)],gap_seconds=1000)
        self.assertEqual(split['train_ids'], [0,1])
        self.assertEqual(split['val_ids'], [3,4])
        self.assertEqual(split['excluded_ids'], [2])

    def test_internal_normalization_never_fits_validation(self):
        x = np.array([[[1.],[2.],[3.]], [[100.],[200.],[300.]]])
        first, limits = training_minmax(x, [3,3], [0])
        x[1] *= 1000
        second, other = training_minmax(x, [3,3], [0])
        np.testing.assert_array_equal(limits, other)
        np.testing.assert_array_equal(first[0], second[0])

    def test_activity_mae_decomposition_and_standby(self):
        r = Record('r','test','1','val',np.arange(8)*6,np.ones(8),np.array([10,10,50,100,50,10,10,10.]),'x')
        ref = dict(labels=np.array([0,0,1,1,1,0,0,0]),valid=np.ones(8,bool),activities=[])
        s = score_record(r,np.zeros(8),None,ref,20,6)
        self.assertAlmostEqual(s['activity_support_MAE'],s['active_fraction']*s['active_MAE']+(1-s['active_fraction'])*s['inactive_MAE'])
        self.assertEqual(s['inactive_overprediction_wh'],0)
        self.assertGreater(s['inactive_true_wh'],0)


if __name__ == '__main__':
    unittest.main()
