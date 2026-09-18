import unittest
import numpy as np
from nilm_experiments.nilm_lab.activity_labels import detect_activities, activity_projection
from nilm_experiments.nilm_lab.sequence_data import fit_label_bundles
from nilm_experiments.nilm_lab.data import Record, Windows, validate_manifest
from src.utils.state_activity_mapping import contiguous_groups, map_blocks, sequence_statistics
import main


class SequenceLabelTests(unittest.TestCase):
    def setUp(self):
        self.cfg = dict(method='simple', threshold=20, t_drop=18, t_min_work=18,
                        context_seconds=12, fs=1/6, resample_fs=1/6)
        self.t = np.arange(24)*6
        self.y = np.zeros(24, dtype=np.float32)
        self.y[5:12] = [30, 30, 2, 2, 70, 70, 70]

    def test_original_activity_internal_low_and_half_open(self):
        aa, intervals = detect_activities(self.t, self.y, self.cfg, 'r', 6)
        from models.extract_active_data.simple_threshold import SimpleThresholdDetector
        original = SimpleThresholdDetector('parity', self.cfg).detect(self.y, self.t)
        self.assertEqual([(a['core_start'], a['core_end_exclusive']-6) for a in aa],
                         [(a['start_time'], a['end_time']) for a in original])
        z, _, valid = activity_projection(len(self.t), aa, np.isfinite(self.y))
        np.testing.assert_array_equal(z[5:12], np.ones(7))
        self.assertEqual(z[4], 0)
        self.assertEqual(z[12], 0)

    def test_missing_activity_points_excluded(self):
        self.y[7] = np.nan
        aa, _ = detect_activities(self.t, self.y, self.cfg, 'r', 6)
        z, _, valid = activity_projection(24, aa, np.isfinite(self.y))
        self.assertFalse(valid[7])
        self.assertEqual(z[7], -100)
        self.assertEqual(z[8], 1)

    def test_merge_groups_never_bridge_holes(self):
        self.assertEqual(contiguous_groups([2, 0, 1], [0, 3, 8], [3, 3, 2]), [[0, 1], [2]])
        with self.assertRaises(ValueError):
            contiguous_groups([0, 1], [0, 2], [3, 3])

    def test_common_control_masks_and_low_power_label(self):
        aa, _ = detect_activities(self.t, self.y, self.cfg, 'r', 6)
        r = Record('r', 'fixture', '1', 'train', self.t, np.ones(24)*100, self.y, 'fingerprint')
        states = np.zeros(24, dtype=int)
        states[9:12] = 1
        bid = np.zeros(24, dtype=int)
        bid[9:12] = 1
        mappings = dict(r=dict(state_full_merge=states, block_id=bid, context_conflict=np.zeros(24, bool)))
        zz, _ = fit_label_bundles([r], {'r': aa}, mappings, 2)
        z = zz['r']
        self.assertGreater(z['primitive'][7], 0)
        self.assertEqual(z['primitive'][4], 0)
        for key in ('primitive', 'bins', 'segment_bins'):
            np.testing.assert_array_equal(z[key] >= 0, z['valid'])
        r.split = 'test'
        with self.assertRaises(ValueError):
            fit_label_bundles([r], {'r': aa}, mappings, 2)

    def test_windows_independent_of_target_missingness(self):
        r = Record('r', 'fixture', '1', 'val', self.t, np.ones(24), self.y.copy(), 'fp')
        before = Windows([r], 8, 4, 100).index
        r.y[:] = np.nan
        after = Windows([r], 8, 4, 100)
        self.assertEqual(before, after.index)
        self.assertFalse(after[0]['observed'].any())

    def test_onoff_mask_does_not_require_primitive_coverage(self):
        r = Record('r', 'fixture', '1', 'train', self.t, np.ones(24), self.y, 'fp')
        bundle = dict(onoff=np.ones(24, np.int64), primitive=np.ones(24, np.int64),
                      valid=np.zeros(24, bool), activity_valid=np.ones(24, bool))
        o = Windows([r], 8, 4, 100, {'r': bundle}, 'O')[0]
        p = Windows([r], 8, 4, 100, {'r': bundle}, 'P')[0]
        self.assertTrue((o['z'] == 1).all())
        self.assertTrue((p['z'] == -100).all())

    def test_no_implicit_test_or_removed_entry(self):
        self.assertNotIn('nilm_evaluate', main.parse_steps('all', main.IMPLEMENTED_STEPS))
        self.assertEqual(main.parse_steps('nilm_evaluate', main.IMPLEMENTED_STEPS), ['nilm_evaluate'])
        for name in ('fewshot', 'pam', 'split'):
            with self.assertRaisesRegex(ValueError, 'Removed workflow'):
                main.parse_steps(name, main.IMPLEMENTED_STEPS)

    def test_household_leakage_rejected(self):
        m = dict(protocol='cross_house', records=[dict(id=str(i), dataset='d', house=str(h), split=s, path=str(i))
            for i, (h, s) in enumerate([(1, 'train'), (1, 'val'), (2, 'test')])])
        with self.assertRaisesRegex(ValueError, 'Household leakage'):
            validate_manifest(m)

    def test_conflicting_context_order_invariance(self):
        aa, _ = detect_activities(self.t, self.y, self.cfg, 'r', 6)
        a = dict(aa[0], csv_idx=0)
        b = dict(a, csv_idx=1, activity_id=1)
        blocks = [dict(csv_idx=i, block_id=i, start=0, end=8, length_samples=8, state_label=i, member_rows=[i]) for i in (0, 1)]
        x, _ = map_blocks(self.t, [a, b], blocks, 6)
        y, _ = map_blocks(self.t, [a, b], blocks[::-1], 6)
        np.testing.assert_array_equal(x['state_full_merge'], y['state_full_merge'])
        self.assertTrue(x['context_conflict'].any())

    def test_block_transitions_do_not_count_point_repeats_or_missing(self):
        seq = [dict(activity_id=0, start=a, end=b, state_label=i, left_censored=False, right_censored=False,
                    duration_seconds=(b-a)*6) for a,b,i in [(0, 3, 0), (3, 9, 1), (10, 12, 0)]]
        trans, durations = sequence_statistics(seq, np.ones(12, bool), np.zeros(12, bool), 2, 6, 8)
        self.assertEqual(trans['counts'], [[0, 1], [0, 0]])

    def test_report_pairs_seeds_within_houses(self):
        from src.utils.nilm_report_summary import summarize_test
        results = [dict(trial=dict(model='FCN', case=case, seed=seed),
                        scores=[dict(dataset='fixture', house=str(h), mae=10.+seed+h-offset) for h in (1,2)])
                   for case, offset in [('B', 0), ('P', 2)] for seed in (0,1)]
        summary = summarize_test(results, 'fixture_appliance')
        pair = summary['paired_test'][0]
        self.assertEqual(pair['households'], 2)
        self.assertEqual(pair['mae_improvement_w'], 2)
        self.assertIsNone(pair['ci95'])


if __name__ == '__main__':
    unittest.main()
