"""Synthetic source/test audit with real subprocess trainer and artifact handoff."""
from pathlib import Path
import json
import os
import subprocess
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd
import yaml
from nilm_experiments.nilm_lab.common import digest, write_json, read_json

PROJECT = Path(__file__).resolve().parents[1]


class WorkflowIntegrationTests(unittest.TestCase):
    def test_training_selection_explicit_evaluation_and_dictionary_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            torch_python = os.environ.get('NILM_TORCH_PYTHON', sys.executable)
            cfg = yaml.safe_load((PROJECT / 'config/config_nilm_sequence_pilot.yaml').read_text())
            cfg['runtime'] = dict(require_slurm=False, torch_python=torch_python)
            cfg['data_protocol']['manifest'] = str(root / 'split.json')
            cfg['nilm'].update(k=2, window_length=32, train_stride=16, eval_stride=16,
                models=['FCN'], cases=[dict(name='P_gru', arm='P', activity_head='gru', weights=[[.01, .01]])])
            cfg['nilm']['training'].update(device='cpu', epochs=1, patience=1, batch_size=8)
            cfg['extract_active_data'].update(threshold=20, t_drop=18, t_min_work=18, context_seconds=12)
            (root / 'config.yaml').write_text(yaml.safe_dump(cfg), encoding='utf-8')
            t = np.arange(160)*6 + 1600000002
            y = np.zeros(160, np.float32)
            y[20:40] = 50
            y[60:85] = 150
            entries = []
            for i, split in enumerate(('train', 'val', 'test')):
                path = root / f'{split}.csv'
                entries.append(dict(id=split, dataset='fixture', house=str(i), split=split,
                    path=str(path), native_seconds=6))
                if split != 'test':
                    pd.DataFrame(dict(timestamp=t+i*6000, mains=y+100+i, target=y)).to_csv(path, index=False)
            write_json(root / 'split.json', dict(protocol='cross_house', sample_seconds=6,
                min_coverage=1, records=entries, missing_policy='strict'))
            env = os.environ.copy()
            env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')

            def run(steps, success=True):
                proc = subprocess.run([sys.executable, str(PROJECT / 'main.py'), '--config', str(root / 'config.yaml'),
                    '--profile', 'nilm', '--run-id', 'fixture', '--steps', steps], cwd=root,
                    env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                if success:
                    self.assertEqual(proc.returncode, 0, proc.stdout[-10000:])
                else:
                    self.assertNotEqual(proc.returncode, 0, proc.stdout)
                return proc.stdout

            run('nilm_data,extract')
            log = root / 'log/fixture'
            manifest = read_json(log / 'run_manifest.json')
            ap = log / manifest['steps']['extract_active_data']['artifacts']['activities']
            am = read_json(ap)
            source = read_json(log / 'nilm_data/split_manifest.json')
            with np.load(log / 'nilm_data' / source['records'][0]['file']) as z:
                tt = z['timestamp']
            mapping_dir = log / 'fixture_state_sequence'
            mapping_dir.mkdir()
            state = np.zeros(160, np.int64)
            state[50:] = 1
            np.savez_compressed(mapping_dir / 'mapping.npz', timestamp=tt, state_full_merge=state,
                block_id=state, context_conflict=np.zeros(160, bool))
            mp = mapping_dir / 'mapping.json'
            write_json(mp, dict(source_only=True, k=2, dictionary_id='synthetic_teacher_v1',
                input_hashes={str(ap): digest(ap)}, records=[dict(record_id='train', file='mapping.npz',
                                                               sha256=digest(mapping_dir / 'mapping.npz'))]))
            manifest['steps']['state_sequence'] = dict(artifacts=dict(mapping='fixture_state_sequence/mapping.json'))
            write_json(log / 'run_manifest.json', manifest)
            run('nilm_labels,nilm_train,nilm_select,nilm_report')
            # The test CSV does not even exist during fitting and validation selection.
            self.assertFalse((root / 'test.csv').exists())
            selection = read_json(log / 'nilm_select/selection.json')
            self.assertFalse(selection['test_opened'])
            self.assertEqual(len(selection['runs']), 1)
            checkpoint = Path(selection['runs'][0]['path']) / 'best.pt'
            original_mtime = checkpoint.stat().st_mtime_ns
            run('nilm_train')
            self.assertEqual(checkpoint.stat().st_mtime_ns, original_mtime)
            pd.DataFrame(dict(timestamp=t+12000, mains=y+103, target=y)).to_csv(root / 'test.csv', index=False)
            run('nilm_evaluate,nilm_report')
            predictions = read_json(log / 'nilm_evaluate/prediction_manifest.json')
            self.assertEqual(predictions[0]['record_id'], 'test')
            with np.load(log / 'nilm_evaluate' / predictions[0]['file']) as pred:
                self.assertEqual(pred['state_probabilities'].shape, (3, 160))
                self.assertTrue(np.isfinite(pred['activity']).all())
            run('nilm_evaluate', success=False)
            changed = read_json(mp)
            changed['dictionary_id'] = 'different_teacher'
            write_json(mp, changed)
            output = run('nilm_train', success=False)
            self.assertIn('Dictionary changed', output)


if __name__ == '__main__':
    unittest.main()
