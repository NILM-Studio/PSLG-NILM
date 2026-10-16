"""Run the restored discovery workflow on one day of real microwave data."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import pandas as pd
import yaml

root = Path(__file__).resolve().parents[1]
run_id = 'mainline_zhouslab_smoke_' + time.strftime('%Y%m%d_%H%M%S')
execution = Path(os.environ['PSLG_EXECUTION_DIR'])
work = execution / 'inputs_snapshot'
work.mkdir(parents=True, exist_ok=False)
source = root / 'datasets/self_collected/Microwave.csv'
pd.read_csv(source, nrows=86400).to_csv(work/'microwave_24h.csv', index=False)
cfg = yaml.safe_load((root/'config/config.yaml').read_text())
cfg['runtime'] = {'require_slurm': False}
cfg['extract_active_data'].update(threshold=100, t_drop=18, t_min_work=18,
    context_seconds=12, fs=1/6, resample_fs=1/6)
cfg['time_segmentation'].update(max_seg_len=64)
cfg['feature_extract'].update(epochs=1, latent_dim=4, batch_size=8, patience=1, cache=False)
cfg['temporal_state_merge'].update(fs=1/6, min_block_seconds=12)
config = work/'config.yaml'
config.write_text(yaml.safe_dump(cfg))
(execution/'config.resolved.json').write_text(json.dumps(cfg, indent=2))
(execution/'inputs.json').write_text(json.dumps({'source':str(source),'rows':86400,'snapshot':str(work/'microwave_24h.csv')}))
command = [sys.executable, str(root/'main.py'), '--profile', 'discovery',
    '--config', str(config), '--raw-series', str(work/'microwave_24h.csv'),
    '--run-id', run_id, '--appliance', 'microwave', '--segment-method', 'prim-glr',
    '--feature-model', 'cnn_ae', '--cluster-method', 'kmeans', '--n-clusters', '2',
    '--cluster-tag', 'kmeans_k2_merged',
    '--steps', 'extract,segment,feature,cluster,state_merge,state_sequence']
env = dict(os.environ, OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
    TF_NUM_INTRAOP_THREADS='2', TF_NUM_INTEROP_THREADS='2', TF_FORCE_GPU_ALLOW_GROWTH='true')
with (work/'workflow.log').open('w') as log:
    result = subprocess.run(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
print((work/'workflow.log').read_text()[-10000:])
if result.returncode:
    raise SystemExit(result.returncode)
manifest = execution/'run_manifest.json'
steps = json.loads(manifest.read_text())['steps']
expected = {'extract_active_data','time_segmentation','feature_extract','time_clustering','state_sequence'}
assert expected <= set(steps), steps.keys()
assert 'kmeans_k2_merged' in steps['time_clustering']['results']
print(json.dumps({'status':'pass','run_manifest':str(manifest),'steps':list(steps)}))
