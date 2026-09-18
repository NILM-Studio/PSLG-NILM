"""Read-only source audit: frozen discovery restoration and simple baselines."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-id', required=True)
    args = p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Run source audits on Slurm')
    from src.framework.run_paths import run_directories
    root = run_directories(args.run_id, PROJECT)[0]
    manifest = json.loads((root / 'run_manifest.json').read_text())
    def artifact(step, key):
        return root / manifest['steps'][step]['artifacts'][key]
    from models.feature_extract.detsec_pc import PhyConstrainedDeTSEC, _extract, _normalize
    cfg = json.loads(artifact('feature_extract', 'model_config').read_text())
    x = np.load(artifact('time_segmentation', 'X'))
    lengths = np.load(artifact('time_segmentation', 'lengths')).ravel()
    xn = _normalize(x, lengths, cfg['norm_mode'])
    model = PhyConstrainedDeTSEC(cfg['n_features'], cfg['embed_dim'], cfg['nonneg_channels'],
        embed_proj=cfg['embed_proj'], nonneg_activation=cfg['nonneg_activation'])
    _extract(model, xn[:1], lengths[:1], 1)
    with np.load(artifact('feature_extract', 'weights')) as z:
        model.set_weights([z[f'arr_{i}'] for i in range(cfg['weight_count'])])
    actual = _extract(model, xn, lengths, 8)
    expected = np.load(artifact('feature_extract', 'features'))
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-4)
    with np.load(artifact('feature_extract', 'normalization')) as z:
        limits = z['percentile_limits']
    mask = np.arange(x.shape[1])[None, :] < lengths[:, None]
    np.testing.assert_allclose(limits, np.stack([np.percentile(x[:,:,i][mask], [1,99]) for i in range(x.shape[2])]))
    from nilm_experiments.nilm_lab.data import load_records, Windows
    from nilm_experiments.nilm_lab.activity_metrics import activity_reference, score_record
    source = json.loads(artifact('nilm_data', 'split_manifest').read_text())
    config = json.loads((root / 'nilm_data/request.json').read_text())['config']
    records, _ = load_records(source['input_manifest'], ('val',))
    baselines = []
    for r in records:
        reference = activity_reference(r, config['extract_active_data'], config['nilm']['sample_seconds'])
        windows = Windows([r], config['nilm']['window_length'], config['nilm']['eval_stride'], 1)
        covered = np.zeros(len(r.t), bool)
        for _, a in windows.index:
            covered[a:a+windows.length] = True
        pred = np.where(covered, 0., np.nan)
        baselines.append(score_record(r, pred, pred, reference, config['nilm']['threshold'], config['nilm']['sample_seconds']))
    snapshots = {}
    for folder in ('src/steps', 'src/utils', 'nilm_experiments/nilm_lab', 'models/feature_extract', 'models/extract_active_data'):
        for path in (PROJECT / folder).glob('*.py'):
            snapshots[path.relative_to(PROJECT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    report = dict(discovery_restore_verified=True, features_max_abs_difference=float(np.abs(actual-expected).max()),
                  normalization_verified=True, all_off_validation=baselines, test_opened=False,
                  source_code_sha256=snapshots, run_id=args.run_id)
    dest = root / 'source_audit.json'
    if dest.exists():
        raise FileExistsError(dest)
    dest.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k != 'source_code_sha256'}, indent=2))


if __name__ == '__main__':
    main()
