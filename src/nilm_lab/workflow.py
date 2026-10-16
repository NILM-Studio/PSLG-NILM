"""Isolated numeric worker for Step -> Workflow -> RunManifest.

Only `evaluate` opens test records. All other commands resolve source artifacts.
"""
from __future__ import annotations
import argparse
import gc
import os
from pathlib import Path
import sys
import numpy as np
from .common import digest, signature, read_json, write_json, code_snapshot
from .data import Record, load_records, validate_manifest


def fingerprint(cfg):
    return signature({k: v for k, v in cfg.items() if not k.startswith('_')})


def path_from_config(cfg, path):
    return (Path(cfg['_config_path']).parent / path).resolve()


def artifact(req, step, key):
    p = req['manifest']['steps'].get(step, {}).get('artifacts', {}).get(key)
    if not p:
        raise ValueError(f'Missing prerequisite {step}.{key}')
    return Path(req['log_root']) / p


def source_records(req):
    p = artifact(req, 'nilm_data', 'split_manifest')
    m = read_json(p)
    if m['config_fingerprint'] != fingerprint(req['config']):
        raise ValueError('Frozen config changed')
    if digest(m['input_manifest']) != m['input_manifest_sha256']:
        raise ValueError('Split manifest changed')
    records = []
    for e in m['records']:
        path = p.parent / e['file']
        if e['split'] != 'train' or digest(path) != e['sha256']:
            raise ValueError('Source training artifact mismatch')
        with np.load(path) as z:
            records.append(Record(e['id'], e['dataset'], e['house'], 'train', z['timestamp'],
                z['mains'], z['target'], e['fingerprint'], e['content_hash'], z['context_valid'], z['mains_observed']))
    return records, m


def prepare_data(req):
    cfg, out = req['config'], Path(req['output'])
    manifest = path_from_config(cfg, cfg['data_protocol']['manifest'])
    raw = read_json(manifest)
    validate_manifest(raw)
    if raw['sample_seconds'] != cfg['nilm']['sample_seconds']:
        raise ValueError('Sampling mismatch')
    trials(cfg['nilm'])
    tc = cfg['nilm']['training']
    if min(tc['epochs'], tc['patience'], tc['batch_size']) < 1:
        raise ValueError('Invalid training budget')
    records, _ = load_records(manifest, ('train',))
    entries, qc = [], []
    for i, r in enumerate(records):
        name = f'source_{i}.npz'
        context = np.isfinite(r.x) if r.context_valid is None else r.context_valid
        ox = np.isfinite(r.x) if r.x_observed is None else r.x_observed
        np.savez_compressed(out / name, timestamp=r.t, mains=r.x, target=r.y,
                            context_valid=context, mains_observed=ox)
        entry = next(e for e in raw['records'] if e['id'] == r.id)
        entries.append(dict(id=r.id, dataset=r.dataset, house=r.house, split=r.split,
            appliance=cfg['run']['appliance'], device_id=entry.get('device_id', r.id),
            sample_seconds=raw['sample_seconds'], file=name, sha256=digest(out / name),
            fingerprint=r.fingerprint, content_hash=r.content_hash,
            start_timestamp=int(r.t[0]), end_timestamp_exclusive=int(r.t[-1]+raw['sample_seconds'])))
        qc.append(dict(record_id=r.id, points=len(r.t), target_observed=int(np.isfinite(r.y).sum()),
                       mains_observed=int(ox.sum()), context_valid=int(context.sum())))
    write_json(out / 'split_manifest.json', dict(records=entries, protocol=raw,
        input_manifest=str(manifest), input_manifest_sha256=digest(manifest),
        config_fingerprint=fingerprint(cfg), source_only=True, test_opened=False))
    write_json(out / 'data_qc.json', qc)


def prepare_labels(req):
    from .sequence_data import fit_label_bundles
    cfg, out = req['config'], Path(req['output'])
    records, source = source_records(req)
    mp = artifact(req, 'state_sequence', 'mapping')
    sm = read_json(mp)
    if not sm['source_only'] or sm['k'] != cfg['nilm']['k']:
        raise ValueError('Only matching source-only dictionaries may supervise NILM')
    for path, sha in sm['input_hashes'].items():
        if digest(path) != sha:
            raise ValueError('State discovery inputs changed')
    ap = artifact(req, 'extract_active_data', 'activities')
    aa = read_json(ap)
    if aa['activity_config'] != cfg['extract_active_data']:
        raise ValueError('Activity detector configuration mismatch')
    activities = {r.id: [a for a in aa['activities'] if a['record_id'] == r.id] for r in records}
    mappings = {}
    for e in sm['records']:
        p = mp.parent / e['file']
        if digest(p) != e['sha256']:
            raise ValueError('Mapping changed')
        with np.load(p) as z:
            mappings[e['record_id']] = dict(z)
    for r in records:
        if not np.array_equal(r.t, mappings[r.id]['timestamp']):
            raise ValueError('Mapping timestamps do not match frozen training timeline')
    bundles, meta = fit_label_bundles(records, activities, mappings, cfg['nilm']['k'],
                                     include_shape=cfg['nilm'].get('include_shape', False))
    meta.update(dictionary_id=sm['dictionary_id'], config_fingerprint=fingerprint(cfg),
                manifest_sha256=source['input_manifest_sha256'],
                mapping_sha256=digest(mp), activity_sha256=digest(ap), records=[])
    qc = []
    for i, r in enumerate(records):
        z = bundles[r.id]
        name = f'labels_{i}.npz'
        np.savez_compressed(out / name, timestamp=r.t, **z)
        meta['records'].append(dict(id=r.id, fingerprint=r.fingerprint, file=name, sha256=digest(out / name)))
        active = z['onoff'] == 1
        qc.append(dict(record_id=r.id, activity_points=int(active.sum()),
            supervised_active_points=int((active & z['valid']).sum()),
            active_coverage=float((active & z['valid']).sum()/max(1, active.sum())),
            low_power_active_points=int((active & (r.y <= cfg['nilm']['threshold'])).sum()),
            low_power_active_supervised=int((active & z['valid'] & (r.y <= cfg['nilm']['threshold'])).sum()),
            observed_regression_points=int(z['regression_observed'].sum()),
            class_counts={key: np.bincount(z[key][z[key] >= 0], minlength=cfg['nilm']['k']+1).tolist()
                          for key in ('primitive', 'bins', 'segment_bins')},
            missing_target_auxiliary_points=int((z['valid'] & ~np.isfinite(r.y)).sum())))
    write_json(out / 'metadata.json', meta)
    write_json(out / 'label_qc.json', qc)


def trials(cfg):
    jobs = []
    for model in cfg['models']:
        for seed in cfg['seeds']:
            for case in cfg['cases']:
                if model == 'BERT4NILM' and case['arm'] != 'R':
                    continue
                if model == 'SGN' and (case['arm'] != 'O' or case.get('activity_head', 'none') != 'none'):
                    continue
                for lp, la in case['weights']:
                    job = dict(model=model, seed=seed, case=case['name'], arm=case['arm'],
                        activity_head=case.get('activity_head', 'none'), lambda_p=float(lp), lambda_a=float(la))
                    if (job['arm'] not in {'R', 'O', 'B', 'BP', 'P'} or min(lp, la) < 0
                            or not np.isfinite([lp, la]).all() or (job['arm'] == 'R' and (lp or la))
                            or (la and job['activity_head'] == 'none') or (job['activity_head'] == 'gru' and job['arm'] == 'R')):
                        raise ValueError('Invalid trial definition')
                    jobs.append(job)
    if len({signature(j) for j in jobs}) != len(jobs) or not jobs:
        raise ValueError('Empty or duplicate trial matrix')
    return jobs


def trial_folder(root, job):
    return Path(root) / f"{job['model']}_{job['case']}_s{job['seed']}_{signature(job)[:12]}"


def load_labels(req, records):
    p = artifact(req, 'nilm_labels', 'metadata')
    m = read_json(p)
    if m['config_fingerprint'] != fingerprint(req['config']):
        raise ValueError('Label/config mismatch')
    mp = artifact(req, 'state_sequence', 'mapping')
    if digest(mp) != m['mapping_sha256'] or read_json(mp)['dictionary_id'] != m['dictionary_id']:
        raise ValueError('Dictionary changed after label export')
    result = {}
    for r in records:
        e = next(e for e in m['records'] if e['id'] == r.id)
        path = p.parent / e['file']
        if e['fingerprint'] != r.fingerprint or digest(path) != e['sha256']:
            raise ValueError('Label data or training support mismatch')
        with np.load(path) as z:
            if not np.array_equal(z['timestamp'], r.t):
                raise ValueError('Label timestamps changed')
            result[r.id] = dict(z)
    return result, m


def train(req):
    if req['config']['nilm'].get('campaign'):
        from .campaign import train_campaign
        return train_campaign(req)
    from .train import fit_run
    from .activity_metrics import activity_reference
    cfg, out = req['config'], Path(req['output'])
    records, source = source_records(req)
    labels, meta = load_labels(req, records)
    val, _ = load_records(source['input_manifest'], ('val',))
    if {r.content_hash for r in val} & {r.content_hash for r in records}:
        raise ValueError('Duplicate recording across train and validation')
    references = {r.id: activity_reference(r, cfg['extract_active_data'], cfg['nilm']['sample_seconds']) for r in val}
    nc = dict(cfg['nilm'], config_fingerprint=fingerprint(cfg))
    planned = trials(nc)
    write_json(out / 'trials.json', planned)
    for job in planned:
        folder = trial_folder(out, job)
        if folder.exists():
            rm = read_json(folder / 'run.json')
            fm = read_json(folder / 'finished.json')
            if (rm['trial'] != job or rm['config'] != nc or rm['state_metadata'] != meta
                    or rm['code_sha256'] != code_snapshot() or digest(folder / 'best.pt') != fm['checkpoint_sha256']):
                raise ValueError('Existing trial changed; use new run-id, never overwrite')
            continue
        print('TRAIN', job, flush=True)
        model = fit_run(nc, records, val, labels, meta, job, folder, references)
        del model
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def select(req):
    if req['config']['nilm'].get('campaign'):
        from .campaign import select_campaign
        return select_campaign(req)
    cfg, out = req['config'], Path(req['output'])
    root = artifact(req, 'nilm_train', 'trials').parent
    planned = trials(cfg['nilm'])
    if read_json(root / 'trials.json') != planned:
        raise ValueError('Trial matrix changed')
    groups = {}
    for job in planned:
        folder = trial_folder(root, job)
        rm, fm = read_json(folder / 'run.json'), read_json(folder / 'finished.json')
        if rm['trial'] != job or rm['config']['config_fingerprint'] != fingerprint(cfg):
            raise ValueError('Run provenance mismatch')
        if not np.isfinite(fm['best_val_mae']) or digest(folder / 'best.pt') != fm['checkpoint_sha256']:
            raise ValueError('Invalid validation result/checkpoint')
        groups.setdefault((job['model'], job['case'], job['seed']), []).append(dict(
            path=str(folder.resolve()), val_mae=fm['best_val_mae'], trial=job,
            checkpoint_sha256=fm['checkpoint_sha256'], run_sha256=digest(folder / 'run.json')))
    chosen = [min(values, key=lambda e: (e['val_mae'], e['path'])) for _, values in sorted(groups.items())]
    write_json(out / 'selection.json', dict(config_fingerprint=fingerprint(cfg), runs=chosen,
        selection_rule='source_validation_house_macro_MAE', test_opened=False))


def evaluate(req):
    import torch
    from .train import model_for_trial, predict, score_records
    from .data import Windows
    from .activity_metrics import activity_reference
    cfg, out = req['config'], Path(req['output'])
    selection_path = artifact(req, 'nilm_select', 'selection')
    selection = read_json(selection_path)
    if selection['config_fingerprint'] != fingerprint(cfg):
        raise ValueError('Selection/config mismatch')
    _, source = source_records(req)
    # Validate all selected checkpoints BEFORE accessing held-out records.
    runs = []
    for entry in selection['runs']:
        folder = Path(entry['path'])
        if digest(folder / 'run.json') != entry['run_sha256'] or digest(folder / 'best.pt') != entry['checkpoint_sha256']:
            raise ValueError('Selected run/checkpoint changed')
        rm = read_json(folder / 'run.json')
        if rm['code_sha256'] != code_snapshot():
            raise ValueError('Runtime changed since training')
        runs.append((entry, rm))
    test, _ = load_records(source['input_manifest'], ('test',))
    references = {r.id: activity_reference(r, cfg['extract_active_data'], cfg['nilm']['sample_seconds']) for r in test}
    results, outputs = [], []
    for j, (entry, rm) in enumerate(runs):
        if {r.content_hash for r in test} & set(rm['source_record_content_hashes']):
            raise ValueError('Test recording duplicates source data')
        nc, meta, job = rm['config'], rm['state_metadata'], rm['trial']
        model = model_for_trial(nc, meta, job)
        model.load_state_dict(torch.load(Path(entry['path']) / 'best.pt', map_location='cpu', weights_only=True)['model'])
        device = torch.device(nc['training']['device'])
        model.to(device)
        data = Windows(test, nc['window_length'], nc['eval_stride'], meta['scale'])
        predictions = predict(model, data, device, nc['training']['batch_size'])
        scores = score_records(test, predictions, references, nc['threshold'], nc['sample_seconds'])
        results.append(dict(trial=job, scores=scores, checkpoint_sha256=entry['checkpoint_sha256']))
        for i, (r, pred) in enumerate(zip(test, predictions)):
            name = f'prediction_{j}_{i}.npz'
            np.savez_compressed(out / name, timestamp=r.t, target=r.y,
                **{key: value for key, value in pred.items() if value is not None})
            outputs.append(dict(trial=job, record_id=r.id, file=name, sha256=digest(out / name),
                                record_fingerprint=r.fingerprint, dictionary_id=meta['dictionary_id']))
        del model
        gc.collect()
    write_json(out / 'prediction_manifest.json', outputs)
    write_json(out / 'results.json', dict(selection_sha256=digest(selection_path), results=results, test_opened=True))


def report(req):
    out = Path(req['output'])
    selection = read_json(artifact(req, 'nilm_select', 'selection'))
    rows = []
    for e in selection['runs']:
        folder = Path(e['path'])
        rm, fm = read_json(folder / 'run.json'), read_json(folder / 'finished.json')
        rows.append(dict(**rm['trial'], val_mae=e['val_mae'], seconds=fm['seconds'],
                          epochs=fm['epochs'], parameters=rm['parameter_count'],
                          validation_scores=read_json(folder / 'validation_scores.json')))
    evaluation = req['manifest']['steps'].get('nilm_evaluate')
    tests = read_json(artifact(req, 'nilm_evaluate', 'results')) if evaluation else None
    comparisons = []
    for model, seed in sorted({(r['model'], r['seed']) for r in rows}):
        cases = {r['case']: r for r in rows if r['model'] == model and r['seed'] == seed}
        for baseline, proposed in [('R', 'O'), ('B', 'P'), ('BP', 'P'), ('P_direct', 'P_gru'), ('BP_gru', 'P_gru'), ('O_gru', 'P_gru')]:
            if baseline in cases and proposed in cases:
                comparisons.append(dict(model=model, seed=seed, baseline=baseline, proposed=proposed,
                    validation_mae_improvement=cases[baseline]['val_mae']-cases[proposed]['val_mae']))
    write_json(out / 'report.json', dict(validation=rows, paired_validation=comparisons, test=tests,
                note='Development results are not held-out efficacy evidence; seeds are not independent houses'))
    lines = ['# NILM sequence workflow results', '',
             'Source validation only; held-out tests have not been opened.' if tests is None else 'Explicit held-out evaluation is included in report.json.',
             '', '| Model | Case | Seed | Validation MAE (W) | Parameters | Seconds |', '|---|---|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['model']} | {r['case']} | {r['seed']} | {r['val_mae']:.4f} | {r['parameters']} | {r['seconds']:.1f} |")
    lines += ['', 'Activity references use the original detector. Capacity and activity-supervision differences must be considered in sequence comparisons.',
              'This first implementation predicts one target appliance per run; it does not build a joint dictionary for all appliances.']
    (out / 'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def main_request(req):
    if req['config'].get('runtime', {}).get('require_slurm', True) and not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Run data preparation/training/evaluation through Slurm')
    commands = dict(data=prepare_data, labels=prepare_labels, train=train, select=select, evaluate=evaluate, report=report)
    commands[req['command']](req)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('request')
    main_request(read_json(p.parse_args().request))


if __name__ == '__main__':
    main()
