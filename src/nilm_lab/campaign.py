"""Approved staged campaign on immutable Step artifacts; target remains sealed until freeze."""
from copy import deepcopy
from pathlib import Path
import gc
import itertools
import numpy as np
from .common import read_json, write_json, digest, signature, code_snapshot


CASE_SPECS = {
    'R': ('R','none'), 'O': ('O','none'), 'B': ('B','none'), 'BP': ('BP','none'),
    'S': ('S','none'), 'P': ('P','none'), 'P_direct': ('P','direct_matched'),
    'P_point': ('P','point'), 'P_gru': ('P','gru'), 'BP_gru': ('BP','gru'),
    'O_gru': ('O','gru'), 'H_gru': ('R','feature_gru'),
    'O_common': ('O_common','none'), 'B_full': ('B_full','none')}
CORE_CASES = list(CASE_SPECS)[:12]


def trial_spec(model, case, seed, weight):
    arm, mode = CASE_SPECS[case]
    lp, la = (0., 0.) if case == 'R' else (0., weight) if case == 'H_gru' else (
        (weight, 0.) if mode == 'none' else (weight/2, weight/2))
    return dict(model=model, seed=int(seed), case=case, arm=arm, activity_head=mode,
                lambda_p=float(lp), lambda_a=float(la))


def immutable_json(path, value):
    path = Path(path)
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f'Immutable campaign artifact changed: {path}')
    else:
        write_json(path, value)


def run_entry(folder, nc, job, meta, code):
    rm, fm = read_json(folder/'run.json'), read_json(folder/'finished.json')
    if rm['trial'] != job or rm['config'] != nc or rm['state_metadata'] != meta or rm['code_sha256'] != code:
        raise ValueError(f'Trial provenance mismatch: {folder}')
    if digest(folder/'best.pt') != fm['checkpoint_sha256'] or not np.isfinite(fm['best_val_mae']):
        raise ValueError(f'Trial checkpoint mismatch: {folder}')
    return dict(path=str(folder.resolve()), val_mae=fm['best_val_mae'], trial=job,
                checkpoint_sha256=fm['checkpoint_sha256'], run_sha256=digest(folder/'run.json'))


def source_audit(req, records, val, labels, meta, references):
    from .data import Windows
    from .activity_metrics import score_record
    from .workflow import artifact
    cfg = req['config']['nilm']
    root = Path(req['output'])
    quality, baselines = [], []
    for r in records:
        z = labels[r.id]
        active = z['activity_valid'] & (z['onoff'] == 1)
        covered = active & z['valid']
        if np.any(z['valid'] & ~np.isfinite(r.y)):
            raise ValueError('Missing targets have state supervision')
        for key in ['primitive','bins','segment_bins','shape_bins']:
            if not np.array_equal(z[key] >= 0, z['valid']):
                raise ValueError('Control masks differ')
        counts = [len(set(z['activity_id'][covered & (z['primitive'] == j)].tolist())) for j in range(1,meta['k']+1)]
        tr = Windows([r], cfg['window_length'],cfg['train_stride'],meta['scale'],{r.id:z},'P')
        # Difference array counts actual repeated exposure without expanding all windows.
        delta = np.zeros(len(r.t)+1, np.int64)
        for _, start in tr.index:
            delta[start] += 1
            delta[start+tr.length] -= 1
        exposure = np.cumsum(delta[:-1])
        quality.append(dict(record_id=r.id, points=len(r.t), windows=len(tr),
            observed_target_points=int(np.isfinite(r.y).sum()), source_predictable_points=int((exposure>0).sum()),
            activity_points=int(active.sum()), state_activity_points=int(covered.sum()),
            activity_coverage=float(covered.sum()/max(1,active.sum())),
            primitive_class_activity_counts=counts,
            class_unique_counts=np.bincount(z['primitive'][z['valid']],minlength=meta['k']+1).tolist(),
            class_window_exposure=[int(exposure[z['primitive']==j].sum()) for j in range(meta['k']+1)],
            activity_full_only_points=int((z['activity_valid'] & ~z['valid']).sum())))
    source_y = np.concatenate([r.y[np.isfinite(r.y)] for r in records])
    standby = source_y[source_y <= cfg['threshold']]
    constants = dict(zero=0., standby=float(np.median(standby)) if len(standby) else 0.,
                     source_median=float(np.median(source_y)))
    for r in val:
        windows = Windows([r],cfg['window_length'],cfg['eval_stride'],meta['scale'])
        available = np.zeros(len(r.t),bool)
        for _, start in windows.index:
            available[start:start+windows.length] = True
        for name, value in constants.items():
            p = np.where(available,value,np.nan)
            score = score_record(r,p,np.where(available,0.,np.nan),references[r.id],cfg['threshold'],cfg['sample_seconds'])
            baselines.append(dict(name=name,value=value,score=score))
    # Layered dependency hashes coexist with the unchanged whole-run guard.
    parents = {f'{step}.{key}':digest(artifact(req,step,key)) for step,key in [
        ('nilm_data','split_manifest'),('nilm_labels','metadata'),('state_sequence','mapping'),('state_sequence','dictionary')]}
    occlusion_path = Path(req['log_root'])/'occlusion_audit.json'
    occlusion = read_json(occlusion_path)
    if not occlusion['source_only'] or occlusion['test_opened'] or occlusion['dictionary_sha256'] != parents['state_sequence.dictionary']:
        raise ValueError('Occlusion audit does not match frozen source dictionary')
    parents['occlusion_audit'] = digest(occlusion_path)
    report = dict(source_only=True,test_opened=False, quality=quality,baselines=baselines,
                  constants=constants,parent_hashes=parents, optional_cases=meta.get('optional_support_cases',[]),
                  quality_warnings=[r['record_id'] for r in quality if min(r['primitive_class_activity_counts'])<5],
                  occlusion_summary=[{k:v for k,v in report.items() if k!='diagnostics'} for report in occlusion['reports']])
    immutable_json(root/'source_qc.json',report)
    if any(r['activity_coverage'] < cfg['campaign']['min_activity_coverage'] for r in quality):
        raise ValueError('State coverage below predeclared source threshold; see source_qc.json')
    return report


def train_campaign(req):
    import torch
    if not req.get('worker_model') and torch.cuda.device_count() >= 2:
        return parallel_campaign(req)
    from .workflow import source_records,load_labels,fingerprint
    from .data import load_records
    from .activity_metrics import activity_reference
    from .train import fit_run
    records, source = source_records(req)
    labels, meta = load_labels(req,records)
    val,_ = load_records(source['input_manifest'],('val',))
    if {r.content_hash for r in records} & {r.content_hash for r in val}:
        raise ValueError('Source/validation duplicate recording')
    cfg = req['config']
    out = Path(req['output'])
    base = dict(deepcopy(cfg['nilm']), config_fingerprint=fingerprint(cfg))
    if req.get('worker_model'):
        if req['worker_model'] not in base['models']:
            raise ValueError('Unknown worker backbone')
        base['models'] = [req['worker_model']]
    cp = base['campaign']
    code = code_snapshot()
    refs = {r.id:activity_reference(r,cfg['extract_active_data'],base['sample_seconds']) for r in val}
    audit = source_audit(req,records,val,labels,meta,refs)
    base['parent_artifact_hashes'] = audit['parent_hashes']
    cases = CORE_CASES + meta.get('optional_support_cases',[])
    immutable_json(out/'campaign_lock.json',dict(code=code,config=base,labels=meta,cases=cases))
    all_entries = {}
    def run(model,case,seed,weight,lr,reduction='onoff_balanced'):
        nc = deepcopy(base)
        nc['training'].update(lr=lr,auxiliary_reduction=reduction)
        job = trial_spec(model,case,seed,weight)
        key = signature(dict(config=nc,trial=job))[:16]
        folder = out/f'{model}_{case}_s{seed}_{key}'
        if not folder.exists():
            print('CAMPAIGN_TRAIN',job,'lr',lr,'reduction',reduction,flush=True)
            model_obj = fit_run(nc,records,val,labels,meta,job,folder,refs)
            del model_obj
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        entry = run_entry(folder,nc,job,meta,code)
        all_entries[key] = entry
        write_json(out/'trials.json',list(all_entries.values()))
        return entry
    selected_lr = {}
    lr_results = []
    for model in base['models']:
        choices = []
        for lr in cp['learning_rates']:
            entries = [run(model,'R',s,0.,lr) for s in cp['development_seeds']]
            mean = float(np.mean([e['val_mae'] for e in entries]))
            choices.append((mean,lr))
            lr_results.append(dict(model=model,lr=lr,mean_mae=mean,runs=entries))
        selected_lr[model] = min(choices)[1]
    immutable_json(out/'learning_rate_selection.json',dict(selected=selected_lr,results=lr_results,test_opened=False))
    selected = {}
    development = []
    for model,case in itertools.product(base['models'],cases):
        choices = []
        for weight in ([0.] if case=='R' else cp['auxiliary_weights']):
            entries = [run(model,case,s,weight,selected_lr[model]) for s in cp['development_seeds']]
            mean = float(np.mean([e['val_mae'] for e in entries]))
            choices.append((mean,weight))
            development.append(dict(model=model,case=case,weight=weight,mean_mae=mean,runs=entries))
        best = min(choices)
        selected[(model,case)] = dict(weight=best[1],mean_mae=best[0])
    controls = [run(model,case,seed,.01,selected_lr[model],reduction='mean')
                for model,case,seed in itertools.product(base['models'],cp['loss_control_cases'],cp['development_seeds'])]
    source_selection = dict(selected=[dict(model=m,case=c,**s) for (m,c),s in selected.items()],
        learning_rates=selected_lr,development=development,loss_control_runs=controls,test_opened=False,
        selection_rule='mean best H5 MAE across development seeds; lower lambda tie break',
        simple_control={},sequence_control={})
    for model in base['models']:
        source_selection['simple_control'][model] = min(
            [c for c in ['O','B','BP','S','B_full'] if c in cases],key=lambda c:(selected[(model,c)]['mean_mae'],c))
        source_selection['sequence_control'][model] = min(
            ['P_direct','P_point','H_gru'],key=lambda c:(selected[(model,c)]['mean_mae'],c))
    immutable_json(out/'source_selection.json',source_selection)
    baseline_scores = {}
    from .train import house_macro_mae
    for name in audit['constants']:
        baseline_scores[name] = house_macro_mae([e['score'] for e in audit['baselines'] if e['name']==name])
    strongest_constant = min(baseline_scores.values())
    source_pass = all(selected[(m,source_selection['simple_control'][m])]['mean_mae'] < strongest_constant
                      for m in base['models'])
    gate = dict(passed=source_pass,baseline_mae=baseline_scores,
        rule='Best source-selected simple group on each backbone beats strongest source-fitted constant on H5',
        source_selection_sha256=digest(out/'source_selection.json'),test_opened=False,
        class_support_warnings=audit['quality_warnings'])
    immutable_json(out/'source_gate.json',gate)
    if not source_pass:
        raise ValueError('Source efficacy gate failed; development preserved; formal/test not run')
    formal = [run(model,case,seed,selected[(model,case)]['weight'],selected_lr[model])
              for model,case,seed in itertools.product(base['models'],cases,cp['formal_seeds'])]
    immutable_json(out/'formal_freeze.json',dict(config_fingerprint=fingerprint(cfg),runs=formal,
        source_selection_sha256=digest(out/'source_selection.json'),source_gate_sha256=digest(out/'source_gate.json'),
        code=code,parent_hashes=audit['parent_hashes'],test_opened=False,
        selection_rule='lambda/lr frozen across formal seeds; each checkpoint selected on source H5 MAE'))


def parallel_campaign(req):
    """Independent backbone workers on two allocated GPUs; same shared teacher."""
    import os
    import subprocess
    import sys
    out=Path(req['output'])
    devices=os.environ.get('CUDA_VISIBLE_DEVICES','0,1').split(',')
    models=req['config']['nilm']['models']
    if len(devices)<len(models):
        raise ValueError('Insufficient allocated GPUs')
    workers=[]
    for model,device in zip(models,devices):
        child=deepcopy(req)
        child['output']=str(out/model)
        child['worker_model']=model
        path=out/model/'worker_request.json'
        immutable_json(path,child)
        env=os.environ.copy()
        env['CUDA_VISIBLE_DEVICES']=device
        log=(out/model/'worker.log').open('a',encoding='utf-8')
        process=subprocess.Popen([sys.executable,'-u','-m','nilm_lab.campaign',str(path)],env=env,stdout=log,stderr=subprocess.STDOUT)
        workers.append((model,process,log))
    failed=[]
    for model,process,log in workers:
        result=process.wait();log.close()
        if result:
            failed.append((model,result))
    if failed:
        raise RuntimeError(f'Backbone campaign failed, test sealed: {failed}')
    freezes=[read_json(out/m/'formal_freeze.json') for m in models]
    if any(f['code']!=code_snapshot() or f['parent_hashes']!=freezes[0]['parent_hashes'] for f in freezes):
        raise ValueError('Backbones used different teacher/runtime')
    selections=[read_json(out/m/'source_selection.json') for m in models]
    selection=dict(test_opened=False,selected=sum([s['selected'] for s in selections],[]),
        development=sum([s['development'] for s in selections],[]),
        loss_control_runs=sum([s['loss_control_runs'] for s in selections],[]),
        simple_control={k:v for s in selections for k,v in s['simple_control'].items()},
        sequence_control={k:v for s in selections for k,v in s['sequence_control'].items()},
        learning_rates={k:v for s in selections for k,v in s['learning_rates'].items()})
    immutable_json(out/'source_selection.json',selection)
    gates={m:read_json(out/m/'source_gate.json') for m in models}
    immutable_json(out/'source_gate.json',dict(passed=all(g['passed'] for g in gates.values()),backbones=gates,test_opened=False))
    immutable_json(out/'source_qc.json',read_json(out/models[0]/'source_qc.json'))
    immutable_json(out/'trials.json',sum([read_json(out/m/'trials.json') for m in models],[]))
    freeze=deepcopy(freezes[0])
    freeze.update(runs=sum([f['runs'] for f in freezes],[]),
        source_selection_sha256=digest(out/'source_selection.json'),source_gate_sha256=digest(out/'source_gate.json'))
    immutable_json(out/'formal_freeze.json',freeze)


def select_campaign(req):
    from .workflow import artifact,fingerprint
    root = artifact(req,'nilm_train','trials').parent
    freeze = read_json(root/'formal_freeze.json')
    if freeze['config_fingerprint'] != fingerprint(req['config']) or freeze['code'] != code_snapshot():
        raise ValueError('Campaign changed after freeze')
    if digest(root/'source_selection.json') != freeze['source_selection_sha256'] or digest(root/'source_gate.json') != freeze['source_gate_sha256']:
        raise ValueError('Source selection/gate changed')
    if not read_json(root/'source_gate.json')['passed']:
        raise ValueError('Source gate failed')
    for entry in freeze['runs']:
        p = Path(entry['path'])
        if digest(p/'run.json') != entry['run_sha256'] or digest(p/'best.pt') != entry['checkpoint_sha256']:
            raise ValueError('Formal checkpoint changed')
    immutable_json(Path(req['output'])/'selection.json',freeze)


if __name__=='__main__':
    import sys
    train_campaign(read_json(sys.argv[1]))
