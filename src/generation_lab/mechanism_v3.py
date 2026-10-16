"""Slurm-only mechanism campaign: prepare -> pilot -> freeze -> train -> evaluate.

No historical campaign is mutated. Real dev evaluation is explicitly exploratory.
New held-out data require an explicit provenance contract before any read.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import random
import time
import joblib
import numpy as np
import torch
from .common import digest, read, write, require_slurm
from .paths import resolve_campaign
from .mechanism_data import matrix, load_dictionary, synthetic, examples
from .mechanism_model import StateDiffusion
from .mechanism_metrics import fit_teacher, evaluate, features
from sklearn.metrics import f1_score, confusion_matrix, classification_report


def code_hashes():
    return {p.name: digest(p) for p in Path(__file__).parent.glob('mechanism_*.py')}


def bundle(batch, device):
    keys = ('y','z','mask','pos','duration')
    return tuple(torch.as_tensor(np.stack([r[k] for r in batch]), device=device) for k in keys)


def store_records(path, records):
    sizes = [len(r['y']) for r in records]
    np.savez_compressed(path, y=np.concatenate([r['y'] for r in records]),
                        z=np.concatenate([r['z'] for r in records]),
                        offsets=np.r_[0,np.cumsum(sizes)], ids=np.asarray([r['id'] for r in records]))


def read_records(path):
    with np.load(path,allow_pickle=False) as a:
        return [dict(id=str(a['ids'][i]), y=a['y'][lo:hi].copy(), z=a['z'][lo:hi].copy())
                for i,(lo,hi) in enumerate(zip(a['offsets'][:-1],a['offsets'][1:]))]


def prepare(args, root):
    if root.exists():
        raise ValueError('Use a new campaign; prepare never overwrites existing outputs')
    data, meta = load_dictionary(args.dictionary)
    n_states=int(meta.get('k',5))
    scale = max(1.,float(np.quantile(np.concatenate([r['y'] for r in data['train']]),.99)))
    root.mkdir(parents=True)
    files = {}
    for split, records in data.items():
        p=root/f'real_{split}.npz';store_records(p,records);files[p.name]=digest(p)
    # Synthetic mechanisms share clean data across noise levels and methods.
    for duration in ('fixed','variable'):
        for split in ('train','dev','test'):
            p=root/f'synthetic_{duration}_{split}.npz'
            store_records(p,synthetic(split,duration=='variable',args.synthetic_activities,n_states))
            files[p.name]=digest(p)
    for dataset in ('real','synthetic_fixed','synthetic_variable'):
        if not getattr(args,'teacher_diagnostics',False):continue
        train=read_records(root/f'{dataset}_train.npz');dev=read_records(root/f'{dataset}_dev.npz')
        teacher=fit_teacher(train);p=root/f'{dataset}_teacher.joblib';joblib.dump(teacher,p);files[p.name]=digest(p)
        z=np.concatenate([r['z'] for r in dev]);pred=np.concatenate([teacher.predict(features(r['y'])) for r in dev])
        write(root/f'{dataset}_teacher_quality.json',dict(dev_macro_f1=float(f1_score(z,pred,labels=range(1,n_states+1),average='macro',zero_division=0)),
                  n_states=n_states,confusion_matrix=confusion_matrix(z,pred,labels=range(1,n_states+1)).tolist(),
                  per_class=classification_report(z,pred,labels=range(1,n_states+1),output_dict=True,zero_division=0),
                  status='diagnostic; validate teacher before interpreting generator adherence'))
    trials=matrix()
    for i,row in enumerate(trials):row['index']=i
    write(root/'plan.json',trials)
    write(root/'manifest.json',dict(schema='mechanism_v3',code_hashes=code_hashes(),files=files,
          dictionary=str(Path(args.dictionary).resolve()),dictionary_manifest_sha256=digest(Path(args.dictionary)/'manifest.json'),
          dictionary_identity=meta['dictionary_sha256'],scale=scale,length=args.length,n_states=n_states,
          source_scope=meta.get('evaluation_scope','training-dictionary'),
          max_updates=args.updates,pilot_updates=args.pilot_updates,batch_size=32,width=64,
          validation_interval=args.validation_interval,plan_sha256=digest(root/'plan.json'),
          test_contract_sha256=digest(args.test_contract) if args.test_contract else None,
          evaluation_schema='waveform_quality_v2',teacher_diagnostics=getattr(args,'teacher_diagnostics',False),
          robust_consistency_weight=.2,robust_extra_noise_normalized_std=.03,
          confirmation_status='real dev only; fresh held-out contract required',
          synthetic_noise_units='fraction of fixed 1500 W scale; Gaussian plus 1% impulses',
          backbones='Project-built masked transformer and dilated convolution; not upstream reproductions'))


def contract(root):
    meta=read(root/'manifest.json')
    if meta['code_hashes']!=code_hashes() or meta['plan_sha256']!=digest(root/'plan.json'):
        raise ValueError('Frozen code or plan changed; use a new campaign')
    return meta


def checked_file(root,meta,name):
    p=root/name
    if digest(p)!=meta['files'][name]:raise ValueError(f'Changed campaign input: {name}')
    return p


def dataset_name(trial):
    return 'real' if trial['data']=='real' else 'synthetic_'+trial['duration']


def trial_dir(root,trial):
    return root/'trials'/f'{trial["index"]:03d}_{trial["phase"]}_{trial["backbone"]}_{trial["case"]}_s{trial["seed"]}'


def freeze(root):
    meta=contract(root);pilots=[t for t in read(root/'plan.json') if t['phase']=='pilot']
    hashes={}
    for t in pilots:
        folder=trial_dir(root,t);f=read(folder/'finished.json')
        if f['status']!='completed' or f['manifest_sha256']!=digest(root/'manifest.json'):
            raise ValueError('Pilot incomplete or stale')
        if f['checkpoint_sha256']!=digest(folder/'best.pt'):raise ValueError('Changed pilot checkpoint')
        hashes[str(t['index'])]=digest(folder/'finished.json')
    if (root/'freeze.json').exists():raise ValueError('Already frozen')
    write(root/'freeze.json',dict(manifest_sha256=digest(root/'manifest.json'),pilot_hashes=hashes,
          gate='engineering only; classifier quality is not a gate',
          primary_metrics=['waveform_quality.global_power_w1','waveform_quality.energy_kwh_w1'],
          secondary_metrics=['waveform_quality.mean_log_psd_l1','waveform_quality.waveform_std_w1',
                             'conditional_quality.macro_state_power_w1'],
          confirmatory_real_test_opened=False))


def train(args,root):
    meta=contract(root);trial=read(root/'plan.json')[args.index]
    if trial['phase']=='formal':
        if read(root/'freeze.json')['manifest_sha256']!=digest(root/'manifest.json'):raise ValueError('Freeze mismatch')
    if not torch.cuda.is_available():raise RuntimeError('GPU allocation required')
    seed=trial['seed'];torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    torch.set_num_threads(4)
    out=trial_dir(root,trial);out.mkdir(parents=True,exist_ok=True)
    if (out/'finished.json').exists():
        f=read(out/'finished.json')
        if f['manifest_sha256']!=digest(root/'manifest.json') or f['checkpoint_sha256']!=digest(out/'best.pt'):
            raise ValueError('Completed trial provenance mismatch')
        return
    dataset=dataset_name(trial);scale=meta['scale'] if trial['data']=='real' else 1500.
    sets={}
    for split in ('train','dev'):
        records=read_records(checked_file(root,meta,f'{dataset}_{split}.npz'))
        sets[split]=examples(records,trial['case'],meta['length'],seed=711 if split=='train' else 712,
                             noise=trial['noise'],scale=scale,n_states=meta['n_states'])
    # Equal probability per activity; within activity use valid-point weights.
    # Therefore splitting into more blocks does not multiply an activity's weight.
    groups={}
    for i,r in enumerate(sets['train']):groups.setdefault(r['id'],[]).append(i)
    aids=sorted(groups)
    probabilities={aid:np.array([sets['train'][i]['mask'].sum() for i in ids],float) for aid,ids in groups.items()}
    probabilities={aid:p/p.sum() for aid,p in probabilities.items()}
    model=StateDiffusion(trial['backbone'],meta['width'],meta['n_states']).cuda();ema=copy.deepcopy(model).eval()
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4)
    first=0;best=float('inf');history=[];stale=0
    provenance=dict(trial=trial,manifest_sha256=digest(root/'manifest.json'),job_id=os.environ['SLURM_JOB_ID'])
    if (out/'resume.pt').exists():
        ck=torch.load(out/'resume.pt',map_location='cuda',weights_only=False)
        if ck['manifest_sha256']!=provenance['manifest_sha256']:raise ValueError('Resume contract mismatch')
        model.load_state_dict(ck['model']);ema.load_state_dict(ck['ema']);opt.load_state_dict(ck['optimizer'])
        first,best,history,stale=ck['step'],ck['best'],ck['history'],ck['stale']
    write(out/'run.json',provenance)
    max_steps=meta['pilot_updates'] if trial['phase']=='pilot' else meta['max_updates']
    vrng=np.random.default_rng(778)
    val=[sets['dev'][i] for i in vrng.choice(len(sets['dev']),min(256,len(sets['dev'])),replace=False)]
    started=time.time()
    for step in range(first+1,max_steps+1):
        torch.manual_seed(seed*100003+step)
        rng=np.random.default_rng(seed*100003+step)
        ids=[int(rng.choice(groups[a],p=probabilities[a])) for a in rng.choice(aids,meta['batch_size'])]
        batch=bundle([sets['train'][i] for i in ids],'cuda')
        model.train();opt.zero_grad(set_to_none=True)
        loss=model.loss(batch,robust=trial['case']=='block_robust')
        if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
        with torch.no_grad():
            for e,p in zip(ema.parameters(),model.parameters()):e.lerp_(p,.005)
        if step%meta['validation_interval']==0 or step==max_steps:
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]),torch.no_grad():
                torch.manual_seed(991);scores=[]
                for b in range(0,len(val),32):
                    chunk=val[b:b+32];scores.append((float(ema.loss(bundle(chunk,'cuda'))),len(chunk)))
                score=float(np.average([s for s,n in scores],weights=[n for s,n in scores]))
            history.append(dict(step=step,train_loss=float(loss),dev_raw_x0_l1=score))
            if score<best:
                best=score;stale=0
                torch.save(dict(model=ema.state_dict(),trial=trial,scale=scale,**{k:v for k,v in provenance.items() if k!='trial'}),out/'best.tmp')
                (out/'best.tmp').replace(out/'best.pt')
            else:stale+=1
            torch.save(dict(model=model.state_dict(),ema=ema.state_dict(),optimizer=opt.state_dict(),step=step,
                            best=best,history=history,stale=stale,manifest_sha256=provenance['manifest_sha256']),out/'resume.tmp')
            (out/'resume.tmp').replace(out/'resume.pt')
            write(out/'history.json',history);print(args.index,step,score,flush=True)
            if stale>=5:break
    ck=torch.load(out/'best.pt',map_location='cuda',weights_only=False);ema.load_state_dict(ck['model'])
    smoke=ema.sample(bundle(val[:4],'cuda'),steps=5)
    if not torch.isfinite(smoke).all():raise ValueError('Nonfinite pilot sample')
    write(out/'finished.json',dict(status='completed',**provenance,checkpoint_sha256=digest(out/'best.pt'),
          steps=history[-1]['step'],parameters=sum(p.numel() for p in model.parameters()),
          wall_seconds_this_invocation=time.time()-started,max_cuda_memory_bytes=torch.cuda.max_memory_allocated(),
          test_opened=False))


def score(args,root):
    meta=contract(root);trial=read(root/'plan.json')[args.index];out=trial_dir(root,trial)
    complete=read(out/'finished.json')
    if complete['checkpoint_sha256']!=digest(out/'best.pt'):raise ValueError('Changed checkpoint')
    if not torch.cuda.is_available():raise RuntimeError('Sampling requires allocated GPU')
    dataset=dataset_name(trial);split='test' if trial['data']=='synthetic' else 'dev'
    if trial['data']=='real' and args.test_contract:
        if meta.get('test_contract_sha256')!=digest(args.test_contract):
            raise ValueError('Test contract must be registered during prepare, before model training')
        if read(root/'freeze.json')['manifest_sha256']!=digest(root/'manifest.json'):raise ValueError('Freeze mismatch')
        tc=read(args.test_contract)
        if not tc.get('fresh_unseen') or tc['dictionary_identity']!=meta['dictionary_identity']:
            raise ValueError('Need fresh held-out data labelled using this frozen training dictionary')
        if digest(Path(tc['dictionary'])/'manifest.json')!=tc['manifest_sha256']:raise ValueError('Test manifest changed')
        data,heldmeta=load_dictionary(tc['dictionary'],('holdout',))
        if heldmeta['dictionary_sha256']!=meta['dictionary_identity']:
            raise ValueError('Held-out labels do not identify the frozen training dictionary')
        records=data['holdout'];split='fresh_holdout'
        known={r['id'] for s in ('train','dev') for r in read_records(checked_file(root,meta,f'real_{s}.npz'))}
        if known & {r['id'] for r in records}:raise ValueError('Held-out activity IDs overlap source data')
    else:
        records=read_records(checked_file(root,meta,f'{dataset}_{split}.npz'))
    suffix=digest(args.test_contract)[:12] if args.test_contract else split
    target=out/f'evaluation_{suffix}'
    if target.exists():raise ValueError('Evaluation already exists; never overwrite test results')
    teacher=joblib.load(checked_file(root,meta,f'{dataset}_teacher.joblib')) if meta.get('teacher_diagnostics') else None
    ck=torch.load(out/'best.pt',map_location='cuda',weights_only=False)
    model=StateDiffusion(trial['backbone'],meta['width'],meta['n_states']).cuda().eval();model.load_state_dict(ck['model'])
    generated=[];torch.manual_seed(73000+trial['seed'])
    # Evaluate all eligible records, no random subsampling or duplicated activity bootstrap units.
    for r in records:
        rows=examples([r],trial['case'],meta['length'],seed=811,scale=ck['scale'],n_states=meta['n_states'])
        y=np.empty(len(r['y']),np.float32)
        for b in range(0,len(rows),32):
            chunk=rows[b:b+32];samples=model.sample(bundle(chunk,'cuda')).cpu().numpy()*ck['scale']
            for row,sample in zip(chunk,samples):y[row['start']:row['end']]=sample[:row['end']-row['start']]
        generated.append(y)
    conditional=trial['case'] in ('window_p','block_p','block_robust')
    result=evaluate(records,generated,teacher,ck['scale'],n_states=meta['n_states'],conditional=conditional)
    result.update(trial=trial,split=split,checkpoint_sha256=digest(out/'best.pt'),
                  teacher_quality=read(root/f'{dataset}_teacher_quality.json') if teacher is not None else None,
                  scope='conditional generation given requested state path and duration; not free activity planning',
                  confirmatory=split in ('test','fresh_holdout'),test_contract_sha256=digest(args.test_contract) if args.test_contract else None)
    target.mkdir();store_records(target/'samples.npz',[dict(id=r['id'],y=y,z=r['z']) for r,y in zip(records,generated)])
    write(target/'results.json',result)


def summarize(root):
    """Descriptive seed aggregation, never pool windows as independent trials."""
    contract(root);groups={};rows=[]
    for path in sorted((root/'trials').glob('*/evaluation_*/results.json')):
        value=read(path);t=value['trial']
        key=(t['phase'],t['backbone'],t['case'],t['data'],t['noise'],t['duration'],
             value['split'],value.get('test_contract_sha256'))
        groups.setdefault(key,[]).append(value)
    for key,values in groups.items():
        seeds=[v['trial']['seed'] for v in values]
        if len(seeds)!=len(set(seeds)):raise ValueError('Duplicate seed/evaluation group')
        metrics={}
        for metric in ('global_power_w1','energy_kwh_w1','peak_w1','mean_log_psd_l1',
                       'waveform_std_w1','reference_to_generated_distance','negative_fraction'):
            a=np.array([v['waveform_quality'][metric] for v in values])
            metrics[metric]=dict(mean=float(a.mean()),seed_sd=float(a.std(ddof=1)) if len(a)>1 else None)
        conditional_metrics={}
        if all(v['conditional_quality'] is not None for v in values):
            a=np.array([v['conditional_quality']['macro_state_power_w1'] for v in values])
            conditional_metrics['macro_state_power_w1']=dict(mean=float(a.mean()),seed_sd=float(a.std(ddof=1)) if len(a)>1 else None)
        rows.append(dict(phase=key[0],backbone=key[1],case=key[2],data=key[3],noise=key[4],duration=key[5],
                         split=key[6],test_contract_sha256=key[7],seeds=seeds,metrics=metrics,conditional_metrics=conditional_metrics))
    write(root/'summary.json',dict(groups=rows,complete_evaluations=sum(map(len,groups.values())),
          interpretation='Seed SD is not a confidence interval; compare matched noise/duration/backbone/split groups',
          primary_comparisons=['window_p - window_u','block_p - window_p','block_p - block_u',
                               'block_p - block_shuffle','block_robust - block_p']))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stage',choices=('prepare','train','freeze','evaluate','summary'))
    ap.add_argument('--campaign',required=True);ap.add_argument('--dictionary')
    ap.add_argument('--index',type=int);ap.add_argument('--length',type=int,default=128)
    ap.add_argument('--updates',type=int,default=10000);ap.add_argument('--pilot-updates',type=int,default=500)
    ap.add_argument('--validation-interval',type=int,default=500)
    ap.add_argument('--synthetic-activities',type=int,default=120);ap.add_argument('--test-contract')
    ap.add_argument('--teacher-diagnostics',action='store_true',help='Optional auxiliary classifier; never a training gate')
    args=ap.parse_args();require_slurm()
    if args.length<8 or min(args.updates,args.pilot_updates,args.validation_interval,args.synthetic_activities)<1:
        ap.error('Positive budgets and length >= 8 required')
    root=resolve_campaign(args.campaign)
    if args.stage=='prepare':
        if not args.dictionary:ap.error('--dictionary required')
        prepare(args,root)
    elif args.stage=='freeze':freeze(root)
    elif args.stage=='summary':summarize(root)
    else:
        if args.index is None or not 0<=args.index<126:ap.error('--index must be 0..125')
        (train if args.stage=='train' else score)(args,root)


if __name__=='__main__':main()
