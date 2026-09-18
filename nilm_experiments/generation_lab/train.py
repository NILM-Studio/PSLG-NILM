"""Slurm-only immutable/resumable paired generative trials; no holdout loading."""
import argparse
import copy
import os
from pathlib import Path
from .paths import resolve_campaign
import random
import time
import numpy as np
import torch
from .common import digest, read, write, require_slurm
from .models import Diffusion


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--campaign',required=True);ap.add_argument('--gap',type=int,default=18)
    ap.add_argument('--phase',choices=['pilot','development','formal','fourier','sensitivity','parameter_match'],required=True)
    ap.add_argument('--index',type=int,required=True);args=ap.parse_args();require_slurm()
    if not torch.cuda.is_available():raise RuntimeError('GPU allocation required')
    root=resolve_campaign(args.campaign);plan=root/f'plan_gap{args.gap}';gate=read(plan/'gate.json')
    if not gate['passed']:raise RuntimeError('Data gate failed')
    trial=read(plan/f'{args.phase}.json')[args.index]
    folder=root/f'dictionary_gap{args.gap}';meta=read(folder/'manifest.json')
    if digest(folder/'manifest.json')!=gate['dictionary_manifest_sha256']:raise ValueError('Changed dictionary')
    out=root/f'trials_gap{args.gap}'/args.phase/f'{args.index:03d}_{trial["backbone"]}_{trial["case"]}_s{trial["seed"]}'
    out.mkdir(parents=True,exist_ok=True)
    if (out/'finished.json').exists():return
    started=time.time()
    if trial['skipped_reason']:
        write(out/'finished.json',dict(status='skipped',reason=trial['skipped_reason'],trial=trial));return
    fourier=args.phase=='fourier'
    if args.phase=='formal':
        frozen=read(root/f'freeze_gap{args.gap}.json');fourier=frozen['fourier']
        if frozen['gate_sha256']!=digest(plan/'gate.json'):raise ValueError('Freeze mismatch')
    elif args.phase in ('sensitivity','parameter_match'):
        fourier=read(root/'freeze_gap18.json')['fourier']
    seed=trial['seed'];torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    torch.set_num_threads(4)
    length=trial['length'];case=trial['case'];backbone=trial['backbone'];records={}
    for r in meta['records']:
        assert r['split'] in ('train','dev')
        with np.load(folder/r['file']) as z:
            labels=z['P' if case in ('P_context','Wz') else case if case in ('BP','S') else 'P'].copy()
            if case in ('U','W0'):labels[:]=6
            records[r['activity_id']]=(z['power'].copy()/meta['scale'],labels)
    windows=read(plan/f'windows_{length}.json')
    byaid={}
    for row in windows['train']:byaid.setdefault(row[0],[]).append(row)
    aids=sorted(byaid)
    def batch(rows):
        yy=[];zz=[]
        for aid,start in rows:
            y,z=records[aid];yy.append(y[start:start+length]);zz.append(z[start:start+length])
        y=torch.from_numpy(np.stack(yy)[...,None]).float().cuda()
        z=torch.from_numpy(np.stack(zz)).long().cuda()
        if not torch.isfinite(y).all() or (z<0).any():raise ValueError('Invalid training window')
        return y,z
    model=Diffusion(backbone,length,fourier,unet_base=trial.get('unet_base',32)).cuda();ema=copy.deepcopy(model).eval()
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4)
    source_hashes={str(p):digest(p) for p in Path(__file__).parent.glob('*.py')}
    source_hashes.update({str(p):digest(p) for p in (Path(__file__).parents[1]/'third_party/diffusion_ts').glob('*.py')})
    source_hashes.update({str(p):digest(p) for p in (Path(__file__).parent/'backbones').glob('*.py')})
    policy_file = Path(resolve_campaign.__code__.co_filename)
    source_hashes[str(policy_file)] = digest(policy_file)
    provenance=dict(trial=trial,phase=args.phase,fourier=fourier,dictionary_hash=gate['dictionary_manifest_sha256'],
        plan_hash=digest(plan/f'{args.phase}.json'),code_hashes=source_hashes,job_id=os.environ['SLURM_JOB_ID'],
        parameters=sum(p.numel() for p in model.parameters()),scale=meta['scale'])
    first=0;best=float('inf');stale=0;history=[]
    if (out/'resume.pt').exists():
        ck=torch.load(out/'resume.pt',map_location='cuda',weights_only=False)
        if ck['provenance']['code_hashes']!=source_hashes or ck['provenance']['dictionary_hash']!=provenance['dictionary_hash']:
            raise ValueError('Cannot resume changed code or dictionary')
        model.load_state_dict(ck['model']);ema.load_state_dict(ck['ema']);optimizer.load_state_dict(ck['optimizer'])
        first,best,stale,history=ck['step'],ck['best'],ck['stale'],ck['history']
    write(out/'run.json',provenance)
    # Fixed deterministic dev sample across backbones/conditions at the same length.
    val_rng=np.random.default_rng(7281);val_rows=windows['dev']
    val_ids=val_rng.choice(len(val_rows),size=min(256,len(val_rows)),replace=False)
    def validate():
        values=[];original=ema.fourier;ema.fourier=False
        with torch.no_grad():
            for b in range(0,len(val_ids),32):
                y,z=batch([val_rows[i] for i in val_ids[b:b+32]])
                values.append((float(ema.loss(y,z,900000+b)),len(y)))
        ema.fourier=original
        return float(np.average([v for v,n in values],weights=[n for v,n in values]))
    optimizer.zero_grad(set_to_none=True)
    for step in range(first+1,trial['updates']+1):
        rng=np.random.default_rng(seed*1000003+step)
        rows=[byaid[aid][rng.integers(len(byaid[aid]))] for aid in rng.choice(aids,size=32)]
        y,z=batch(rows);model.train();optimizer.zero_grad(set_to_none=True)
        loss=model.loss(y,z,seed*1000003+step)
        if not torch.isfinite(loss):raise ValueError('Nonfinite generator loss')
        loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not torch.isfinite(grad):raise ValueError('Nonfinite gradients')
        optimizer.step()
        with torch.no_grad():
            for e,p in zip(ema.parameters(),model.parameters()):e.lerp_(p,1-.995)
        if step%100==0:print(args.phase,args.index,step,float(loss),flush=True)
        if step%500==0 or step==trial['updates']:
            score=validate();history.append(dict(step=step,train_loss=float(loss),dev_x0_loss=score))
            if score<best:
                best=score;stale=0
                torch.save(dict(ema=ema.state_dict(),trial=trial,fourier=fourier,scale=meta['scale'],provenance=provenance),out/'best.pt')
            else:stale+=1
            torch.save(dict(model=model.state_dict(),ema=ema.state_dict(),optimizer=optimizer.state_dict(),
                step=step,best=best,stale=stale,history=history,provenance=provenance),out/'resume.pt')
            write(out/'history.json',history)
            if stale>=5:break
    # Training-time engineering sample only; not a fidelity score or holdout result.
    ck=torch.load(out/'best.pt',map_location='cuda',weights_only=False);ema.load_state_dict(ck['ema'])
    y,z=batch([val_rows[i] for i in val_ids[:8]])
    sample_started=time.time();gen=ema.sample(z,steps=50,seed=seed+73);torch.cuda.synchronize()
    if not torch.isfinite(gen).all():raise ValueError('Nonfinite generation')
    np.savez_compressed(out/'engineering_samples.npz',generated_raw_watts=gen.cpu().numpy()*meta['scale'],
        requested_state=z.cpu().numpy(),conditional_dev_reference=y.cpu().numpy()*meta['scale'],
        scope=np.asarray('engineering conditional diagnostic; not free generation efficacy'))
    write(out/'finished.json',dict(status='completed',trial=trial,steps=history[-1]['step'],fourier=fourier,
        best_dev_x0_loss=best,parameters=provenance['parameters'],wall_seconds=time.time()-started,
        sample_seconds=time.time()-sample_started,max_cuda_memory_bytes=torch.cuda.max_memory_allocated(),
        negative_sample_fraction=float((gen<0).float().mean()),holdout_opened=False))


if __name__=='__main__':main()
