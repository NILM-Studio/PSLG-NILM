"""Train-template free rollout, paired across backbones; never use held-out plans."""
import argparse
from pathlib import Path
from .paths import resolve_campaign
import time
import numpy as np
import torch
from .common import digest, read, write, require_slurm
from .models import Diffusion


def rollout(model,paths,seed,hard=False):
    n,total=paths.shape;l=model.length;h=max(2,l//8)
    if total<l:raise ValueError('Plan shorter than model')
    result=torch.zeros((n,total,1),device=paths.device)
    starts=list(range(0,total-l+1,l if hard else l-h))
    if starts[-1]!=total-l:starts.append(total-l)
    end=0
    for i,start in enumerate(starts):
        overlap=max(0,end-start)
        known=torch.zeros((n,l,1),device=paths.device);mask=torch.zeros_like(known,dtype=torch.bool)
        if overlap and not hard:known[:,:overlap]=result[:,start:start+overlap];mask[:,:overlap]=True
        part=model.sample(paths[:,start:start+l],known,mask,steps=50,seed=seed+1009*i)
        # For hard overlap at the final tail, discard duplicate prefix without conditioning.
        result[:,start+overlap:start+l]=part[:,overlap:]
        end=start+l
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--campaign',required=True);ap.add_argument('--index',type=int,required=True)
    args=ap.parse_args();require_slurm()
    root=resolve_campaign(args.campaign);plan=root/'plan_gap18';read(root/'freeze_gap18.json')
    trial=read(plan/'formal.json')[args.index];base=root/'trials_gap18/formal'/f'{args.index:03d}_{trial["backbone"]}_{trial["case"]}_s{trial["seed"]}'
    finished=read(base/'finished.json');out=root/'samples_gap18'/base.name;out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():return
    if finished['status']=='skipped':write(out/'complete.json',dict(status='skipped',trial=trial));return
    if not torch.cuda.is_available():raise RuntimeError('GPU required')
    gate=read(plan/'gate.json');total=gate['long_length'] or gate['local_length'];meta=read(root/'dictionary_gap18/manifest.json')
    windows=read(plan/f'windows_{total}.json')['train'];by_aid={}
    for aid,s in windows:by_aid.setdefault(aid,[]).append(s)
    rng=np.random.default_rng(39000+trial['seed']);aids=sorted(by_aid);records={}
    for r in meta['records']:
        if r['split']=='train':
            with np.load(root/'dictionary_gap18'/r['file']) as z:records[r['activity_id']]={k:z[k].copy() for k in ('power','P','BP','S')}
    chosen=[(int(aid),int(rng.choice(by_aid[aid]))) for aid in rng.choice(aids,100)]
    column=trial['case'] if trial['case'] in ('BP','S') else 'P'
    paths=np.stack([records[aid][column][s:s+total] for aid,s in chosen])
    ppaths=np.stack([records[aid]['P'][s:s+total] for aid,s in chosen])
    if trial['case'] in ('U','W0'):paths[:]=6
    ck=torch.load(base/'best.pt',map_location='cpu',weights_only=False)
    model=Diffusion(trial['backbone'],trial['length'],ck['fourier']).cuda().eval();model.load_state_dict(ck['ema'])
    started=time.time();generated=[];hard=[]
    for b in range(0,len(paths),8):
        z=torch.from_numpy(paths[b:b+8]).long().cuda()
        generated.append(rollout(model,z,seed=trial['seed']+b*100003).cpu().numpy()*meta['scale'])
        if trial['case']=='P_context':hard.append(rollout(model,z,seed=trial['seed']+b*100003,hard=True).cpu().numpy()*meta['scale'])
    generated=np.concatenate(generated)
    if not np.isfinite(generated).all():raise ValueError('Nonfinite rollout')
    fields=dict(generated_raw_watts=generated,requested_state=paths,primitive_plan=ppaths,source_template=np.asarray(chosen))
    if hard:fields['hard_splice_watts']=np.concatenate(hard)
    np.savez_compressed(out/'generated.npz',**fields)
    write(out/'complete.json',dict(status='completed',trial=trial,length=total,samples=100,
        generation_seconds=time.time()-started,negative_fraction=float((generated<0).mean()),
        checkpoint_sha256=digest(base/'best.pt'),plan_source='train_only',holdout_opened=False))


if __name__=='__main__':main()
