"""Select lengths on train/dev support before model fitting; immutable trial matrix."""
import argparse
from collections import defaultdict
from pathlib import Path
from .paths import resolve_campaign
import numpy as np
from .common import digest, read, write, ranges, require_slurm


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--campaign',required=True);ap.add_argument('--gap',type=int,default=18)
    args=ap.parse_args();require_slurm();campaign=resolve_campaign(args.campaign);folder=campaign/f'dictionary_gap{args.gap}'
    meta=read(folder/'manifest.json');data={}
    for r in meta['records']:
        assert r['split'] in ('train','dev')
        path=folder/r['file']
        if digest(path)!=r['sha256']:raise ValueError('Dictionary export changed')
        with np.load(path) as z:data[r['activity_id']]=(r['split'],z['power'],z['P'])
    support={};indexes={}
    for length in [128,64,32,16,600,1200]:
        counts={s:{str(k):dict(windows=0,activities=set()) for k in range(1,6)} for s in ('train','dev')}
        index={s:[] for s in ('train','dev')};eligible={s:set() for s in ('train','dev')}
        for aid,(split,y,z) in data.items():
            for a,b in ranges(np.isfinite(y)&(z>0)):
                if b-a<length:continue
                eligible[split].add(aid)
                for start in range(a,b-length+1,length):
                    for k in np.unique(z[start:start+length]):
                        counts[split][str(k)]['windows']+=1;counts[split][str(k)]['activities'].add(aid)
                positions=list(range(a,b-length+1,max(1,length//2)))
                if positions[-1]!=b-length:positions.append(b-length)
                index[split].extend([aid,int(s)] for s in positions)
        support[str(length)]=dict(classes={s:{k:dict(windows=v['windows'],activities=len(v['activities'])) for k,v in cc.items()} for s,cc in counts.items()},
                                 eligible_activities={s:len(v) for s,v in eligible.items()},windows={s:len(v) for s,v in index.items()})
        indexes[length]=index
    candidates=[]
    for length in [128,64,32,16]:
        q=support[str(length)]['classes']
        if all(q[s][str(k)]['activities']>=({'train':5,'dev':3}[s]) and q[s][str(k)]['windows']>=({'train':30,'dev':10}[s]) for s in q for k in range(1,6)):
            candidates.append(length)
    long=600 if support['600']['eligible_activities']['train']>=20 and support['600']['eligible_activities']['dev']>=5 else None
    out=campaign/f'plan_gap{args.gap}';out.mkdir(exist_ok=False)
    write(out/'support.json',support)
    if not candidates:
        write(out/'gate.json',dict(passed=False,reason='No preregistered local length meets class/activity support'))
        raise RuntimeError('Data support gate failed; see support.json')
    local=candidates[0]
    for length in {local,long}-{None}:
        write(out/f'windows_{length}.json',indexes[length])
    matrices={}
    for phase,cases,seeds,updates in [('pilot',['U','P_context','Wz'],[101,102],500),
                                     ('development',['U','BP','S','P_context','W0','Wz'],[101,102],10000),
                                     ('formal',['U','BP','S','P_context','W0','Wz'],list(range(5)),10000),
                                     ('fourier',['U'],[101,102],10000),
                                     ('sensitivity',['U','P_context'],[101,102],10000)]:
        matrix=[]
        for backbone in ('diffusion_ts','conditional_unet_1d'):
            for case in cases:
                for seed in seeds:
                    length=long if case.startswith('W') else local
                    matrix.append(dict(backbone=backbone,case=case,seed=seed,length=length,updates=updates,
                        skipped_reason='No eligible long-window comparison support' if length is None else None))
        matrices[phase]=len(matrix);write(out/f'{phase}.json',matrix)
    # Choose width by parameter counts only, without model/data scores.
    from .models import Diffusion
    matching=[]
    for case,length in [('P_context',local),('Wz',long)]:
        base=32;error=None;target=None
        if length is not None:
            target=sum(p.numel() for p in Diffusion('diffusion_ts',length).parameters())
            choices={b:sum(p.numel() for p in Diffusion('conditional_unet_1d',length,unet_base=b).parameters()) for b in [16,24,32,48,64]}
            base=min(choices,key=lambda b:abs(choices[b]-target));error=abs(choices[base]-target)/target
        for seed in (101,102):
            matching.append(dict(backbone='conditional_unet_1d',case=case,seed=seed,length=length,updates=10000,
                unet_base=base,target_parameters=target,relative_parameter_error=error,
                matched_within_15pct=bool(error is not None and error<=.15),
                skipped_reason='No long support' if length is None else None))
    write(out/'parameter_match.json',matching)
    write(out/'gate.json',dict(passed=True,local_length=local,long_length=long,formal_holdout_unopened=True,
        dictionary_manifest_sha256=digest(folder/'manifest.json'),matrices=matrices,
        fourier_selection='development U(no Fourier) vs separate U(Fourier) on same fixed dev loss'))
    print('GATE PASS',local,long,matrices,flush=True)


if __name__=='__main__':main()
