"""Final frozen H1 holdout distribution audit. Does not fit models on holdout."""
import argparse
from collections import defaultdict
from pathlib import Path
from .paths import resolve_campaign
import numpy as np
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import cdist
from .common import digest, read, write, ranges, require_slurm


def energy(x):return np.trapz(x,dx=6,axis=-1)/3600000


def summary(x,real):
    dx=np.diff(x,axis=1)/6;dr=np.diff(real,axis=1)/6
    def acf(v,lag):
        v=v-v.mean(1,keepdims=True)
        return (v[:,:-lag]*v[:,lag:]).mean(1)/(v*v).mean(1).clip(1e-8)
    specx=abs(np.fft.rfft(x-x.mean(1,keepdims=True),axis=1))**2
    specr=abs(np.fft.rfft(real-real.mean(1,keepdims=True),axis=1))**2
    return dict(power_w1=wasserstein_distance(x.ravel(),real.ravel()),
        rate_w1=wasserstein_distance(dx.ravel(),dr.ravel()),energy_kwh_w1=wasserstein_distance(energy(x),energy(real)),
        peak_w1=wasserstein_distance(x.max(1),real.max(1)),negative_fraction=float((x<0).mean()),
        acf_mean_difference={str(lag):float(abs(acf(x,lag).mean()-acf(real,lag).mean())) for lag in (1,2,4,8) if lag<x.shape[1]},
        mean_log_psd_l1=float(abs(np.log1p(specx).mean(0)-np.log1p(specr).mean(0)).mean()),
        unique_waveform_fraction=len(np.unique(x,axis=0))/len(x))


def hist_rates(x,edges):
    return np.stack([np.histogram(np.clip(np.diff(v)/6,edges[0],edges[-1]),edges)[0]/(len(v)-1) for v in x])


def boot_ci(x,r,xgroups,rgroups,edges,seed=14):
    # Cluster bootstrap with common seeds for paired comparisons; binned W1 is explicit.
    hx=hist_rates(x,edges);hr=hist_rates(r,edges)
    def grouped(h,g):return np.stack([h[g==v].mean(0) for v in np.unique(g)])
    hx,hr=grouped(hx,xgroups),grouped(hr,rgroups)
    centers=(edges[:-1]+edges[1:])/2;spacing=np.diff(centers)
    rng=np.random.default_rng(seed);values=[]
    for _ in range(1000):
        px=hx[rng.integers(len(hx),size=len(hx))].mean(0)
        pr=hr[rng.integers(len(hr),size=len(hr))].mean(0)
        values.append(float((abs(np.cumsum(px-pr))[:-1]*spacing).sum()))
    return np.asarray(values)


def retrieve(records,index,paths,length,seed,context):
    rng=np.random.default_rng(seed);pool=np.stack([records[a]['power'][s:s+length] for a,s in index])
    labels=np.stack([records[a]['P'][s:s+length] for a,s in index]);output=[];mismatch=0
    for path in paths:
        target=len(path);series=np.zeros(target);end=0
        starts=list(range(0,target-length+1,length))
        if starts[-1]!=target-length:starts.append(target-length)
        for s in starts:
            overlap=max(end-s,0);cost=(labels!=path[s:s+length]).mean(1);ids=np.flatnonzero(cost==cost.min())
            if context and end>0:
                j=int(ids[np.argmin(abs(pool[ids,overlap]-series[end-1]))])
            else:j=int(rng.choice(ids))
            mismatch+=int(cost[j]>0);series[s+overlap:s+length]=pool[j,overlap:];end=s+length
        output.append(series)
    return np.asarray(output),mismatch


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--campaign',required=True);args=ap.parse_args();require_slurm()
    root=resolve_campaign(args.campaign);freeze=read(root/'freeze_gap18.json');gate=read(root/'plan_gap18/gate.json')
    if freeze['gate_sha256']!=digest(root/'plan_gap18/gate.json'):raise ValueError('Freeze mismatch')
    out=root/'evaluation_gap18';out.mkdir(exist_ok=False)
    total=gate['long_length'] or gate['local_length'];prepared=read(root/'prepared/manifest.json');hold=[]
    for a in prepared['activities']:
        if a['split']!='holdout':continue
        with np.load(root/'prepared'/a['branches']['18']['file']) as z:
            y=z['power'].copy();t=z['timestamp'].copy();rt=z['raw_timestamp'].copy();rp=z['raw_power'].copy()
        for lo,hi in ranges(np.isfinite(y)):
            for s in range(lo,hi-total+1,total):
                mask=(rt>=t[s])&(rt<=t[s+total-1])&np.isfinite(rp)
                offsets=rt[mask]-t[s]
                hold.append((a['activity_id'],s,y[s:s+total],offsets,rp[mask]))
    if not hold:raise RuntimeError('No complete holdout windows; no final fidelity claim')
    byaid=defaultdict(list)
    for i,r in enumerate(hold):byaid[r[0]].append(i)
    rng=np.random.default_rng(77291);aids=sorted(byaid)
    ids=[int(rng.choice(byaid[a])) for a in rng.choice(aids,size=min(100,len(hold)),replace=True)]
    selected=[hold[i] for i in ids];real=np.stack([x[2] for x in selected]);rg=np.array([x[0] for x in selected])
    write(out/'holdout_support.json',dict(activities=len(aids),windows=len(hold),scoring_samples=len(ids),
        inference_supported=len(aids)>=5,freeze_sha256=digest(root/'freeze_gap18.json'),selected=[list(map(int,x[:2])) for x in selected]))
    trainmeta=read(root/'dictionary_gap18/manifest.json');train={}
    for r in trainmeta['records']:
        if r['split']=='train':
            with np.load(root/'dictionary_gap18'/r['file']) as z:train[r['activity_id']]={k:z[k].copy() for k in ('power','P')}
    index=read(root/f'plan_gap18/windows_{gate["local_length"]}.json')['train']
    long_index=read(root/f'plan_gap18/windows_{total}.json')['train']
    train_pool=np.stack([train[a]['power'][s:s+total] for a,s in long_index]).astype(float)
    def neighbor(v):
        # All training windows, bounded query batches; distance in W RMS.
        return np.concatenate([np.sqrt(cdist(v[b:b+16],train_pool,metric='sqeuclidean').min(1)/total)
                               for b in range(0,len(v),16)])
    real_neighbor=neighbor(real)
    # Training-fixed histogram bounds, with explicit reporting of clipped rate tails.
    tr=np.concatenate([np.diff(v['power'])[np.isfinite(v['power'][1:])&np.isfinite(v['power'][:-1])]/6 for v in train.values()])
    bound=max(float(np.max(abs(tr))),1)*2;edges=np.linspace(-bound,bound,257)
    results=[];boots={};retrieval_done=set()
    trials=read(root/'plan_gap18/formal.json')
    for i,trial in enumerate(trials):
        name=f'{i:03d}_{trial["backbone"]}_{trial["case"]}_s{trial["seed"]}'
        folder=root/'samples_gap18'/name;completion=read(folder/'complete.json')
        if completion['status']=='skipped':continue
        with np.load(folder/'generated.npz') as z:
            x=z['generated_raw_watts'][:,:,0].copy();groups=z['source_template'][:,0].copy();paths=z['primitive_plan'].copy()
            hard=z['hard_splice_watts'][:,:,0].copy() if 'hard_splice_watts' in z else None
        variants={trial['case']:x}
        if hard is not None:variants['P_hard']=hard
        if trial['seed'] not in retrieval_done:
            for context,label in [(False,'R0'),(True,'R1')]:
                value,mismatch=retrieve(train,index,paths,gate['local_length'],seed=39000+trial['seed'],context=context)
                variants[label]=value
                write(out/f'{label}_seed{trial["seed"]}_retrieval.json',dict(nonexact_plan_matches=mismatch))
            retrieval_done.add(trial['seed'])
        for label,value in variants.items():
            backbone='retrieval' if label in ('R0','R1') else trial['backbone']
            key=f'{backbone}/{label}/{trial["seed"]}'
            metrics=summary(value,real);projected=summary(np.maximum(value,0),real)
            nn=neighbor(value)
            metrics['nearest_training_window_rms_w_quantiles']=np.quantile(nn,[0,.05,.5,.95,1]).tolist()
            metrics['real_holdout_to_train_rms_w_quantiles']=np.quantile(real_neighbor,[0,.05,.5,.95,1]).tolist()
            metrics['near_exact_training_copy_fraction']=float((nn<1e-4).mean())
            # Requested state transition seams compared with all real power increments;
            # state-pair-specific real boundary scoring is supplied by frozen teacher job.
            joins=np.diff(paths,axis=1)!=0
            metrics['requested_boundary_jump_abs_mean_w']=float(abs(np.diff(value,axis=1))[joins].mean()) if joins.any() else None
            # Histogram overflow cannot disappear silently: report tail fractions.
            tail=float((abs(np.diff(value)/6)>bound).mean())
            bc=boot_ci(value,real,groups,rg,edges);boots[key]=bc
            # Independent raw-time reference, with identical timestamp readout of synthesis.
            raw_rate=[];generated_rate=[]
            for j,h in enumerate(selected):
                offsets,power=h[3],h[4]
                if len(offsets)<2:continue
                dt=np.diff(offsets);ok=(dt>0)&(dt<=18)
                gy=np.interp(offsets,np.arange(total)*6,value[j%len(value)])
                raw_rate.extend((np.diff(power)[ok]/dt[ok]).tolist())
                generated_rate.extend((np.diff(gy)[ok]/dt[ok]).tolist())
            metrics['raw_time_rate_w1']=wasserstein_distance(raw_rate,generated_rate) if raw_rate else None
            results.append(dict(key=key,metrics_raw_output=metrics,metrics_nonnegative_output=projected,
                rate_binned_bootstrap_ci95=np.quantile(bc,[.025,.975]).tolist(),rate_histogram_tail_fraction=tail))
        print('evaluated',name,flush=True)
    paired=[]
    for backbone in ('diffusion_ts','conditional_unet_1d'):
        for other in ('Wz','U','BP','S','P_hard','R1'):
            for seed in range(5):
                a=f'{backbone}/P_context/{seed}';b=f'{"retrieval" if other=="R1" else backbone}/{other}/{seed}'
                if a in boots and b in boots:
                    paired.append(dict(backbone=backbone,baseline=other,seed=seed,
                        binned_rate_w1_difference_ci95=np.quantile(boots[a]-boots[b],[.025,.975]).tolist()))
    write(out/'results.json',dict(results=results,paired=paired,holdout_house='H1',h2_opened=False,
        limitations=['bootstrap W1 uses 256 training-fixed bins; inspect tail fractions',
                     'pseudo-state classifier adherence and per-class coverage require separate frozen-teacher evaluation',
                     'raw-time curves use linear readout, not restored high-frequency truth']))


if __name__=='__main__':main()
