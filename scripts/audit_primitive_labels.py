"""Source-only frozen-teacher restoration and stratified artificial occlusion."""
import argparse
import os
from pathlib import Path
import sys
import numpy as np
PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from nilm_experiments.nilm_lab.common import read_json,write_json,digest


def boundary_f1(first,second,tolerance=2):
    candidates = sorted((abs(int(a)-int(b)),i,j) for i,a in enumerate(first) for j,b in enumerate(second)
                        if abs(int(a)-int(b))<=tolerance)
    used_a,used_b = set(),set()
    for _,i,j in candidates:
        if i not in used_a and j not in used_b:
            used_a.add(i);used_b.add(j)
    return 2*len(used_a)/(len(first)+len(second)) if len(first)+len(second) else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id',required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Use Slurm for source label audit')
    from src.framework.run_paths import run_directories
    root = run_directories(args.run_id, PROJECT)[0]
    dest = root/'occlusion_audit.json'
    if dest.exists():
        raise FileExistsError(dest)
    manifest = read_json(root/'run_manifest.json')
    def artifact(step,key):
        return root/manifest['steps'][step]['artifacts'][key]
    cfg = read_json(root/'nilm_data/request.json')['config']
    source = read_json(artifact('nilm_data','split_manifest'))
    from models.feature_extract.detsec_pc import PhyConstrainedDeTSEC,_extract
    from nilm_experiments.nilm_lab.activity_labels import detect_activities,activity_projection
    from models.time_segmentation.prim_glr import PrimGLRModel
    from src.steps.temporal_state_merge_step import merge_activity
    from src.utils.state_activity_mapping import contiguous_groups
    from sklearn.metrics import adjusted_rand_score,normalized_mutual_info_score
    frozen = read_json(artifact('feature_extract','model_config'))
    model = PhyConstrainedDeTSEC(frozen['n_features'],frozen['embed_dim'],frozen['nonneg_channels'],
        embed_proj=frozen['embed_proj'],nonneg_activation=frozen['nonneg_activation'])
    _extract(model,np.zeros((1,8,4),np.float32),np.array([8]),1)
    with np.load(artifact('feature_extract','weights')) as z:
        model.set_weights([z[f'arr_{i}'] for i in range(frozen['weight_count'])])
    with np.load(artifact('feature_extract','normalization')) as z:
        limits = z['percentile_limits']
    with np.load(artifact('state_sequence','dictionary')) as z:
        mean,scale,centers = z['mean'],z['scale'],z['centers']
    dt = cfg['nilm']['sample_seconds']
    cap = cfg['time_segmentation']['max_seg_len']
    merge = cfg['temporal_state_merge']
    def infer(t,y,rid):
        activities,intervals = detect_activities(t,y,cfg['extract_active_data'],rid,dt)
        samples,groups,starts,lengths = [],[],[],[]
        for activity,interval in zip(activities,intervals):
            signal = interval['data'].power.ffill().bfill().fillna(0.).to_numpy()
            cp = sorted({int(round(x)) for x in PrimGLRModel().train(signal) if 0<x<len(signal)})
            bounds = [0]+cp+[len(signal)]
            rows = []
            for a,b in zip(bounds[:-1],bounds[1:]):
                end = min(b,a+cap) if cap else b
                values = signal[a:end]
                rows.append(len(samples)); starts.append(a);lengths.append(len(values))
                samples.append(np.stack([values,values,values,np.zeros_like(values)],axis=1))
            groups.append(rows)
        if not samples:
            raise ValueError('No activities after source occlusion')
        tensor = np.zeros((len(samples),max(lengths),4),np.float32)
        for i,s in enumerate(samples):
            tensor[i,:len(s)] = s
        mask = np.arange(tensor.shape[1])[None] < np.array(lengths)[:,None]
        for j,(lo,hi) in enumerate(limits):
            tensor[:,:,j] = (np.clip(tensor[:,:,j],lo,hi)-lo)/(hi-lo+1e-7)
        tensor[~mask] = 0
        features = _extract(model,tensor,np.array(lengths),8)
        norm = (features-mean)/scale
        labels = ((norm[:,None]-centers[None])**2).sum(2).argmin(1)
        output = np.zeros(len(t),np.int64)
        blocks = []
        for activity,rows in zip(activities,groups):
            core_a,core_b = activity['core_start_index'],activity['core_end_index']
            output[core_a:core_b] = -100
            for continuous in contiguous_groups(rows,starts,lengths):
                merged = merge_activity(continuous,labels,starts,lengths,norm,features,
                    int(round(merge['min_block_seconds']/dt)),merge['enable_similar_merge'],merge['similar_feature_tol'])
                for block in merged:
                    a = max(core_a,activity['context_start_index']+block['start'])
                    b = min(core_b,activity['context_start_index']+block['end'])
                    if a<b:
                        output[a:b] = block['label']+1
                        blocks.append((a,b))
        return output,activities,blocks
    reports = []
    mapping_meta = read_json(artifact('state_sequence','mapping'))
    for index,r in enumerate(source['records']):
        with np.load(artifact('nilm_data','split_manifest').parent/r['file']) as z:
            t,y,ox = z['timestamp'],z['target'],z['mains_observed']
        with np.load(artifact('nilm_labels','metadata').parent/f'labels_{index}.npz') as z:
            original,valid = z['primitive'],z['valid']
        predicted,aa,blocks = infer(t,y,r['id'])
        support = valid & (original>0)
        parity = float((predicted[support]==original[support]).mean())
        if parity < .999:
            raise ValueError(f'Frozen teacher mapping restoration failed: {parity}')
        on,_,_=activity_projection(len(t),aa,np.isfinite(y)&ox)
        base_boundaries = np.flatnonzero((predicted[1:]!=predicted[:-1]) & (predicted[1:]>0) & (predicted[:-1]>0))+1
        nearby = np.zeros(len(y),bool)
        for b in base_boundaries:
            nearby[max(0,b-5):b+6]=True
        observed = np.isfinite(y)&ox
        strata = dict(low=(on==1)&(y<=20)&~nearby,high=(on==1)&(y>20)&~nearby,boundary=(on==1)&nearby)
        diagnostics = []
        for stratum,eligible in strata.items():
            for width in (1,2):
                possible = np.flatnonzero(eligible & observed & np.r_[observed[1:],False])
                rng = np.random.RandomState(42)
                candidates = rng.permutation(possible)
                chosen,blocked = [],np.zeros(len(y),bool)
                for i in candidates:
                    if not blocked[max(0,i-2):min(len(y),i+width+2)].any():
                        chosen.append(int(i));blocked[max(0,i-2):min(len(y),i+width+2)]=True
                    if len(chosen)>=200:
                        break
                if not chosen:
                    continue
                hidden = np.zeros(len(y),bool)
                for i in chosen:
                    hidden[i:i+width]=True
                altered = y.copy();altered[hidden]=np.nan
                current,_,_=infer(t,altered,r['id'])
                score = support & ~hidden & (current>0)
                new_boundaries=np.flatnonzero((current[1:]!=current[:-1])&(current[1:]>0)&(current[:-1]>0))+1
                diagnostics.append(dict(stratum=stratum,width=width,starts=chosen,hidden_points=int(hidden.sum()),
                    boundary_f1_12s=boundary_f1(base_boundaries,new_boundaries),
                    on_state_retention=float(score.sum()/max(1,(support&~hidden).sum())),
                    state_agreement=float((current[score]==original[score]).mean()) if score.any() else None,
                    adjusted_rand=float(adjusted_rand_score(original[score],current[score])) if score.any() else None,
                    nmi=float(normalized_mutual_info_score(original[score],current[score])) if score.any() else None))
                print('OCCLUSION',r['id'],stratum,width,diagnostics[-1]['state_agreement'],flush=True)
        reliable=[]
        for b in base_boundaries:
            if b>=2 and b+3<=len(y) and observed[b-2:b+3].all():
                reliable.append(int(b))
        reports.append(dict(record_id=r['id'],restoration_agreement=parity,
            reliable_local_boundaries=len(reliable),
            activities_with_two_reliable_boundaries=sum(sum(a['core_start_index']<b<a['core_end_index'] for b in reliable)>=2 for a in aa),
            diagnostics=diagnostics))
    write_json(dest,dict(source_only=True,test_opened=False,frozen_teacher=True,seed=42,
        dictionary_sha256=digest(artifact('state_sequence','dictionary')),reports=reports,
        note='Labels and boundaries rerun with frozen encoder/dictionary; not independent semantic truth or unknown-gap accuracy'))


if __name__=='__main__':
    main()
