"""Raw timestamp interpolation with explicit provenance; full H1 only."""
import argparse
from collections import Counter
from pathlib import Path
from .paths import resolve_campaign
import os
import numpy as np
import pandas as pd
from .common import digest, ranges, require_slurm, write


def interpolate(t, y, grid, maximum):
    """Only finite/nonnegative brackets; never extrapolate or cross long gaps."""
    t, y, grid = np.asarray(t), np.asarray(y), np.asarray(grid)
    out = np.full(len(grid), np.nan, np.float64)
    exact = np.zeros(len(grid), bool)
    gap = np.full(len(grid), np.nan)
    right = np.searchsorted(t, grid)
    if not len(t):
        return out, exact, gap
    j = np.minimum(right, len(t)-1)
    exact = (right < len(t)) & (t[j] == grid) & np.isfinite(y[j]) & (y[j] >= 0)
    out[exact] = y[j[exact]]
    gap[exact] = 0
    inside = (~exact) & (right > 0) & (right < len(t))
    ix = np.flatnonzero(inside)
    r, l = right[ix], right[ix]-1
    width = t[r]-t[l]
    valid = (width > 0) & (width <= maximum) & np.isfinite(y[l]) & np.isfinite(y[r]) & (y[l] >= 0) & (y[r] >= 0)
    ix, r, l, width = ix[valid], r[valid], l[valid], width[valid]
    out[ix] = y[l]+(y[r]-y[l])*(grid[ix]-t[l])/width
    gap[ix] = width
    return out, exact, gap


def activity_spans(t, p, threshold=20., drop=150, minimum=180):
    ix = np.flatnonzero(np.isfinite(p) & (p >= threshold))
    if not len(ix):
        return []
    cuts = np.r_[0, np.flatnonzero(np.diff(t[ix]) > drop)+1, len(ix)]
    return [(int(ix[a]), int(ix[b-1])+1) for a,b in zip(cuts[:-1], cuts[1:])
            if t[ix[b-1]]-t[ix[a]] >= minimum]


def split_for(a, b, lo, hi):
    c1, c2 = lo+.6*(hi-lo), lo+.8*(hi-lo)
    if b <= c1-3600: return 'train'
    if a >= c1+3600 and b <= c2-3600: return 'dev'
    if a >= c2+3600: return 'holdout'
    return 'excluded'


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',required=True); ap.add_argument('--output',required=True)
    args=ap.parse_args(); require_slurm()
    root=Path(args.root)
    requested=Path(args.output)
    if requested.name != 'prepared':
        raise ValueError('--output must be <campaign>/prepared')
    out=resolve_campaign(str(requested.parent), root)/'prepared'
    out.mkdir(parents=True,exist_ok=False)
    source=root/'input/ukdale_washing_machine_full.csv'
    start_stat=source.stat()
    tt,yy=[],[]
    for chunk in pd.read_csv(source,chunksize=500000):
        t=pd.to_numeric(chunk.timestamp,errors='coerce').to_numpy()
        y=pd.to_numeric(chunk.power,errors='coerce').to_numpy()
        if not np.isfinite(t).all():raise ValueError('Invalid timestamp')
        tt.append(t.astype(np.int64)); yy.append(y.astype(np.float64))
    t,p=np.concatenate(tt),np.concatenate(yy); del tt,yy
    order=np.argsort(t,kind='stable'); t,p=t[order],p[order]
    starts=np.r_[0,np.flatnonzero(np.diff(t)>0)+1]
    ends=np.r_[starts[1:],len(t)]
    duplicates=len(t)-len(starts); conflicts=0
    for a,b in zip(starts[ends-starts>1],ends[ends-starts>1]):
        if not np.all(p[a:b]==p[a]):p[a]=np.nan;conflicts+=1
    t,p=t[starts],p[starts]
    p[~np.isfinite(p)|(p<0)]=np.nan
    source_hash=digest(source)
    if (source.stat().st_size,source.stat().st_mtime_ns)!=(start_stat.st_size,start_stat.st_mtime_ns):
        raise RuntimeError('Raw input changed')
    spans=activity_spans(t,p); entries=[]; monthly={}; qc={str(g):Counter() for g in (12,18,24)}
    masked={str(g):[] for g in (12,18,24)}; masked_total=0
    for aid,(a,b) in enumerate(spans):
        rt,rp=t[a:b],p[a:b]
        split=split_for(int(rt[0]),int(rt[-1])+1,int(t[0]),int(t[-1])+1)
        if split=='excluded':
            entries.append(dict(activity_id=aid,split=split,start=int(rt[0]),end=int(rt[-1])+1));continue
        grid=np.arange(((int(rt[0])+5)//6)*6,int(rt[-1])+1,6,dtype=np.int64)
        month=pd.Timestamp(int(rt[0]),unit='s',tz='UTC').strftime('%Y-%m')
        monthly.setdefault(month,Counter())[split]+=1
        entry=dict(activity_id=aid,split=split,start=int(rt[0]),end=int(rt[-1])+1,raw_points=len(rt),branches={})
        for maximum in (12,18,24):
            y,exact,gap=interpolate(rt,rp,grid,maximum)
            folder=out/f'gap{maximum}';folder.mkdir(exist_ok=True)
            path=folder/f'activity_{aid:06d}.npz'
            np.savez_compressed(path,timestamp=grid,power=y.astype(np.float32),exact_observed=exact,
                interpolated=np.isfinite(y)&~exact,invalid=~np.isfinite(y),bracket_gap_seconds=gap.astype(np.float32),
                raw_timestamp=rt,raw_power=rp.astype(np.float32))
            lens=[int(e-s) for s,e in ranges(np.isfinite(y))]
            entry['branches'][str(maximum)]=dict(file=str(path.relative_to(out)),sha256=digest(path),
                valid_points=int(np.isfinite(y).sum()),exact_points=int(exact.sum()),grid_points=len(y),
                continuous_lengths=lens)
            qc[str(maximum)][split+'_valid_points']+=int(np.isfinite(y).sum())
            qc[str(maximum)][split+'_activities']+=1
        # Artificial masking uses train/dev only; isolated targets never reappear in the interpolant.
        if split in ('train','dev') and len(rt)>4:
            selected=np.arange(2,len(rt)-1,20)
            keep=np.ones(len(rt),bool);keep[selected]=False;masked_total+=len(selected)
            for maximum in (12,18,24):
                recovered,_,_=interpolate(rt[keep],rp[keep],rt[selected],maximum)
                good=np.isfinite(recovered)&np.isfinite(rp[selected])
                for j in np.flatnonzero(good):
                    q=selected[j];slope=abs(rp[q+1]-rp[q-1])/max(rt[q+1]-rt[q-1],1)
                    masked[str(maximum)].append((aid,float(rp[q]),float(recovered[j]),float(slope)))
        entries.append(entry)
        if aid%100==0:print(f'prepared {aid+1}/{len(spans)} activities',flush=True)
    masking={}
    for key,rows in masked.items():
        arr=np.asarray(rows,dtype=float).reshape(-1,4)
        np.save(out/f'artificial_mask_gap{key}.npy',arr)
        err=np.abs(arr[:,1]-arr[:,2]) if len(arr) else np.array([])
        masking[key]=dict(attempted=masked_total,reconstructed=len(arr),mae=float(err.mean()) if len(err) else None,
            p95=float(np.quantile(err,.95)) if len(err) else None)
    manifest=dict(schema='activity_generation_prepared_v1',job_id=os.environ['SLURM_JOB_ID'],source=str(source),
        source_sha256=source_hash,raw_points=len(t),raw_extent=[int(t[0]),int(t[-1])],duplicates=duplicates,
        conflicting_duplicate_timestamps=conflicts,activity_rule=dict(threshold=20,drop_seconds=150,min_work_seconds=180),
        split_rule='full_time_60_20_20_guard3600s',activities=entries,
        monthly_activity_counts=monthly,qc=qc,artificial_masking=masking,holdout_waveform_scored=False)
    write(out/'manifest.json',manifest)
    write(out/'complete.json',dict(manifest_sha256=digest(out/'manifest.json'),activities=len(entries),qc=qc))
    print(json_summary(manifest),flush=True)


def json_summary(m):
    import json
    return json.dumps({k:m[k] for k in ('raw_points','raw_extent','qc','artificial_masking')},indent=2)


if __name__=='__main__':main()
