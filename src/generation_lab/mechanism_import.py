"""Import an exact historical clustering result; never refit or merge states."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .common import read, write, digest, require_slurm
from .paths import resolve_campaign


def reconstruct(power, coordinates, lengths, labels):
    z=np.full(len(power),-1,np.int64)
    for start,n,label in zip(coordinates,lengths,labels):
        start,n=int(start),int(n)
        if start<0 or n<1 or start+n>len(power):raise ValueError('Out-of-bounds primitive')
        if np.any(z[start:start+n]!=-1):raise ValueError('Overlapping primitives')
        z[start:start+n]=int(label)
    return z


def export(source,output,tag='dpc_kmeans_k3'):
    source=Path(source).resolve();output=Path(output)
    if output.exists():raise ValueError('Use a new output dictionary')
    manifest=read(source/'run_manifest.json');steps=manifest['steps']
    artifacts=steps['time_clustering']['results'][tag]['artifacts']
    def path(value):
        p=(source/value).resolve()
        if not p.is_relative_to(source):raise ValueError('Source path escapes run')
        return p
    labs=np.load(path(artifacts['labels']),allow_pickle=False).reshape(-1)
    idx=np.load(path(artifacts['indices']),allow_pickle=False)
    kept=np.load(path(artifacts['kept_rows']),allow_pickle=False).reshape(-1)
    lengths=np.load(path(artifacts['seq_len']),allow_pickle=False).reshape(-1)
    seg=steps['time_segmentation']['artifacts']
    orig_idx=np.load(path(seg['indices']),allow_pickle=False)
    orig_lengths=np.load(path(seg['lengths']),allow_pickle=False).reshape(-1)
    x=np.load(path(seg['X']),mmap_mode='r',allow_pickle=False)
    if not (len(labs)==len(idx)==len(kept)==len(lengths)):raise ValueError('Misaligned clustering rows')
    if not np.array_equal(idx[:,:2],orig_idx[kept]) or not np.array_equal(idx[:,2],labs):
        raise ValueError('Cluster coordinates/labels disagree with segmentation')
    if not np.array_equal(lengths,orig_lengths[kept]):raise ValueError('Primitive lengths differ')
    classes=sorted(map(int,np.unique(labs)))
    if len(classes)!=3:raise ValueError('Requested source must contain exactly three classes')
    mapping={label:i+1 for i,label in enumerate(classes)}
    mapped=np.array([mapping[int(label)] for label in labs])
    folder=path(steps['extract_active_data']['artifacts']['segments_dir'])
    files=sorted(folder.glob('*.csv'))
    if len(files)!=int(steps['extract_active_data']['artifacts']['count']):raise ValueError('CSV inventory changed')
    if np.any(idx[:,0]<0) or np.any(idx[:,0]>=len(files)):raise ValueError('Unknown activity index')
    activities=[];source_hashes={};validated=0
    for aid,f in enumerate(files):
        a=np.genfromtxt(f,delimiter=',',names=True,dtype=None,encoding='utf-8')
        y=np.atleast_1d(a['power']).astype(np.float32);t=np.atleast_1d(a['timestamp']).astype(np.int64)
        if len(t)<1 or np.any(np.diff(t)<=0):raise ValueError(f'Nonmonotonic timestamps: {f}')
        rows=np.flatnonzero(idx[:,0]==aid)
        for row in rows:
            start=int(idx[row,1]);n=int(lengths[row])
            if not np.array_equal(x[int(kept[row]),:n,0],y[start:start+n]):
                raise ValueError(f'Raw waveform mismatch: {f.name}, row={row}')
            validated+=1
        z=reconstruct(y,idx[rows,1],lengths[rows],mapped[rows])
        activities.append((aid,f,y,t,z));source_hashes[f.name]=digest(f)
    ordered=sorted(activities,key=lambda a:(int(a[3][0]),a[0]))
    n=len(ordered);cut1=int(.6*n);cut2=int(.8*n)
    boundaries=[int(ordered[cut1][3][0]),int(ordered[cut2][3][0])]
    output.mkdir(parents=True);records=[];excluded=[];support={s:{} for s in ('train','dev','holdout')}
    for order,(aid,f,y,t,z) in enumerate(ordered):
        split='train' if order<cut1 else 'dev' if order<cut2 else 'holdout'
        if any(int(t[0])<=b+3600 and int(t[-1])>=b-3600 for b in boundaries):
            excluded.append(dict(activity=f.name,reason='split boundary 3600s guard'));continue
        valid=np.isfinite(y)&(z>0)
        # Preserve exact original observations; split gaps instead of inventing 6s samples.
        starts=np.r_[0,np.flatnonzero((np.diff(t)!=6)|(valid[1:]!=valid[:-1]))+1]
        ends=np.r_[starts[1:],len(y)]
        for piece,(lo,hi) in enumerate(zip(starts,ends)):
            if not valid[lo]:continue
            p=output/f'activity_{aid:04d}_support_{piece:04d}.npz'
            np.savez_compressed(p,power=y[lo:hi],P=z[lo:hi],timestamp=t[lo:hi])
            records.append(dict(activity_id=f'{manifest["run_id"]}/{f.name}',split=split,file=p.name,
                sha256=digest(p),source_csv=f.name,source_start=int(lo),source_end=int(hi)))
            for k in range(1,4):support[split][str(k)]=support[split].get(str(k),0)+int((z[lo:hi]==k).sum())
    provenance=dict(source_run=str(source),result=tag,label_map={str(k):v for k,v in mapping.items()},
                    manifest_sha256=digest(source/'run_manifest.json'),
                    artifacts={name:digest(path(value)) for name,value in artifacts.items()},
                    segmentation={name:digest(path(value)) for name,value in seg.items()},csv_sha256=source_hashes)
    identity=hashlib.sha256(json.dumps(provenance,sort_keys=True).encode()).hexdigest()
    write(output/'manifest.json',dict(k=3,records=records,dictionary_sha256=identity,source=provenance,
          evaluation_scope='exploratory historical full-pool pseudo-labels; not train-only discovery',
          sample_seconds=6,split='chronological 60/20/20 activities; 3600s boundary guards',
          gaps='split exact observations; no interpolation',state_merge=False))
    write(output/'audit.json',dict(source_activities=n,primitive_rows=len(labs),validated_raw_rows=validated,
          label_map=provenance['label_map'],split_state_points=support,excluded_activities=excluded,
          source_label_counts={str(k):int((labs==k).sum()) for k in classes},
          exported_records=len(records),split_activities={s:len({r['activity_id'] for r in records if r['split']==s}) for s in support}))
    print(json.dumps(read(output/'audit.json'),ensure_ascii=False),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source-run',required=True);ap.add_argument('--campaign',required=True)
    args=ap.parse_args();require_slurm();export(args.source_run,resolve_campaign(args.campaign))


if __name__=='__main__':main()
