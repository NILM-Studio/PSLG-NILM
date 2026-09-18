"""Train-only PrimGLR/DeTSEC-PC dictionary with exact-length batches (no padding)."""
import argparse
from collections import defaultdict
from pathlib import Path
from .paths import resolve_campaign
import os
import numpy as np
from .common import digest, ranges, read, require_slurm, write


def shape(p):
    d=np.diff(p)/6
    return [float(p.mean()),float(p.std()),float(np.quantile(p,.9)-np.quantile(p,.1)),
            float((p[-1]-p[0])/max(6*(len(p)-1),1)),float(np.median(abs(d))) if len(d) else 0,
            float(np.log1p(6*(len(p)-1))),float(not len(d))]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--campaign',required=True);ap.add_argument('--gap',type=int,default=18)
    args=ap.parse_args();require_slurm()
    campaign=resolve_campaign(args.campaign);prepared=campaign/'prepared';out=campaign/f'dictionary_gap{args.gap}'
    out.mkdir(exist_ok=False)
    spec=read(prepared/'manifest.json')
    # Holdout is prepared but never opened in this stage.
    acts=[a for a in spec['activities'] if a['split'] in ('train','dev')]
    from models.time_segmentation.prim_glr import PrimGLRModel
    import contextlib
    pieces=[]; records={};segments=[]
    for a in acts:
        path=prepared/a['branches'][str(args.gap)]['file']
        if digest(path)!=a['branches'][str(args.gap)]['sha256']:raise ValueError('Changed prepared source')
        with np.load(path) as z:y=z['power'].copy();t=z['timestamp'].copy()
        records[a['activity_id']]=(a,y,t)
        for lo,hi in ranges(np.isfinite(y)):
            with open(os.devnull,'w') as null,contextlib.redirect_stdout(null):
                cp=PrimGLRModel().train(y[lo:hi])
            bounds=[lo]+[lo+int(q) for q in cp if 0<q<hi-lo]+[hi]
            group=[]
            for s,e in zip(bounds[:-1],bounds[1:]):
                if e<=s:raise ValueError('Invalid segmentation')
                group.append(len(segments));segments.append(y[s:e].copy())
                pieces.append(dict(activity_id=a['activity_id'],split=a['split'],start=int(s),end=int(e),
                                   support_start=int(lo),support_end=int(hi),start_timestamp=int(t[s])))
        if len(records)%50==0:print('segmented activities',len(records),flush=True)
    train=[i for i,p in enumerate(pieces) if p['split']=='train']
    train_aids=sorted({pieces[i]['activity_id'] for i in train})
    if len(train_aids)<10:raise ValueError('Too few train activities')
    cut=train_aids[max(1,int(.8*len(train_aids)))]
    internal_train=[i for i in train if pieces[i]['activity_id']<cut]
    internal_val=[i for i in train if pieces[i]['activity_id']>=cut]
    if max(map(len,segments))>20000:raise ValueError('Segment >20000 points; inspect, never silently truncate')
    import tensorflow as tf
    from models.feature_extract.detsec_pc import PhyConstrainedDeTSEC,total_loss
    gpus=tf.config.list_physical_devices('GPU')
    if not gpus:raise RuntimeError('TensorFlow GPU unavailable: refuse silent CPU discovery training')
    for gpu in gpus:tf.config.experimental.set_memory_growth(gpu,True)
    def limits(ids):return np.percentile(np.concatenate([segments[i] for i in ids]),[1,99])
    def batches(ids,lim,seed=None):
        buckets=defaultdict(list)
        for i in ids:buckets[len(segments[i])].append(i)
        rng=np.random.default_rng(seed);groups=[]
        for n,ii in sorted(buckets.items()):
            if seed is not None:rng.shuffle(ii)
            groups.extend(ii[j:j+16] for j in range(0,len(ii),16))
        if seed is not None:rng.shuffle(groups)
        for ii in groups:
            p=np.stack([segments[i] for i in ii]);p=(np.clip(p,*lim)-lim[0])/(lim[1]-lim[0]+1e-7)
            x=np.stack([p,p,p,np.zeros_like(p)],axis=-1).astype(np.float32)
            yield ii,tf.convert_to_tensor(x)
    def fit(ids,val,lim,epochs,select):
        tf.keras.backend.clear_session();tf.keras.utils.set_random_seed(42)
        model=PhyConstrainedDeTSEC(4,32,[0,1,2,3],embed_proj='relu')
        model(tf.zeros((1,8,4)),tf.ones((1,8)),tf.constant([8]),tf.zeros((1,8,1)))
        optimizer=tf.keras.optimizers.Adam(1e-4,clipnorm=1.)
        optimizer.build(model.trainable_variables)
        @tf.function(input_signature=[tf.TensorSpec([None,None,4],tf.float32),tf.TensorSpec([],tf.float32)])
        def step(x,ratio):
            mask=tf.ones(tf.shape(x)[:2]);lens=tf.fill([tf.shape(x)[0]],tf.shape(x)[1])
            keep=tf.cast(tf.random.uniform(tf.shape(x)[:2])[...,None]<ratio,tf.float32)
            with tf.GradientTape() as tape:
                _,f,b=model(x,mask,lens,keep);loss,_,_=total_loss(x,f,b,mask,lens,[0,1,2,3],.1)
            grads=tape.gradient(loss,model.trainable_variables)
            optimizer.apply_gradients([(g,v) for g,v in zip(grads,model.trainable_variables) if g is not None])
            return loss
        @tf.function(input_signature=[tf.TensorSpec([None,None,4],tf.float32)])
        def evaluate(x):
            mask=tf.ones(tf.shape(x)[:2]);lens=tf.fill([tf.shape(x)[0]],tf.shape(x)[1])
            _,f,b=model(x,mask,lens,tf.zeros_like(x[:,:,:1]))
            loss,_,_=total_loss(x,f,b,mask,lens,[0,1,2,3],.1)
            return loss
        best=float('inf');best_epoch=1;history=[]
        for epoch in range(1,epochs+1):
            errors=[]
            for ii,x in batches(ids,lim,seed=42+epoch):
                loss=float(step(x,tf.constant(max(0.,1-(epoch-1)/49),tf.float32)))
                if not np.isfinite(loss):raise ValueError('Nonfinite discovery loss')
                errors.append((loss,len(ii)))
            score=float(np.average([v for v,n in errors],weights=[n for v,n in errors]))
            if val:
                ee=[(float(evaluate(x)),len(ii)) for ii,x in batches(val,lim)]
                score=float(np.average([v for v,n in ee],weights=[n for v,n in ee]))
            history.append(dict(epoch=epoch,validation_z_only=score))
            print('discovery', 'select' if select else 'refit',epoch,score,flush=True)
            write(out/('selection_progress.json' if select else 'refit_progress.json'),history)
            if score<best:best,best_epoch=score,epoch
            if select and epoch-best_epoch>=10:break
        return model,best_epoch,history
    _,epochs,selection=fit(internal_train,internal_val,limits(internal_train),50,True)
    lim=limits(train);model,_,history=fit(train,[],lim,epochs,False)
    weights=model.get_weights();np.savez_compressed(out/'weights.npz',*weights)
    features=np.empty((len(segments),32),np.float32)
    for ii,x in batches(list(range(len(segments))),lim):
        emb=model.encode(x,tf.ones(tf.shape(x)[:2])).numpy();features[ii]=emb
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    scaler=StandardScaler().fit(features[train]);fn=scaler.transform(features)
    km=KMeans(5,n_init=30,random_state=42).fit(fn[train]);labels=km.predict(fn)
    from src.steps.temporal_state_merge_step import merge_activity
    starts=np.array([p['start'] for p in pieces]);lengths=np.array([len(s) for s in segments])
    groups=defaultdict(list)
    for i,p in enumerate(pieces):groups[(p['activity_id'],p['support_start'],p['support_end'])].append(i)
    blocks=[]
    for (aid,lo,hi),rows in groups.items():
        for b in merge_activity(rows,labels,starts,lengths,fn,features,15,True,2.):
            power=records[aid][1][b['start']:b['end']]
            blocks.append(dict(activity_id=aid,split=records[aid][0]['split'],start=b['start'],end=b['end'],
                state=int(b['label'])+1,mean=float(power.mean()),shape=shape(power),
                left_censored=b['start']==lo,right_censored=b['end']==hi))
    train_blocks=[i for i,b in enumerate(blocks) if b['split']=='train']
    means=np.array([b['mean'] for b in blocks])[:,None];shapes=np.array([b['shape'] for b in blocks])
    bp=KMeans(5,n_init=30,random_state=42).fit(means[train_blocks]);ss=StandardScaler().fit(shapes[train_blocks])
    sk=KMeans(5,n_init=30,random_state=42).fit(ss.transform(shapes[train_blocks]))
    bpl=bp.predict(means)+1;sl=sk.predict(ss.transform(shapes))+1
    byaid=defaultdict(list)
    for i,b in enumerate(blocks):b['BP']=int(bpl[i]);b['S']=int(sl[i]);byaid[b['activity_id']].append(b)
    labeled=[]
    for aid,(a,y,t) in records.items():
        arrays={k:np.full(len(y),-100,np.int64) for k in ('P','BP','S')}
        for b in byaid[aid]:
            for key in arrays:arrays[key][b['start']:b['end']]=b['state'] if key=='P' else b[key]
        path=out/f'activity_{aid:06d}.npz';np.savez_compressed(path,power=y,timestamp=t,**arrays)
        labeled.append(dict(activity_id=aid,split=a['split'],file=path.name,sha256=digest(path)))
    scale=max(1.,float(np.percentile(np.concatenate([segments[i] for i in train]),99)))
    np.savez_compressed(out/'dictionary.npz',limits=lim,mean=scaler.mean_,scale=scaler.scale_,centers=km.cluster_centers_,
        bp_centers=bp.cluster_centers_,shape_mean=ss.mean_,shape_scale=ss.scale_,shape_centers=sk.cluster_centers_)
    write(out/'blocks.json',blocks)
    meta=dict(records=labeled,scale=scale,k=5,gap_seconds=args.gap,selected_epochs=epochs,
        prepared_manifest_sha256=digest(prepared/'manifest.json'),dictionary_sha256=digest(out/'dictionary.npz'),
        feature_sha256=None,train_activities=len(train_aids),segments=len(segments),blocks=len(blocks),
        discovery_batching='exact_length_no_padding',selection_history=selection,refit_history=history,
        holdout_opened=False,source_code_hashes={p:digest(p) for p in ['models/time_segmentation/prim_glr.py',
        'models/feature_extract/detsec_pc.py','src/steps/temporal_state_merge_step.py',__file__]})
    write(out/'manifest.json',meta);write(out/'complete.json',dict(manifest_sha256=digest(out/'manifest.json')))


if __name__=='__main__':main()
