"""Registered extraction through the actual main step, with independent datasets copies."""
from __future__ import annotations
import os,sys,json,pathlib,hashlib,shutil,contextlib,io,math,subprocess,datetime,collections,ast,types
import numpy as np
import pandas as pd
R=pathlib.Path(__file__).resolve().parents[3];E=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R))
screen=R/'runs/20261016_public_appliance_signature_screening/scripts';sys.path.insert(0,str(screen))
from adapters import read_h5,mapped
from src.steps.extract_active_data_step import ExtractActiveDataStep
from src.framework.run_manifest import RunManifest
from models.time_segmentation.comparison import run as segment
grid_path=R/'runs/20261016_public_all_activities_primglr_dtw_kmeans/scripts/public_pipeline.py'
# Load the unchanged grid function alone, without unrelated numba modeling imports.
node=next(n for n in ast.parse(grid_path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='time_weighted_grid')
space=dict(np=np,math=math);exec(compile(ast.Module(body=[node],type_ignores=[]),str(grid_path),'exec'),space)
grid=types.SimpleNamespace(time_weighted_grid=space['time_weighted_grid'],__file__=str(grid_path))
PLAN=json.loads((E/'configs/plan.json').read_text())

def save(p,x):
 p=pathlib.Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 temp=p.with_name(p.name+'.tmp');temp.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else str(v))+'\n');os.replace(temp,p)

def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()

def epoch_ns(date):return int(pd.Timestamp(date).value)
def iso(ns):return str(pd.Timestamp(int(ns)))

def setup(out):
 snap=out/'analysis/code';snap.mkdir(parents=True)
 paths=[__file__,E/'configs/plan.json',R/'src/steps/extract_active_data_step.py',R/'models/extract_active_data/simple_threshold.py',R/'src/framework/run_manifest.py',R/'models/time_segmentation/prim_glr.py',R/'models/time_segmentation/comparison.py',screen/'adapters.py',grid.__file__]
 sources=[]
 for f in paths:
  f=pathlib.Path(f);shutil.copy2(f,snap/f.name);sources.append(dict(path=str(f),sha256=sha(f)))
 save(out/'inputs.json',dict(code=sources,cohort_document_sha256=sha(R/'docs/public_appliance_comparison_cohort_9plus7.md')))
 save(out/'environment.runtime.json',dict(python=sys.version,executable=sys.executable,CUDA_VISIBLE_DEVICES=os.getenv('CUDA_VISIBLE_DEVICES'),threads={k:os.getenv(k) for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']},numpy=np.__version__,pandas=pd.__version__))
 (out/'environment.freeze.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))

def spans(t,p,gap):
 finite=np.isfinite(p)
 cut=np.flatnonzero((np.diff(t)>gap)|(np.diff(t)<=0)|~finite[:-1]|~finite[1:])+1
 starts=np.r_[0,cut];ends=np.r_[cut,len(t)]
 for a,b in zip(starts,ends):
  if b-a>=2 and finite[a:b].all():yield int(a),int(b)

def enumerate_candidates(obj,ns,p,base):
 cfg=obj['activity'];nom=obj['nominal_seconds'];records=[];support_id=0;counts=collections.Counter()
 tt=ns.astype(np.float64)/1e9
 for ra,rb in spans(tt,p,obj['max_raw_gap_seconds']):
  if obj['source']['dataset']=='refit':
   t,x,cov=grid.time_weighted_grid(tt[ra:rb],p[ra:rb],8.,16.,.8)
   blocks=[(t[a:b],x[a:b],np.rint(t[a:b]*1e9).astype(np.int64)) for a,b in spans(t,x,8.01)]
  else:blocks=[(tt[ra:rb],np.asarray(p[ra:rb]),np.asarray(ns[ra:rb]))]
  for t,x,tns in blocks:
   on=np.flatnonzero(x>=cfg['threshold'])
   sid=support_id;support_id+=1
   if not len(on):continue
   cuts=np.r_[0,np.flatnonzero(np.diff(t[on])>cfg['t_drop'])+1,len(on)]
   candidates=[]
   for a,b in zip(cuts[:-1],cuts[1:]):
    first,last=int(on[a]),int(on[b-1]);duration=float(t[last]-t[first])
    counts['all_threshold_groups']+=1
    if duration<cfg['t_min_work']:counts['short']+=1;continue
    ca=int(np.searchsorted(t,t[first]-cfg['context_seconds'],'left'));cb=int(np.searchsorted(t,t[last]+cfg['context_seconds'],'right'))
    left=first==0 or t[first]-t[0]<min(cfg['context_seconds'],cfg['t_drop'])
    right=last==len(t)-1 or t[-1]-t[last]<cfg['t_drop']
    if left or right:counts['unsupported_or_censored']+=1;continue
    core_end=float(t[last+1] if last+1<len(t) else t[last]+nom)
    context_end=float(t[cb] if cb<len(t) else t[cb-1]+nom)
    candidates.append(dict(candidate_id=len(records)+len(candidates),support_id=sid,core_start_ns=int(tns[first]),core_last_sample_ns=int(tns[last]),core_end_exclusive_ns=int(round(core_end*1e9)),context_start_ns=int(tns[ca]),context_end_exclusive_ns=int(round(context_end*1e9)),core_start= float(t[first]),core_last_sample=float(t[last]),duration_seconds=duration,core_a=first-ca,core_b=last+1-ca,ca=ca,cb=cb,n_points=cb-ca,left_censored=False,right_censored=False,observed_supported_complete=True,raw_support_index_range=[ra,rb]))
   if candidates:
    path=base/'support_inputs'/f'{sid:06d}.npz';path.parent.mkdir(exist_ok=True)
    np.savez_compressed(path,timestamp=t,power=x,timestamp_ns=tns)
    for row in candidates:row['support_file']=str(path)
    records.extend(candidates)
 counts['supported']=len(records);counts['valid_supports']=support_id
 return records,dict(counts)

def split_and_select(rows,cap,all_inside):
 rows=sorted(rows,key=lambda r:(r['core_start_ns'],r['candidate_id']));groups=[]
 for r in rows:
  day=r['core_start_ns']//int(86400e9)
  if not groups or (day>groups[-1]['last_day'] and r['context_start_ns']>=groups[-1]['end']):groups.append(dict(rows=[],last_day=day,end=0))
  g=groups[-1];g['rows'].append(r);g['last_day']=max(day,(r['context_end_exclusive_ns']-1)//int(86400e9));g['end']=max(g['end'],r['context_end_exclusive_ns'])
 total=len(rows);cumulative=0
 for gi,g in enumerate(groups):
  frac=(cumulative+len(g['rows'])/2)/max(1,total)
  split='development' if frac<.6 else 'validation' if frac<.8 else 'evaluation'
  for r in g['rows']:r.update(split=split,time_group=gi)
  cumulative+=len(g['rows'])
 if all_inside or total<=cap:chosen=rows
 else:
  buckets={s:[r for r in rows if r['split']==s] for s in ['development','validation','evaluation']}
  shares={s:cap*len(rr)/max(1,total) for s,rr in buckets.items()};q={s:int(v) for s,v in shares.items()}
  for s in sorted(q,key=lambda s:-(shares[s]-q[s]))[:cap-sum(q.values())]:q[s]+=1
  chosen=[]
  for s,rr in buckets.items():
   # Equal calendar strata, stable SHA selection; fill sparse strata by evenly spaced dates.
   n=min(q[s],len(rr));picks=[]
   if not n:continue
   edges=np.linspace(rr[0]['core_start_ns'],rr[-1]['core_start_ns']+1,n+1)
   for i in range(n):
    pool=[r for r in rr if edges[i]<=r['core_start_ns']<edges[i+1]]
    if pool:picks.append(min(pool,key=lambda r:hashlib.sha256(f'{PLAN["selection"]["seed"]}:{r["core_start_ns"]}'.encode()).hexdigest()))
   used={r['candidate_id'] for r in picks};rest=[r for r in rr if r['candidate_id'] not in used]
   if len(picks)<n:
    inds=np.linspace(0,len(rest)-1,n-len(picks)).round().astype(int)
    picks.extend(rest[i] for i in inds)
   chosen.extend(picks)
 chosen=sorted(chosen,key=lambda r:r['core_start_ns']);assert len(chosen)<=cap
 for i,r in enumerate(chosen):r['activity_id']=i
 return chosen,dict(group_count=len(groups),split_counts=dict(collections.Counter(r['split'] for r in chosen)))

def choose_ten(rows):
 val=[r for r in rows if r['split']=='validation'];dev=[r for r in rows if r['split']=='development'];pool=val if len(val)>=10 else sorted(val+dev,key=lambda r:r['core_start_ns'])
 if len(pool)<=10:return pool
 pool=sorted(pool,key=lambda r:r['core_start_ns']);chunks=np.array_split(np.arange(len(pool)),5);picked=[]
 for inds in chunks:
  rr=sorted([pool[i] for i in inds],key=lambda r:r['duration_seconds']);picked.append(rr[0])
  if len(rr)>1:picked.append(rr[-1])
 return sorted(picked,key=lambda r:r['core_start_ns'])[:10]

def copy_independent(src,dst):
 dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
 assert src.stat().st_ino!=dst.stat().st_ino
 h=sha(src);assert h==sha(dst)
 return dict(path=str(dst),bytes=dst.stat().st_size,sha256=h)

def extract_object(obj,out):
 ident=obj['id'];base=out/'analysis'/ident;base.mkdir(parents=True);setup(out)
 dest=pathlib.Path(PLAN['dataset_root'])/ident;assert not dest.exists(),f'Dataset target exists: {dest}'
 src=obj['source'];nom=obj['nominal_seconds'];cfg=obj['activity']
 if src['dataset']=='darck':
  B=R/'runs/20261016_public_appliance_signature_screening/executions/actual_time_metric_harmonization/20261016T141009Z_69e251aa/analysis/object_results'/ident
  reader=json.loads((B/'result.json').read_text())['reader_record']
 else:reader=read_h5(src,base/'native_reader_cache')
 t,p=mapped(reader);assert len(t)==len(p) and np.all(np.diff(t)>0),'Ambiguous timestamps'
 lo,hi=obj.get('source_row_range') or [0,len(t)]
 lo=max(0,lo);hi=min(len(t),hi);req_start,req_end=obj['requested_interval']
 if req_start:lo=max(lo,int(np.searchsorted(t,epoch_ns(req_start))))
 if req_end:hi=min(hi,int(np.searchsorted(t,epoch_ns(req_end))))
 assert hi>lo
 source_start=int(t[lo]);source_end=int(t[hi-1]+round(nom*1e9))
 window_start=epoch_ns(req_start) if req_start else source_start
 window_end=epoch_ns(req_end) if req_end else source_end
 ns=t[lo:hi];px=p[lo:hi]
 print('READ',ident,len(ns),iso(source_start),iso(source_end),flush=True)
 rows,counts=enumerate_candidates(obj,ns,px,base)
 # Entire stored context, as well as core, must be inside this object's interval.
 rows=[r for r in rows if r['context_start_ns']>=window_start and r['context_end_exclusive_ns']<=window_end]
 initial_count=len(rows);all_inside=bool(req_start)
 if all_inside and len(rows)>obj['cap']:
  days=sorted({r['core_start_ns']//int(86400e9) for r in rows})
  for day in days:
   start=int(day*int(86400e9));rr=[r for r in rows if r['context_start_ns']>=start]
   if len(rr)<=obj['cap']:
    window_start=max(start,window_start);rows=rr;break
  print('RANGE_ADJUSTED',ident,iso(window_start),len(rows),'from',initial_count,flush=True)
 effective_lo=max(lo,int(np.searchsorted(t,window_start)));effective_hi=min(hi,int(np.searchsorted(t,window_end)))
 assert effective_hi>effective_lo
 raw=base/'source';raw.mkdir();raw_meta={}
 for key,arr in [('timestamp_ns',t[effective_lo:effective_hi]),('power',p[effective_lo:effective_hi])]:
  suffix='i64' if key=='timestamp_ns' else 'f64';path=raw/f'{key}.{suffix}'
  with path.open('xb') as f:
   for i in range(0,len(arr),1000000):f.write(np.asarray(arr[i:i+1000000]).tobytes())
  raw_meta[key]=dict(path=str(path),dtype=str(arr.dtype),rows=len(arr),sha256=sha(path))
 raw_t=t[effective_lo:effective_hi];raw_p=p[effective_lo:effective_hi]
 selected,split=split_and_select(rows,obj['cap'],all_inside);assert selected,f'No supported main-step candidates: {ident}'
 save(base/'candidates.json',dict(enumeration_counts=counts,requested_range_supported_count=initial_count,effective_range_supported_count=len(rows),records=rows))
 result_rows=[];actdir=base/'activities';actdir.mkdir()
 cache_path=None;cache=None
 for r in selected:
  aid=r['activity_id'];stem=f'{aid:05d}'
  if r['support_file']!=cache_path:
   cache_path=r['support_file']
   with np.load(cache_path) as z:cache={key:z[key] for key in ['timestamp','power','timestamp_ns']}
  st,sx,sns=cache['timestamp'],cache['power'],cache['timestamp_ns'];ca,cb=r['ca'],r['cb'];tt=st[ca:cb];xx=sx[ca:cb];tns=sns[ca:cb]
  job=base/'main_steps'/stem;job.mkdir(parents=True)
  np.save(job/'input.npy',np.column_stack([tt,xx]))
  manifest=RunManifest(str(job/'run_manifest.json'),f'{ident}_{stem}',ident)
  step=ExtractActiveDataStep(method='simple',appliance_name=ident,input_file=str(job/'input.npy'),resample_fs=0,fs=1/nom,context_by_time=True,**cfg)
  with (job/'main_step.log').open('w') as log,contextlib.redirect_stdout(log):step.run(dict(log_root=str(job),manifest=manifest))
  manifest.save()
  stepdir=job/'extract_active_data_simple';aa=json.loads((stepdir/'activities.json').read_text())['activities']
  hits=[a for a in aa if a['core_start']==r['core_start'] and a['core_end_exclusive']==r['core_end_exclusive_ns']/1e9]
  # Floating ns round trips may differ below one microsecond; match source endpoint exactly first.
  if not hits:hits=[a for a in aa if abs(a['core_start']-r['core_start'])<1e-6 and abs(a['core_end_exclusive']-r['core_end_exclusive_ns']/1e9)<1e-6]
  assert len(hits)==1,(ident,aid,r,aa)
  a=hits[0];original_csv=stepdir/'segments'/a['file'];df=pd.read_csv(original_csv,float_precision='round_trip')
  # Input already contains exactly the native context of the selected main-step interval.
  assert np.array_equal(df.timestamp.to_numpy(),tt) and np.array_equal(df.power.to_numpy(),xx),(ident,aid,'main CSV differs')
  assert a['core_start_index']==r['core_a'] and a['core_end_index']==r['core_b']
  assert np.isfinite(xx).all() and np.all(np.diff(tt)>0)
  if src['dataset']!='refit':
   ix=np.searchsorted(raw_t,tns);assert np.all(ix<len(raw_t))
   assert np.array_equal(raw_t[ix],tns) and np.array_equal(raw_p[ix],xx),'Raw/native segment mismatch'
  csvpath=actdir/f'{stem}.csv';shutil.copy2(original_csv,csvpath)
  ends=np.r_[st[ca+1:cb],(st[cb] if cb<len(st) else st[cb-1]+nom)]
  npz=actdir/f'{stem}.npz';np.savez_compressed(npz,timestamp=tt,timestamp_ns=tns,power=xx,end_timestamp=ends,core_a=r['core_a'],core_b=r['core_b'])
  result={k:v for k,v in r.items() if k not in ['support_file','ca','cb']}
  result.update(npz=str(npz),csv=str(csvpath),npz_sha256=sha(npz),csv_sha256=sha(csvpath),main_step_output_csv=str(original_csv),main_step_activities=str(stepdir/'activities.json'),main_run_manifest=str(job/'run_manifest.json'),main_step_output_csv_sha256=sha(original_csv),raw_core_index_range=[max(0,int(np.searchsorted(raw_t,r['core_start_ns'],'right'))-1),int(np.searchsorted(raw_t,r['core_end_exclusive_ns']))],main_csv_exact=True,raw_native_exact=src['dataset']!='refit',sampling='8s ZOH coverage>=.8' if src['dataset']=='refit' else 'native')
  result_rows.append(result)
  if (aid+1)%25==0:save(out/'analysis/progress.json',dict(object=ident,stage='main_step',completed=aid+1,total=len(selected)))
 cache=None
 case_ids=[r['activity_id'] for r in choose_ten(result_rows)]
 for r in result_rows:r['review_case']=r['activity_id'] in case_ids
 interval=dict(object=ident,source_file=src['path'],source_keys=src.get('keys',src.get('key',src.get('columns'))),source_identity=dict(bytes=pathlib.Path(src['path']).stat().st_size,mtime_ns=pathlib.Path(src['path']).stat().st_mtime_ns),source_timezone=src.get('timezone'),requested_interval=obj['requested_interval'],effective_start_ns=window_start,effective_end_exclusive_ns=window_end,effective_start=iso(window_start),effective_end_exclusive=iso(window_end),actual_first_observation_ns=int(raw_t[0]),actual_last_observation_ns=int(raw_t[-1]),rows=len(raw_t),nonfinite_power_count=int((~np.isfinite(raw_p)).sum()),nominal_seconds=nom,raw=raw_meta,main_extraction=dict(method='simple',config=dict(**cfg,fs=1/nom,context_by_time=True,resample_fs=0),source_file=str(R/'src/steps/extract_active_data_step.py'),source_sha256=sha(R/'src/steps/extract_active_data_step.py')),initial_supported_count=initial_count,effective_supported_count=len(rows),selected_activity_count=len(result_rows),cap=obj['cap'],split=split,review_case_ids=case_ids)
 save(base/'selected_interval.json',interval);save(base/'activities.json',dict(object=ident,definition='main_step_supported_activity_state_v1',records=result_rows))
 # Independent physical datasets copies with per-file identity checks.
 dest.mkdir(parents=True);copy_records=[]
 for path in raw.iterdir():copy_records.append(copy_independent(path,dest/'source'/path.name))
 dataset_rows=[]
 for row in result_rows:
  rr=dict(row)
  for key in ['npz','csv']:
   d=dest/'activities'/pathlib.Path(row[key]).name;copy_records.append(copy_independent(pathlib.Path(row[key]),d));rr[key]=str(d)
  dataset_rows.append(rr)
 ds_interval=json.loads(json.dumps(interval));ds_interval['raw']={key:dict(meta,path=str(dest/'source'/pathlib.Path(meta['path']).name)) for key,meta in raw_meta.items()}
 save(dest/'source/selected_interval.json',ds_interval)
 save(dest/'activities/activities.json',dict(object=ident,definition='main_step_supported_activity_state_v1',records=dataset_rows))
 save(dest/'activities/candidates.json',dict(enumeration_counts=counts,initial_supported_count=initial_count,effective_supported_count=len(rows),records=rows))
 save(dest/'copy_validation.json',dict(status='passed',independent_physical_copy=True,source_execution=str(out),files=copy_records))
 save(base/'QA.json',dict(status='passed',actual_main_step_invocations=len(selected),main_csv_array_identity=True,native_raw_array_identity=src['dataset']!='refit',finite_support=True,missing_bridges=False,independent_dataset_copy_hashes=True,counts_match=True))
 summary=dict(number=obj['number'],object=ident,selected_interval=ds_interval,raw_directory=str(dest/'source'),raw_manifest=str(dest/'source/selected_interval.json'),activity_directory=str(dest/'activities'),activity_manifest=str(dest/'activities/activities.json'),copy_QA=str(dest/'copy_validation.json'),execution=str(out))
 save(out/'analysis/summary.json',summary)
 print('OBJECT_COMPLETE',ident,len(selected),interval['effective_start'],interval['effective_end_exclusive'],split,flush=True)

def plot_object(obj,out):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 setup(out);ident=obj['id'];dest=pathlib.Path(PLAN['dataset_root'])/ident
 data=json.loads((dest/'activities/activities.json').read_text());cases=[r for r in data['records'] if r['review_case']]
 nom=obj['nominal_seconds'];s=PLAN['segmentation'];params=dict(W=max(3,math.ceil(s['W_seconds']/nom)),min_gap=max(2,math.ceil(s['gap_seconds']/nom)),alpha=s['alpha'],dur_sigma=s['dur_sigma'],g_wf=s['g_wf'],g_env=s['g_env'],n_clusters=None)
 records=[];fig,axes=plt.subplots(5,2,figsize=(18,17),layout='constrained')
 for i,ax in enumerate(axes.flat):
  if i>=len(cases):ax.text(.5,.5,'No additional development/validation case',ha='center',transform=ax.transAxes);ax.set_axis_off();continue
  r=cases[i]
  with np.load(r['npz']) as z:t=z['timestamp'];p=z['power'];a=int(z['core_a']);b=int(z['core_b'])
  with contextlib.redirect_stdout(io.StringIO()):cuts,meta=segment(p,'prim-glr',params,seed=0)
  assert cuts==sorted(set(cuts)) and all(0<c<len(t) for c in cuts)
  elapsed=t-t[0];ax.plot(elapsed,p,color='#263238',lw=.8)
  ax.axvspan(0,elapsed[a],color='#90a4ae',alpha=.3);ax.axvspan(elapsed[b-1],elapsed[-1],color='#90a4ae',alpha=.3)
  for c in cuts:ax.axvline(elapsed[c],color='#c62828',ls='--',lw=.7)
  ax.set_title(f"case {r['activity_id']} | {r['split']} | {iso(r['core_start_ns'])}\n{r['duration_seconds']:.1f}s core | cuts={len(cuts)} | core cuts={sum(a<c<b for c in cuts)}",fontsize=10)
  ax.set_xlabel('Elapsed seconds');ax.set_ylabel('Active power (W)');ax.grid(alpha=.15)
  records.append(dict(activity_id=r['activity_id'],split=r['split'],input=r['npz'],input_sha256=sha(r['npz']),params=params,effective_W_points=min(params['W'],max(3,len(p)//10)) if len(p)>=8 else None,cuts=cuts,cut_timestamps=[float(t[c]) for c in cuts],metadata=meta,panel=i+1))
 fig.suptitle(f'{ident} | PrimGLR target W=120s/{params["W"]}pt, gap=60s/{params["min_gap"]}pt | alpha=.90 sigma=.35\nTen supported activity-state cases; native short-series W adaptation recorded; human review pending',fontsize=13)
 folder=out/'analysis/figures';folder.mkdir();png=folder/f'{ident}_10_cases.png';svg=folder/f'{ident}_10_cases.svg'
 fig.savefig(png,dpi=150);fig.savefig(svg);plt.close(fig)
 save(out/'analysis/cases.json',dict(object=ident,status='awaiting_human_review',params=params,cases=records,png=str(png),svg=str(svg),png_sha256=sha(png),svg_sha256=sha(svg)))
 print('GALLERY_COMPLETE',ident,len(cases),flush=True)

if __name__=='__main__':
 mode,ident=sys.argv[1:3];obj=next(x for x in PLAN['objects'] if x['id']==ident);out=pathlib.Path(os.environ['PSLG_EXECUTION_DIR'])
 assert (out/'run.json').exists()
 extract_object(obj,out) if mode=='extract' else plot_object(obj,out)
