"""Fork focused contracts and power-content audit into the new experiment."""
import ast,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'runs/20261015_multidataset_primglr_dtw_kmeans'
NEW=ROOT/'runs/20261016_public_all_activities_primglr_dtw_kmeans'
cfgfile=NEW/'configs/all_activities.json'
cfg=json.loads(cfgfile.read_text(encoding='utf-8'))
cfg['resources']['process_scope']='one fresh registered process per object; no reuse of partitioned fits'
cfg['stability_support']='at least 20 uncensored activity cores; descriptive flag only, never excludes a target'
cfg['randomness']=dict(prototype_and_final_kmeans_seeds=[0,1,2],prim_glr_internal_seed=0,silhouette_sampling_seed=42,manual_audit_sampling_seed=20261016)
cfgfile.write_text(json.dumps(cfg,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
code=(OLD/'scripts/check_public_adapter.py').read_text(encoding='utf-8')
start=code.index('    rows=[dict(row=i,activity_id=i//2')
end=code.index('    rr=rows[:2]',start)
code=code[:start]+'''    rows=[dict(row=i,activity_id=i//2,start=(i%2)*4,end=(i%2+1)*4,duration_seconds=32.) for i in range(8)]
    eligible=np.arange(len(rows)); proto=prototype_rows(rows,eligible,2,4)
    assert np.array_equal(proto,prototype_rows(rows,eligible,2,4))
    assert len(set(rows[i]['activity_id'] for i in proto))==4
    waves=np.stack([phase_wave(np.array([i,i+1.,i+2])) for i in range(8)])
    f=distance_features(waves,waves[proto],13); scaler=StandardScaler().fit(f)
    assert np.allclose(scaler.mean_,f.mean(axis=0))
    checks.append('all activities eligible for reproducible stratified prototypes and full-population scaling')
    tt=np.arange(0.,128.,8.); pp=np.zeros(len(tt)); pp[6:11]=100
    aa,_=extract_activities(tt,pp,cfg,8,0)
    assert len(aa)==1 and aa[0]['core_start']<64<aa[0]['core_end']
    assert aa[0]['ca']==0 and aa[0]['cb']==len(tt)
    assert 'split' not in aa[0]
    checks.append('activity across an arbitrary calendar boundary retained with all supported context')
'''+code[end:]
ast.parse(code);(NEW/'scripts/check_public_adapter.py').write_text(code,encoding='utf-8')
code=(OLD/'scripts/result_activity_audit.py').read_text(encoding='utf-8')
code=code.replace('out=execution();record_environment(out);source=a.source.resolve()','out=execution();source=a.source.resolve()')
code=code.replace("split=event['split'],",'')
code=code.replace('# Held-out review is selected by physical duration, not by detector confidence.','# Review is sampled across all accepted cores, weighted by physical duration.')
code=code.replace("test=[v for v in activities if v['split']=='test'];review=[]","population=activities;review=[]")
code=code.replace('if test:','if population:').replace('for v in test','for v in population').replace('e=test[int(rng.choice(len(test),p=probabilities))]','e=population[int(rng.choice(len(population),p=probabilities))]')
code=code.replace('test-core time-uniform','all-core time-uniform').replace('heldout_time_uniform_review.csv','all_activity_time_uniform_review.csv')
code=code.replace("save(out/'inputs.json'","save(out/'audit_inputs.json'")
assert "['split']" not in code
ast.parse(code);(NEW/'scripts/result_activity_audit.py').write_text(code,encoding='utf-8')
print(NEW)
