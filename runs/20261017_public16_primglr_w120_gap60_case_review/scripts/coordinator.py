import os,sys,json,pathlib,subprocess,concurrent.futures,shutil
from PIL import Image
import xml.etree.ElementTree as ET
R=pathlib.Path(__file__).resolve().parents[3];E=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R/'scripts'));from create_execution import register
sys.path.insert(0,str(E/'scripts'));from pipeline import sha,save
plan=json.loads((E/'configs/plan.json').read_text());py=str(R/'.venv-nilm-data/bin/python')
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',NUMBA_NUM_THREADS='2')
old={r['object']:r for r in json.loads(pathlib.Path(plan['comparison_source']).read_text())}
def queue(dataset):
 results=[]
 for obj in plan['objects']:
  if obj['source']['dataset']!=dataset:continue
  ident=obj['id'];ex=register(E.name,'plot_'+ident,dict(object=ident,segmentation=plan['segmentation'],comparison_source=plan['comparison_source']))
  subprocess.run([py,str(R/'scripts/run_execution.py'),str(ex),'--',py,str(E/'scripts/pipeline.py'),'plot',ident],env=env,check=True)
  z=json.loads((ex/'analysis/cases.json').read_text());before=old[ident]
  assert len(z['cases'])==10
  assert [c['activity_id'] for c in z['cases']]==[c['activity_id'] for c in before['cases']]
  for a,b in zip(z['cases'],before['cases']):
   assert a['input_sha256']==b['input_sha256']==sha(a['input'])
   assert a['split']==b['split'] and a['split']!='evaluation'
   assert {k:v for k,v in a['params'].items() if k!='min_gap'}=={k:v for k,v in b['params'].items() if k!='min_gap'}
  results.append(z);print('DONE',ident,flush=True)
 return results
results=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
 for result in pool.map(queue,['ukdale','refit','smartnialmeter','darck']):results.extend(result)
results.sort(key=lambda z:next(o['number'] for o in plan['objects'] if o['id']==z['object']))
out=pathlib.Path(os.environ['PSLG_EXECUTION_DIR']);report=E/'reports'/('final_'+out.name);(report/'figures').mkdir(parents=True)
lines=['# PrimGLR W=120s、gap=60s：16对象十例图审','','每对象与gap=30秒版使用相同10个活动片段；仅gap改为60秒。黑线为功率，灰色为上下文，红色虚线为切点。短片段实际W按原生规则缩小，逐例见figure_manifest.json。参数待人工审核。','','|对象|gap=60秒|gap=30秒（服务器）|','|---|---|---|']
for z in results:
 for kind in ['png','svg']:
  src=pathlib.Path(z[kind]);dst=report/'figures'/src.name;shutil.copy2(src,dst);assert sha(src)==sha(dst)
  if kind=='png':
   with Image.open(dst) as im:im.verify()
  else:assert ET.parse(dst).getroot().tag.endswith('svg')
  z['review_'+kind]=str(dst)
 lines.append('|'+z['object']+'|[十例图](figures/'+pathlib.Path(z['png']).name+')|[十例图]('+old[z['object']]['png']+')|')
save(report/'figure_manifest.json',results)
save(report/'QA.json',dict(status='passed',object_count=len(results),case_count=sum(len(z['cases']) for z in results),same_cases_and_input_hashes=True,only_segmentation_change='gap_seconds:30->60',png_svg_readable=True,evaluation_cases=0))
(report/'index.md').write_text('\n'.join(lines)+'\n')
save(E/'data/delivery.json',dict(report=str(report/'index.md'),figures=str(report/'figures'),object_count=16,case_count=160))
with (R/'runs/EXPERIMENT_LOG.md').open('a') as f:f.write('\n- '+E.name+'：16对象160例gap60图审完成；与gap30输入逐例哈希相同，参数待人工审核。\n')
print('DELIVERY',report,flush=True)
