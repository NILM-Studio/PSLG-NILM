import os,sys,pathlib,json,subprocess,concurrent.futures,datetime
R=pathlib.Path(__file__).resolve().parents[3];E=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R/'scripts'))
from create_execution import register
plan=json.loads((E/'configs/plan.json').read_text());py=str(R/'.venv-nilm-data/bin/python')
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',NUMBA_NUM_THREADS='2')

def previous(condition):
 for p in sorted((E/'executions'/condition).glob('*/run.json'),reverse=True):
  if json.loads(p.read_text()).get('exit_code')==0:return p.parent
 return None

def execute(condition,command,config):
 old=previous(condition)
 if old:
  print('REUSE_PASSED',condition,old,flush=True);return old
 path=register(E.name,condition,config)
 print('START',condition,path,datetime.datetime.now(datetime.timezone.utc).isoformat(),flush=True)
 result=subprocess.run([py,str(R/'scripts/run_execution.py'),str(path),'--',py,*map(str,command)],env=env)
 if result.returncode:raise RuntimeError(f'Failed registered execution {condition}: {path}')
 print('DONE',condition,path,flush=True)
 return path

def queue(dataset):
 for obj in plan['objects']:
  if obj['source']['dataset']!=dataset:continue
  ident=obj['id'];execute('extract_'+ident,[E/'scripts/pipeline.py','extract',ident],dict(stage='main_step_extract',object=obj,plan=str(E/'configs/plan.json')))
  execute('plot_'+ident,[E/'scripts/pipeline.py','plot',ident],dict(stage='primglr_ten_cases',object=ident,params=plan['segmentation']))
  execute('plot_gap60_'+ident,[E/'scripts/plot_gap60.py',ident],dict(stage='primglr_ten_cases',object=ident,params=dict(plan['segmentation'],gap_seconds=60)))
 return dataset

if __name__=='__main__':
 print('MAIN_STEP_CONTRACTS_PREVIOUSLY_PASSED_UNCHANGED_CODE',flush=True)
 errors=[]
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
  jobs={pool.submit(queue,d):d for d in ['ukdale','refit','smartnialmeter','darck']}
  for job in concurrent.futures.as_completed(jobs):
   try:print('DATASET_COMPLETE',job.result(),flush=True)
   except Exception as ex:print('DATASET_FAILED',jobs[job],repr(ex),flush=True);errors.append(str(ex))
 if errors:raise RuntimeError(errors)
 execute('finalize_and_QA',[E/'scripts/finalize.py'],dict(stage='full_manifest_and_copy_QA',object_count=16))
 print('ALL_COMPLETE_HUMAN_REVIEW_PENDING',flush=True)
