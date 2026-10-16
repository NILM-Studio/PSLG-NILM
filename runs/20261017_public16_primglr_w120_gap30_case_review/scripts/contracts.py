import os,pathlib,sys,json
import numpy as np
import pandas as pd
R=pathlib.Path(__file__).resolve().parents[3];sys.path.insert(0,str(R))
from models.extract_active_data.simple_threshold import SimpleThresholdDetector
from src.steps.extract_active_data_step import ExtractActiveDataStep
from src.framework.run_manifest import RunManifest
out=pathlib.Path(os.environ['PSLG_EXECUTION_DIR']);t=1700000000+np.array([.125,1.375,3.25,9.125,10.75,12.25]);p=np.array([0.,0.,10.,10.,0.,0.])
cfg=dict(threshold=5,t_drop=6,t_min_work=5,context_seconds=2,fs=1/5.033,context_by_time=True)
det=SimpleThresholdDetector('test',cfg);rows=det.detect(p,t);assert len(rows)==1
a=rows[0];assert a['start_time']==t[2] and a['end_time']==t[3]
assert a['core_end_exclusive']==t[4]
assert np.array_equal(a['data'].timestamp.to_numpy(),t[1:5])
np.save(out/'input.npy',np.column_stack([t,p]))
root=out/'analysis/main';root.mkdir(parents=True)
m=RunManifest(str(root/'run_manifest.json'),'contract','synthetic')
ExtractActiveDataStep(method='simple',appliance_name='synthetic',input_file=str(out/'input.npy'),**cfg).run(dict(log_root=str(root),manifest=m))
x=json.loads((root/'extract_active_data_simple/activities.json').read_text())['activities'][0]
assert x['core_start']==t[2] and x['core_end_exclusive']==t[4]
assert x['core_start_index']==2 and x['core_end_index']==4
z=pd.read_csv(root/'extract_active_data_simple/segments'/x['file'],float_precision='round_trip')
assert np.array_equal(z.timestamp.to_numpy(),t[1:5]) and np.array_equal(z.power.to_numpy(),p[1:5])
det=SimpleThresholdDetector('default',dict(threshold=5,t_drop=2,t_min_work=3,context_seconds=2,fs=1))
t=np.arange(20.);p=np.zeros(20);p[7:12]=10
x=det.detect(p,t);assert len(x)==1 and np.array_equal(x[0]['data'].timestamp.to_numpy(),t[5:14])
(out/'analysis/QA.json').write_text(json.dumps(dict(status='passed',checks=['fractional_core','actual_time_context','real_main_step','CSV_array_identity','legacy_regular_context']),indent=2))
print('PASS main-step fractional timestamp and context contracts')
