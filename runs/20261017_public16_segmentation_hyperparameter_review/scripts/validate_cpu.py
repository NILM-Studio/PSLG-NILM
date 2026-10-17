"""Real input CPU execution, exact replay and CPU/GPU configuration consistency."""
from pathlib import Path
import time
import numpy as np
import review
from e2usd_cpu import run_cpu

plan = review.load(review.PLAN)
obj = next(o for o in plan['objects'] if o['object'] == 'ukdale_h2_washing_machine')
activity = obj['activities'][0]
with np.load(activity['npz']) as z:
    x = z['power'].copy()
params = dict(window=32, steps=20)
start = time.monotonic()
a, info = run_cpu(x, params, seed=0)
b, replay = run_cpu(x, params, seed=0)
assert a == b and info['device'] == 'cpu' and info['network']['cuda'] is False
old = sorted((review.EXP/'executions/review_ukdale_h2_washing_machine_e2usd').glob('*/analysis/records.json'))[-1]
row = next(r for r in review.load(old) if r['params'] == params and r['seed'] == 0 and r['activity_id'] == activity['activity_id'])
assert row['input_sha256'] == activity['npz_sha256']
gpu_network = dict(row['metadata']['network'])
cpu_network = dict(info['network'])
assert gpu_network.pop('cuda') is True and cpu_network.pop('cuda') is False
assert cpu_network == gpu_network
assert info['source_e2usd_py_sha256'] == row['metadata']['source_e2usd_py_sha256']
result = dict(status='passed', input_sha256=activity['npz_sha256'], cpu_replay_exact_cuts=True,
              same_author_source=True, same_model_and_training_parameters_except_cuda=True,
              cpu_cut_count=len(a), previous_gpu_cut_count=len(row['cuts']),
              elapsed_seconds=time.monotonic()-start,
              note='CPU and GPU floating point outputs need not have identical cuts; final E2USD gallery uniformly uses CPU.')
review.save(review.OUT/'analysis/validation.json', result)
print(result, flush=True)
