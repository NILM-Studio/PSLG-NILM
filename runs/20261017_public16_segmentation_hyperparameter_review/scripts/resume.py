"""Resume only missing jobs; retain completed/running executions and use two GPUs."""
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import review
from create_execution import register


def previous(ident, method):
    condition = 'review_' + ident + '_' + method.replace('-', '_')
    paths = sorted((review.EXP/'executions'/condition).glob('*/run.json'))
    return paths[-1].parent if paths else None


def launch(ident, method, gpu):
    existing = previous(ident, method)
    if existing:
        while review.load(existing/'run.json')['status'] in ('running', 'registered'):
            time.sleep(2)
        run = review.load(existing/'run.json')
        if run['exit_code'] == 0:
            return dict(object=ident, method=method, execution=str(existing), exit_code=0, reused_execution=True)
        raise RuntimeError(f'Existing failed worker requires explicit repair: {existing}')
    config = dict(plan=str(review.PLAN), plan_sha256=review.sha(review.PLAN), object=ident, method=method,
                  resource_revision='two_workers_each_on_GPU2_GPU6')
    execution = register(review.EXP.name, 'review_'+ident+'_'+method.replace('-', '_'), config)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu) if method == 'e2usd' else '',
               PYTHONPATH=str(review.ROOT/'runtime/segmentation/overlay_site'), OMP_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMBA_NUM_THREADS='1')
    code = subprocess.call([sys.executable, str(review.ROOT/'scripts/run_execution.py'), str(execution), '--',
                            sys.executable, str(Path(review.__file__).resolve()), '--object', ident, '--method', method],
                           cwd=review.ROOT, env=env)
    return dict(object=ident, method=method, execution=str(execution), exit_code=code, reused_execution=False)


def main():
    plan = review.load(review.PLAN)
    review.provenance()
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as cpu, \
         concurrent.futures.ThreadPoolExecutor(max_workers=2) as gpu2, \
         concurrent.futures.ThreadPoolExecutor(max_workers=2) as gpu6:
        futures = []
        for i, obj in enumerate(plan['objects']):
            for method in review.METHODS:
                gpu = 2 if i % 2 == 0 else 6
                pool = (gpu2 if gpu == 2 else gpu6) if method == 'e2usd' else cpu
                futures.append(pool.submit(launch, obj['object'], method, gpu))
        for f in concurrent.futures.as_completed(futures):
            results.append(f.result())
            review.save(review.EXP/'data/coordinator_progress.json',
                        dict(completed=len(results), total=80, jobs=results, coordinator_execution=str(review.OUT)))
            print('JOB_DONE', json.dumps(results[-1]), flush=True)
    assert all(r['exit_code'] == 0 for r in results), [r for r in results if r['exit_code']]
    qa = register(review.EXP.name, 'complete_grid_delivery_qa', dict(plan_sha256=review.sha(review.PLAN)))
    code = subprocess.call([sys.executable, str(review.ROOT/'scripts/run_execution.py'), str(qa), '--',
                            sys.executable, str(Path(review.__file__).resolve()), '--index', '--require-complete'], cwd=review.ROOT)
    assert code == 0
    review.save(review.OUT/'analysis/validation.json', dict(status='passed', jobs=results, figures=160, qa_execution=str(qa)))
    print('COMPLETE: 160 full-grid figures; awaiting human review', flush=True)


if __name__ == '__main__':
    main()
