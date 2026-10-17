"""Complete the user-requested CPU gallery, preserving existing CPU executions."""
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import review
from create_execution import register


def launch(ident, method):
    condition = ('review_cpu_' if method == 'e2usd' else 'review_') + ident + '_' + method.replace('-', '_')
    paths = sorted((review.EXP/'executions'/condition).glob('*/run.json'))
    if paths:
        existing = paths[-1].parent
        while review.load(existing/'run.json')['status'] in ('running', 'registered'):
            time.sleep(2)
        assert review.load(existing/'run.json')['exit_code'] == 0, existing
        return dict(object=ident, method=method, execution=str(existing), exit_code=0, reused=True)
    execution = register(review.EXP.name, condition,
                         dict(plan_sha256=review.sha(review.PLAN), object=ident, method=method, backend='cpu'))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PSLG_E2USD_DEVICE='cpu',
               PYTHONPATH=str(review.ROOT/'runtime/segmentation/overlay_site'), OMP_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMBA_NUM_THREADS='1')
    code = subprocess.call([sys.executable, str(review.ROOT/'scripts/run_execution.py'), str(execution), '--',
                            sys.executable, str(Path(review.__file__).resolve()), '--object', ident, '--method', method],
                           cwd=review.ROOT, env=env)
    return dict(object=ident, method=method, execution=str(execution), exit_code=code, reused=False)


def main():
    plan = review.load(review.PLAN)
    review.provenance()
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        fs = [pool.submit(launch, o['object'], m) for o in plan['objects'] for m in review.METHODS]
        for f in concurrent.futures.as_completed(fs):
            results.append(f.result())
            review.save(review.EXP/'data/coordinator_progress.json',
                        dict(completed=len(results), total=80, jobs=results, backend='cpu', coordinator_execution=str(review.OUT)))
            print('JOB_DONE', json.dumps(results[-1]), flush=True)
    assert all(r['exit_code'] == 0 for r in results), [r for r in results if r['exit_code']]
    qa = register(review.EXP.name, 'complete_grid_delivery_qa', dict(plan_sha256=review.sha(review.PLAN), backend='cpu'))
    code = subprocess.call([sys.executable, str(review.ROOT/'scripts/run_execution.py'), str(qa), '--',
                            sys.executable, str(Path(review.__file__).resolve()), '--index', '--require-complete'], cwd=review.ROOT)
    assert code == 0
    review.save(review.OUT/'analysis/validation.json', dict(status='passed', jobs=results, figures=160, qa_execution=str(qa), backend='cpu'))
    print('COMPLETE: 160 CPU full-grid figures; awaiting human review', flush=True)


if __name__ == '__main__':
    main()
