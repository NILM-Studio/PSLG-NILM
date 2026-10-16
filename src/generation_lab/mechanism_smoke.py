"""Small end-to-end synthetic verification, only on allocated compute nodes."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import numpy as np
import torch
from .common import require_slurm, write, digest
from .mechanism_data import synthetic
from .mechanism_v3 import prepare, train, freeze, score, summarize, trial_dir, read


def main():
    require_slurm()
    if not torch.cuda.is_available():raise RuntimeError('GPU required')
    project=Path(__file__).resolve().parents[2]
    destination=project/'runs/generation';destination.mkdir(parents=True,exist_ok=True)
    parent=Path(tempfile.mkdtemp(prefix='mechanism_v3_verification_',dir=destination))
    source=parent/'dictionary';source.mkdir();rows=[]
    for split in ('train','dev'):
        for i,r in enumerate(synthetic(split,True,2,n_states=3)):
            p=source/f'{split}_{i}.npz';np.savez_compressed(p,power=r['y'],P=r['z'])
            rows.append(dict(activity_id=r['id'],split=split,file=p.name,sha256=digest(p)))
    write(source/'manifest.json',dict(k=3,records=rows,dictionary_sha256='synthetic-verification-only'))
    args=SimpleNamespace(dictionary=str(source),length=32,updates=2,pilot_updates=2,
                         validation_interval=1,synthetic_activities=2,index=0,test_contract=None)
    root=parent/'campaign';prepare(args,root)
    for i in range(12):args.index=i;train(args,root)
    freeze(root)
    for i in (12,17,42,72):
        args.index=i;train(args,root);score(args,root)
    # Exercise idempotence, then an interrupted trial restored from resume.pt.
    train(args,root)
    trial=read(root/'plan.json')[72]
    finished=trial_dir(root,trial)/'finished.json'
    finished.rename(finished.with_name('previous_finished.json'))
    train(args,root)
    summarize(root)
    assert (trial_dir(root,trial)/'evaluation_test/results.json').exists()
    assert not list(root.glob('*teacher.joblib'))
    result=read(trial_dir(root,read(root/'plan.json')[17])/'evaluation_dev/results.json')
    assert result['conditional_quality'] is not None and result['auxiliary_classifier'] is None
    write(parent/'verification.json',dict(status='passed',synthetic=True,scientific_evidence=False,
          checks=['12 pilots','freeze','both formal backbones','mechanism training','sampling and state evaluation',
                  'idempotence','checkpoint resume','seed aggregation']))
    print('MECHANISM_V3_VERIFIED',parent,flush=True)


if __name__=='__main__':main()
