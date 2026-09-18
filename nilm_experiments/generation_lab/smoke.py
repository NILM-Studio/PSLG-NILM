"""Bounded GPU integration test with synthetic fixtures, never scientific evidence."""
import os
from pathlib import Path
from .paths import resolve_campaign
import subprocess
import sys
import numpy as np
from .common import digest, read, write, require_slurm


def main():
    require_slurm()
    campaign=resolve_campaign(os.environ['CAMPAIGN'])/'integration_tests'/os.environ['SLURM_JOB_ID']
    folder=campaign/'dictionary_gap18';folder.mkdir(parents=True,exist_ok=False)
    records=[]
    for aid,split in [(0,'train'),(1,'dev')]:
        p=(100+20*np.sin(np.arange(64)/4)).astype(np.float32);labels=np.ones(64,np.int64)
        path=folder/f'{aid}.npz';np.savez(path,power=p,P=labels,BP=labels,S=labels)
        records.append(dict(activity_id=aid,split=split,file=path.name,sha256=digest(path)))
    write(folder/'manifest.json',dict(records=records,scale=200.))
    plan=campaign/'plan_gap18';write(plan/'gate.json',dict(passed=True,dictionary_manifest_sha256=digest(folder/'manifest.json')))
    write(plan/'windows_16.json',dict(train=[[0,i] for i in (0,16,32,48)],dev=[[1,i] for i in (0,16,32,48)]))
    matrix=[dict(backbone=b,case='P_context',seed=101,length=16,updates=4,skipped_reason=None) for b in ('diffusion_ts','conditional_unet_1d')]
    write(plan/'pilot.json',matrix)
    from .models import selftest
    selftest('cuda')
    for i in range(2):
        subprocess.run([sys.executable,'-m','nilm_experiments.generation_lab.train','--campaign',str(campaign),
                        '--phase','pilot','--index',str(i)],check=True)
        row=matrix[i];out=campaign/'trials_gap18/pilot'/f'{i:03d}_{row["backbone"]}_P_context_s101'
        assert read(out/'finished.json')['status']=='completed'
    # Full-window and multi-window rollout: exact length and seed reproducibility.
    import torch
    from .models import Diffusion
    from .sample import rollout
    m=Diffusion('conditional_unet_1d',16).cuda().eval();z=torch.ones((1,37),device='cuda',dtype=torch.long)
    a=rollout(m,z,123);b=rollout(m,z,123)
    assert a.shape==(1,37,1) and torch.equal(a,b) and torch.isfinite(a).all()
    from .evaluate import summary
    metrics=summary(np.ones((3,16))*100,np.ones((4,16))*100)
    assert metrics['power_w1']==0 and metrics['rate_w1']==0 and metrics['energy_kwh_w1']==0
    write(campaign/'complete.json',dict(passed=True,kind='synthetic_integration_test'))
    print('Both backbones: train/checkpoint/sample/rollout/metric integration passed',flush=True)


if __name__=='__main__':main()
