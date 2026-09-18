"""Check completed stages and freeze one common Fourier setting before formal fits."""
import argparse
from pathlib import Path
from .paths import resolve_campaign
import numpy as np
from .common import digest, read, require_slurm, write


def collect(root,gap,phase):
    rows=read(root/f'plan_gap{gap}'/f'{phase}.json');result=[]
    for i,row in enumerate(rows):
        path=root/f'trials_gap{gap}'/phase/f'{i:03d}_{row["backbone"]}_{row["case"]}_s{row["seed"]}'/'finished.json'
        r=read(path)
        if r['status'] not in ('completed','skipped'):raise ValueError('Unfinished trial')
        result.append(r)
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--campaign',required=True);ap.add_argument('--gap',type=int,default=18)
    ap.add_argument('--stage',choices=['pilot','freeze','summary'],required=True)
    a=ap.parse_args();require_slurm();root=resolve_campaign(a.campaign)
    if a.stage=='pilot':
        rows=collect(root,a.gap,'pilot')
        if not all(np.isfinite(r['best_dev_x0_loss']) for r in rows if r['status']=='completed'):raise ValueError('Failed numerical gate')
        write(root/f'pilot_gate_gap{a.gap}.json',dict(passed=True,trials=rows,claim='engineering only'))
    elif a.stage=='freeze':
        rows=collect(root,a.gap,'development');extra=collect(root,a.gap,'fourier')
        base={(r['trial']['backbone'],r['trial']['seed']):r['best_dev_x0_loss'] for r in rows if r['status']=='completed' and r['trial']['case']=='U'}
        ratios=[r['best_dev_x0_loss']/base[(r['trial']['backbone'],r['trial']['seed'])] for r in extra if r['status']=='completed']
        # Paired normalized mean loss, monotone with each within-pair rank; tie chooses no Fourier.
        wins=sum(x<1 for x in ratios);fourier=wins>len(ratios)/2
        if not ratios:raise ValueError('No Fourier comparison')
        write(root/f'freeze_gap{a.gap}.json',dict(fourier=fourier,paired_fourier_loss_ratios=ratios,
            selection='majority paired normalized rank; ties no Fourier',gate_sha256=digest(root/f'plan_gap{a.gap}/gate.json'),
            holdout_opened=False,models_and_lengths_frozen=True))
    else:
        rows=collect(root,a.gap,'formal')
        write(root/f'formal_training_summary_gap{a.gap}.json',dict(trials=rows,holdout_opened=False,
            status='formal fits complete; generation fidelity evaluation remains separate'))


if __name__=='__main__':main()
