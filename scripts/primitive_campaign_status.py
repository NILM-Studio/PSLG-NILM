"""Read-only summary of current run artifacts; never loads a dataset."""
import argparse
from pathlib import Path
import json


def read(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError,json.JSONDecodeError):
        return None


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run-id',default='primitive_source_20260915_v2')
    p.add_argument('--audit',action='store_true')
    args=p.parse_args()
    import sys
    project = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project))
    from src.framework.run_paths import run_directories
    root = run_directories(args.run_id, project)[0]
    result=dict(run_id=args.run_id,test_evaluated=(root/'nilm_evaluate/results.json').exists(),models=[])
    audit=read(root/'occlusion_audit.json')
    if audit and args.audit:
        result['occlusion']=[dict(record_id=r['record_id'],restoration=r['restoration_agreement'],
            reliable_boundaries=r['reliable_local_boundaries'],activities_with_two_boundaries=r['activities_with_two_reliable_boundaries'],
            conditions=[{k:d[k] for k in ['stratum','width','hidden_points','boundary_f1_12s','state_agreement','on_state_retention']} for d in r['diagnostics']]) for r in audit['reports']]
    for name in ['NILMFormer','FCN']:
        folder=root/'nilm_train'/name
        runs=[]
        for path in folder.glob('*/run.json'):
            run=read(path)
            if run is None:
                continue
            finished=read(path.parent/'finished.json')
            history=read(path.parent/'history.json') or []
            runs.append(dict(**run['trial'],finished=finished is not None,epochs=len(history),
                best_mae=finished['best_val_mae'] if finished else min((h['val_house_macro_mae'] for h in history),default=None),
                seconds=finished['seconds'] if finished else None,mtime=path.stat().st_mtime))
        runs.sort(key=lambda x:x.pop('mtime'))
        item=dict(model=name,completed=sum(r['finished'] for r in runs),started=len(runs),last=runs[-1:] or None,
            lr_selection=read(folder/'learning_rate_selection.json'),
            source_selected=(folder/'source_selection.json').exists(),gate=read(folder/'source_gate.json'),
            formal_frozen=(folder/'formal_freeze.json').exists())
        if item['lr_selection']:
            item['lr_selection']=item['lr_selection']['selected']
        result['models'].append(item)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
