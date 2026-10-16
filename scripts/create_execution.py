"""Register an isolated execution; this command never launches computation."""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import uuid

ROOT = Path(__file__).resolve().parents[1]

def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')

def register(experiment, condition, config=None):
    if not re.fullmatch(r'\d{8}_[a-zA-Z0-9_-]+', experiment):
        raise ValueError('experiment must be YYYYMMDD_name')
    if not re.fullmatch(r'[a-zA-Z0-9_-]+', condition): raise ValueError('Invalid condition')
    base=ROOT/'runs'/experiment
    if not (base/'proposal.md').is_file(): raise ValueError('Create proposal.md before registration')
    now=dt.datetime.now(dt.timezone.utc).isoformat()
    eid=dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]
    target=base/'executions'/condition/eid
    target.mkdir(parents=True, exist_ok=False)
    for name in ('logs','checkpoints','features','generations','analysis'): (target/name).mkdir()
    try: commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    except subprocess.CalledProcessError: commit='unknown'
    save(target/'run.json',dict(experiment=experiment,condition=condition,execution_id=eid,
        status='registered',created_at=now,exit_code=None))
    save(target/'config.resolved.json',config or {'configuration':'unknown; supply --config JSON'})
    save(target/'command.json',{'argv':None,'cwd':str(ROOT)})
    save(target/'environment.json',{'host':platform.node(),'platform':platform.platform(),
        'registration_python':platform.python_version(),'git_commit':commit,'dirty_tree':True,
        'note':'Registration environment only; runtime environment must be recorded by launcher.'})
    save(target/'inputs.json',{'inputs':[],'status':'not-recorded'})
    save(target/'artifacts.json',{'artifacts':[],'status':'not-validated'})
    # Use an advisory lock because independent executions can register concurrently.
    import fcntl
    index=base/'executions/index.json'
    with (base/'executions/.index.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        entries=json.loads(index.read_text()) if index.exists() else []
        entries.append({'condition':condition,'execution_id':eid,'path':str(target.relative_to(base)),
                        'status_source':str((target/'run.json').relative_to(base))})
        tmp=index.with_suffix('.tmp'); save(tmp,entries); os.replace(tmp,index)
    return target

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('experiment'); p.add_argument('condition'); p.add_argument('--config',type=Path)
    a=p.parse_args()
    print(register(a.experiment,a.condition,json.loads(a.config.read_text()) if a.config else None))
