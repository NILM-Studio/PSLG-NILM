"""Launch an already registered execution and capture its command, logs and status."""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
from create_execution import ROOT, save

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('execution',type=Path); p.add_argument('command',nargs=argparse.REMAINDER)
    a=p.parse_args(); target=a.execution.resolve()
    if not target.is_relative_to(ROOT/'runs') or not (target/'run.json').is_file():
        p.error('Not a registered execution')
    command=a.command[1:] if a.command[:1]==['--'] else a.command
    if not command: p.error('Missing command')
    run=json.loads((target/'run.json').read_text())
    if run['status']!='registered': p.error('Execution already used; register a new execution')
    env=dict(os.environ,PSLG_EXECUTION_DIR=str(target))
    save(target/'command.json',{'argv':command,'cwd':str(ROOT),
        'environment_overrides':{'PSLG_EXECUTION_DIR':str(target)}})
    run.update(status='running',started_at=dt.datetime.now(dt.timezone.utc).isoformat())
    save(target/'run.json',run)
    code=1
    try:
        with (target/'logs/console.log').open('w') as log:
            code=subprocess.call(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    finally:
        run.update(status='passed' if code==0 else 'failed',exit_code=code,
                   completed_at=dt.datetime.now(dt.timezone.utc).isoformat())
        save(target/'run.json',run)
        artifacts=[]
        for f in target.rglob('*'):
            if f.is_file() and f.name!='artifacts.json':
                record={'path':str(f.relative_to(target)),'size_bytes':f.stat().st_size}
                if f.stat().st_size<32*1024*1024: record['sha256']=hashlib.sha256(f.read_bytes()).hexdigest()
                artifacts.append(record)
        save(target/'artifacts.json',{'artifacts':artifacts,'validation':'process exit only; scientific validation belongs in report'})
    print(json.dumps({'execution':str(target),'exit_code':code}))
    raise SystemExit(code)

if __name__=='__main__': main()
