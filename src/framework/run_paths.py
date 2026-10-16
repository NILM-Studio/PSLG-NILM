"""Resolve new and historical workflow locations without compatibility links."""
from pathlib import Path
import os


def run_directories(run_id, project_root='.'):
    if not run_id or run_id in {'.', '..'} or '/' in run_id or '\\' in run_id:
        raise ValueError('run_id must be a single directory name')
    root = Path(project_root)
    explicit = os.environ.get('PSLG_EXECUTION_DIR')
    if explicit:
        target = Path(explicit).resolve()
        if not target.is_relative_to((root/'runs').resolve()) or 'executions' not in target.parts:
            raise ValueError('PSLG_EXECUTION_DIR must be a registered runs execution')
        if not (target/'run.json').is_file():
            raise ValueError('Register execution with scripts/create_execution.py first')
        return target, target
    current, legacy = root/'runs'/run_id, root/'legacy'/'log'/run_id
    if current.exists():
        if legacy.exists() and legacy.resolve() != current.resolve():
            raise ValueError(f'Ambiguous run_id in runs and legacy/log: {run_id}')
        return current, current
    if legacy.exists():
        return legacy, root/'legacy'/'output'/run_id
    return current, current


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    print(run_directories(args.run_id)[0])
