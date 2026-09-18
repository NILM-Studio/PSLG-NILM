"""Resolve new and historical workflow locations without compatibility links."""
from pathlib import Path


def run_directories(run_id, project_root='.'):
    if not run_id or run_id in {'.', '..'} or '/' in run_id or '\\' in run_id:
        raise ValueError('run_id must be a single directory name')
    root = Path(project_root)
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
