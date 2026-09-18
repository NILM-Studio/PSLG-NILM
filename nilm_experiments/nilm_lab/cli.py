"""Compatibility entry into the canonical project Workflow (no duplicate trainer)."""
import argparse
from pathlib import Path
from .common import load_config
from .workflow import trials
from .standalone import run


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['matrix', 'train', 'select', 'evaluate', 'report'])
    p.add_argument('--config', required=True)
    p.add_argument('--run-id', required=False)
    args = p.parse_args()
    cfg = load_config(args.config)
    if args.command == 'matrix':
        import json
        print(json.dumps(trials(cfg['nilm']), indent=2))
        return
    if not args.run_id:
        p.error('--run-id must reference a source-only Workflow run')
    run(args.command, str(Path(args.config).resolve()), args.run_id)


if __name__ == '__main__':
    main()
