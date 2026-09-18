"""Workflow-only orchestration; never import Torch/vendor src in this process."""
from __future__ import annotations
import json
import os
from pathlib import Path
import subprocess
import sys
from src.framework.step import Step
from nilm_experiments.nilm_lab.common import digest, signature, read_json, write_json

PROJECT = Path(__file__).resolve().parents[2]


def config_fingerprint(cfg):
    return signature({k: v for k, v in cfg.items() if not k.startswith('_')})


def require_artifact(context, step, key):
    p = Step.resolve(context, step, key)
    if not p or not Path(p).exists():
        raise FileNotFoundError(f'Required {step}.{key}; run its producer first')
    return Path(p)


def strict_guard(context):
    if context['config'].get('workflow', {}).get('profile') != 'nilm':
        return
    p = require_artifact(context, 'nilm_data', 'split_manifest')
    m = read_json(p)
    if m['config_fingerprint'] != config_fingerprint(context['config']):
        raise ValueError('Frozen run configuration changed; use a new run-id')
    if digest(m['input_manifest']) != m['input_manifest_sha256']:
        raise ValueError('Split/channel manifest changed after nilm_data')
    for r in m['records']:
        if r['split'] != 'train' or digest(p.parent / r['file']) != r['sha256']:
            raise ValueError('Source discovery data changed or contains nontraining records')


class NilmStep(Step):
    def __init__(self, cfg, selection):
        super().__init__()
        self.cfg, self.selection = cfg, selection

    def fresh_dir(self, context):
        p = Path(context['log_root']) / self.step_type
        p.mkdir(parents=True, exist_ok=False)
        return p

    def register(self, context, out, files):
        self.record(context, {key: self.rel(context, str(out / file)) for key, file in files.items()})
        return context

    def worker(self, context, command, output):
        runtime = self.cfg.get('runtime', {})
        python = runtime.get('torch_python', sys.executable)
        request = output / 'request.json'
        write_json(request, dict(command=command, config=self.cfg,
            log_root=str(Path(context['log_root']).resolve()), output=str(output.resolve()),
            manifest=context['manifest'].data))
        env = os.environ.copy()
        env['PYTHONPATH'] = str(PROJECT / 'nilm_experiments')
        command_line = [python, '-m', 'nilm_lab.workflow', str(request.resolve())]
        # Native CPU kernels avoid nonfinite outputs seen in the host's oneDNN
        # integration checks. GPU workers retain their normal CUDA execution.
        if command in {'train', 'evaluate'} and self.cfg.get('nilm', {}).get('training', {}).get('device', 'cpu').split(':')[0] == 'cpu':
            command_line = [python, '-c',
                'import torch; torch.backends.mkldnn.enabled=False; from nilm_lab.workflow import main; main()',
                str(request.resolve())]
        subprocess.run(command_line,
                       cwd=PROJECT / 'nilm_experiments', env=env, check=True)
