"""Read/write verification using only canonical paths and synthetic data."""
import json
from pathlib import Path

from src.framework.workflow import Workflow
from src.framework.step import Step
from src.framework.run_paths import run_directories


class SyntheticStep(Step):
    step_type = 'layout_verification'

    def run(self, context):
        path = Path(self.log_dir(context))/'result.json'
        path.write_text(json.dumps({'values': [1, 2, 3], 'synthetic': True}))
        self.record(context, {'result': self.rel(context, str(path))})
        return context


def test_canonical_roundtrip(tmp_path, monkeypatch):
    from visualize import viz_common
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(viz_common, 'PROJECT_ROOT', str(tmp_path))
    workflow = Workflow('synthetic', 'fixture', {})
    workflow.add(SyntheticStep())
    workflow.run()
    manifest = viz_common.load_manifest('synthetic')
    result = Path(manifest.artifact_path('layout_verification', 'result'))
    assert result.resolve().is_relative_to(tmp_path/'runs/synthetic')
    assert json.loads(result.read_text())['synthetic']
    assert Path(viz_common.fig_dir('synthetic', 'check')).is_relative_to(tmp_path/'runs/synthetic')
    assert not (tmp_path/'log').exists()
    assert not (tmp_path/'output').exists()
    assert run_directories('synthetic')[0] == Path('runs/synthetic')
