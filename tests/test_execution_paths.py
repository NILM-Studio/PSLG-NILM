import pytest
from src.framework.run_paths import run_directories
from src.generation_paths import resolve_campaign

def test_registered_execution_override(tmp_path,monkeypatch):
    target=tmp_path/'runs/20261015_test/executions/smoke/id'
    target.mkdir(parents=True); (target/'run.json').write_text('{}')
    monkeypatch.setenv('PSLG_EXECUTION_DIR',str(target))
    assert run_directories('test',tmp_path)==(target,target)
    assert resolve_campaign('test',tmp_path)==target

def test_unregistered_and_outside_rejected(tmp_path,monkeypatch):
    target=tmp_path/'runs/20261015_test/executions/smoke/id'
    monkeypatch.setenv('PSLG_EXECUTION_DIR',str(target))
    with pytest.raises(ValueError):run_directories('test',tmp_path)
    with pytest.raises(ValueError):resolve_campaign('test',tmp_path)
    monkeypatch.setenv('PSLG_EXECUTION_DIR',str(tmp_path/'outside'))
    with pytest.raises(ValueError):run_directories('test',tmp_path)
