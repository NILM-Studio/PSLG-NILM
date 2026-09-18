"""Compatibility contracts for the 2026-09-17 layout migration."""
import os
from pathlib import Path
import tarfile
import types

import pytest

from nilm_experiments.run_paths import resolve_campaign
from src.framework.workflow import Workflow

PROJECT = Path(__file__).resolve().parents[1]


def test_campaign_paths(tmp_path):
    assert resolve_campaign('new', tmp_path) == tmp_path/'runs/generation/new'
    assert resolve_campaign('runs/generation/new', tmp_path) == tmp_path/'runs/generation/new'
    legacy = tmp_path/'nilm_experiments/old'
    legacy.mkdir(parents=True)
    assert resolve_campaign(str(legacy), tmp_path) == legacy
    with pytest.raises(ValueError):
        resolve_campaign('nilm_experiments/new', tmp_path)
    with pytest.raises(ValueError):
        resolve_campaign('../escape', tmp_path)


def test_new_and_existing_workflows(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    new = Workflow('new', 'fridge', {})
    assert Path(new.log_root) == Path('runs/new')
    assert not Path('log/new').exists()
    assert not Path('output/new').exists()
    new.manifest.save()
    assert Workflow('new', 'fridge', {}).manifest.path == new.manifest.path
    Path('legacy/log/old').mkdir(parents=True)
    old = Workflow('old', 'fridge', {})
    assert old.log_root == 'legacy/log/old'
    assert old.output_root == 'legacy/output/old'
    with pytest.raises(ValueError):
        Workflow('../bad', 'fridge', {})


def test_third_party_has_no_aliases():
    exp = PROJECT/'nilm_experiments'
    for path in ['FCN/src', 'NILMFormer', 'BERT4NILM', 'SGN', 'generation_lab/vendor_diffusion_ts']:
        assert not os.path.lexists(exp/path)
    assert (exp/'third_party/nilmformer/src/baselines/nilm/fcn.py').is_file()
    assert (exp/'third_party/diffusion_ts/transformer.py').is_file()


@pytest.mark.parametrize('name', ['diffusion_ts', 'conditional_unet_1d'])
def test_split_preserves_weights_forward_loss_gradients_and_sampling(name):
    import torch
    from nilm_experiments.generation_lab.models import Diffusion
    torch.set_num_threads(2)
    archive = PROJECT/'.backups/layout_refactor_20260917/originals.tgz'
    if not archive.exists():
        pytest.skip('One-time migration comparison requires the preserved originals')
    with tarfile.open(archive) as original:
        code = original.extractfile('nilm_experiments/generation_lab/models.py').read()
    # The historical source imports a removed alias; redirect just that import.
    code = code.replace(b'from .vendor_diffusion_ts.transformer', b'from ..third_party.diffusion_ts.transformer')
    old = types.ModuleType('nilm_experiments.generation_lab._before_refactor')
    old.__package__ = 'nilm_experiments.generation_lab'
    exec(compile(code, 'before_refactor/models.py', 'exec'), old.__dict__)
    torch.manual_seed(71)
    before = old.Diffusion(name, 16, fourier=True)
    after = Diffusion(name, 16, fourier=True)
    after.load_state_dict(before.state_dict(), strict=True)
    assert before.state_dict().keys() == after.state_dict().keys()
    y = torch.rand(2, 16, 1)
    z = torch.ones(2, 16, dtype=torch.long)
    a, b = before.loss(y, z, 113), after.loss(y, z, 113)
    assert torch.equal(a, b)
    a.backward()
    b.backward()
    for (_, p), (_, q) in zip(before.named_parameters(), after.named_parameters()):
        assert (p.grad is None) == (q.grad is None)
        if p.grad is not None:
            assert torch.equal(p.grad, q.grad)
    mask = torch.zeros_like(y, dtype=torch.bool)
    mask[:, :2] = True
    first = before.sample(z, y*mask, mask, steps=4, seed=17)
    second = after.sample(z, y*mask, mask, steps=4, seed=17)
    assert torch.equal(first, second)
    assert torch.equal(second[:, :2], y[:, :2])
