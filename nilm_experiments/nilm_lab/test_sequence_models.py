"""Run in a fresh Torch process: project src must not be imported first."""
import unittest
import numpy as np
import torch
from .models import make_model
from .train import seed_all
from .sequence_losses import sequence_objective


class SequenceModelTests(unittest.TestCase):
    def test_two_backbones_shapes_gradients_and_reset(self):
        torch.set_num_threads(1)
        for name in ('NILMFormer', 'FCN'):
            with self.subTest(model=name):
                seed_all(7)
                model = make_model(name, 32, 3, activity_mode='gru', hidden_size=8).eval()
                x = torch.randn(2, 9, 32)
                p, z, a = model(x)
                self.assertEqual(tuple(p.shape), (2, 32))
                self.assertEqual(tuple(z.shape), (2, 3, 32))
                expected = a.detach().clone()
                model(x.flip(-1))
                torch.testing.assert_close(model(x)[2], expected)
                a.sum().backward()
                self.assertGreater(model.state_head.weight.grad.abs().sum().item(), 0)
                self.assertGreater(sum(q.grad.abs().sum().item() for q in model.base.parameters() if q.grad is not None), 0)

    def test_auxiliary_heads_preserve_backbone_initialization(self):
        for name in ('NILMFormer', 'FCN'):
            bases = []
            for classes, mode in ((0, 'none'), (3, 'none'), (3, 'direct'), (3, 'gru')):
                seed_all(9)
                model = make_model(name, 32, classes, activity_mode=mode)
                bases.append({k: v.detach().clone() for k,v in model.base.state_dict().items()})
            for base in bases[1:]:
                for key in base:
                    torch.testing.assert_close(base[key], bases[0][key])

    def test_missing_targets_have_zero_loss_gradient(self):
        power = torch.ones(1, 4, requires_grad=True)
        z = torch.randn(1, 3, 4, requires_grad=True)
        a = torch.randn(1, 4, requires_grad=True)
        b = dict(y=torch.zeros(1,4), z=torch.ones(1,4,dtype=torch.long),
                 activity=torch.ones(1,4,dtype=torch.long), observed=torch.tensor([[True, False, True, False]]))
        loss, *_ = sequence_objective(power, z, a, b, .01, .01)
        loss.backward()
        for grad in (power.grad[:,1::2], z.grad[:,:,1::2], a.grad[:,1::2]):
            self.assertEqual(grad.abs().sum().item(), 0)


if __name__ == '__main__':
    unittest.main()
