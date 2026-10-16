import unittest
import math
import torch
from .sequence_heads import SequenceActivityHead, PointActivityHead, FeatureActivityHead
from .sequence_losses import stratified_mean, sequence_objective


class FormalModelTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA diagnostic regression')
    def test_cuda_gru_gradient_diagnostic(self):
        import numpy as np
        from .models import make_model
        from .train import gradient_diagnostic
        from .data import Record,Windows
        r=Record('r','fixture','1','train',np.arange(64)*6,np.ones(64),np.ones(64),'f')
        labels={'r':dict(onoff=np.arange(64)%2,primitive=np.arange(64)%6,
            activity_valid=np.ones(64,bool),valid=np.ones(64,bool))}
        data=Windows([r],32,16,100,labels,'P')
        for mode,classes in [('gru',6),('point',6),('feature_gru',0)]:
            model=make_model('FCN',32,classes,activity_mode=mode).cuda()
            result=gradient_diagnostic(model,data,torch.device('cuda'),
                dict(lambda_p=.01 if classes else 0.,lambda_a=.01),dict(auxiliary_reduction='onoff_balanced'))
            self.assertGreater(result['gradient_norms'][2],0)

    def test_new_arms_train_on_both_backbones_with_same_initialization(self):
        from .models import make_model
        from .train import seed_all
        torch.set_num_threads(1)
        torch.backends.mkldnn.enabled=False
        for name in ('NILMFormer','FCN'):
            baseline=None
            for classes,mode in [(0,'none'),(6,'point'),(6,'direct_matched'),(0,'feature_gru')]:
                seed_all(55)
                model=make_model(name,32,classes,activity_mode=mode)
                values={k:v.detach().clone() for k,v in model.base.state_dict().items()}
                if baseline is None:
                    baseline=values
                else:
                    for key in baseline:
                        torch.testing.assert_close(values[key],baseline[key])
                batch=dict(y=torch.ones(2,32),observed=torch.ones(2,32,dtype=torch.bool),
                    z=torch.arange(32)[None].repeat(2,1)%6,activity=torch.arange(32)[None].repeat(2,1)%2)
                optimizer=torch.optim.Adam(model.parameters(),lr=.0001)
                for _ in range(2):
                    optimizer.zero_grad()
                    outputs=model(torch.randn(2,9,32))
                    loss=sequence_objective(*outputs,batch,.01 if classes else 0.,
                        .01 if mode!='none' else 0.,reduction='onoff_balanced')[0]
                    self.assertTrue(torch.isfinite(loss))
                    loss.backward();optimizer.step()

    def test_equal_parameter_probability_heads_and_locality(self):
        gru, point = SequenceActivityHead(6), PointActivityHead(6)
        self.assertEqual(sum(p.numel() for p in gru.parameters()),3873)
        self.assertEqual(sum(p.numel() for p in point.parameters()),3873)
        x = torch.randn(2,6,20,requires_grad=True)
        y = point(x)
        changed = x.detach().clone()
        changed[:,:,0] += torch.arange(6)[None]
        torch.testing.assert_close(point(changed)[:,1:],y[:,1:])
        gru(x).sum().backward()
        self.assertGreater(float(x.grad.abs().sum()),0)

    def test_balanced_duplication_invariance_and_empty_strata(self):
        a = stratified_mean(torch.tensor([1.,2.]),torch.tensor([0,1]),'onoff_balanced')
        b = stratified_mean(torch.tensor([1.,1.,1.,2.]),torch.tensor([0,0,0,1]),'onoff_balanced')
        torch.testing.assert_close(a,b)
        torch.testing.assert_close(stratified_mean(torch.tensor([2.]),torch.tensor([1]),'onoff_balanced'),torch.tensor(2.))

    def test_missing_target_has_zero_loss_gradient(self):
        power = torch.randn(1,4,requires_grad=True)
        logits = torch.randn(1,6,4,requires_grad=True)
        active = torch.randn(1,4,requires_grad=True)
        batch = dict(observed=torch.tensor([[True,False,True,False]]),y=torch.zeros(1,4),
                     z=torch.tensor([[0,-100,2,-100]]),activity=torch.tensor([[0,-100,1,-100]]))
        loss = sequence_objective(power,logits,active,batch,.01,.01,reduction='onoff_balanced')[0]
        loss.backward()
        self.assertEqual(float(power.grad[:,[1,3]].abs().sum()),0)
        self.assertEqual(float(logits.grad[:,:,[1,3]].abs().sum()),0)
        self.assertEqual(float(active.grad[:,[1,3]].abs().sum()),0)

    def test_feature_gru_resets_window_and_backpropagates(self):
        head = FeatureActivityHead(50,6)
        x = torch.randn(2,50,20,requires_grad=True)
        first = head(x)
        head(torch.randn_like(x))
        torch.testing.assert_close(first,head(x))
        first.sum().backward()
        self.assertGreater(float(x.grad.abs().sum()),0)


if __name__ == '__main__':
    unittest.main()
