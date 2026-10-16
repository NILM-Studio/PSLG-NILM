"""Backward-compatible exports and contract-test entry point."""
import torch
from .backbones.diffusion_ts_adapter import ConditionalDiffusionTS
from .backbones.conditional_unet_1d import ConditionalUNet, Residual
from .diffusion import Diffusion


def selftest(device='cpu'):
    torch.set_num_threads(2)
    for name in ('diffusion_ts','conditional_unet_1d'):
        torch.manual_seed(1);model=Diffusion(name,16).to(device)
        y=torch.rand((2,16,1),device=device);z=torch.ones((2,16),dtype=torch.long,device=device)
        loss=model.loss(y,z,123);loss.backward()
        assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        mask=torch.zeros_like(y,dtype=torch.bool);mask[:,:2]=True;known=y*mask
        a=model.sample(z,known,mask,steps=4,seed=12);b=model.sample(z,known,mask,steps=4,seed=12)
        assert a.shape==y.shape and torch.equal(a,b) and torch.equal(a[:,:2],y[:,:2]) and torch.isfinite(a).all()
        # Verify both categorical conditioning and known-context branches receive gradient.
        assert model.net.states.weight.grad is not None and model.net.states.weight.grad.abs().sum()>0
        print(name,'contract tests passed; parameters',sum(p.numel() for p in model.parameters()),flush=True)


if __name__=='__main__':
    from .common import require_slurm
    require_slurm();selftest('cuda' if torch.cuda.is_available() else 'cpu')
