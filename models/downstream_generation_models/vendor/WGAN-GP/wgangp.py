"""Project PyTorch 1D adaptation of Gulrajani et al. WGAN-GP (2017).

Not an exact reproduction of the authors' TensorFlow image architecture.
U and primitive-state conditional P jointly generate length-512 waveforms.
"""
from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F

@dataclass
class PrimitiveCondition:
    state_id: torch.Tensor
    continuous: torch.Tensor
    dictionary_id: str

class ConditionEncoder(nn.Module):
    def __init__(self,k):
        super().__init__();self.types=nn.Embedding(k,16);self.projection=nn.Conv1d(20,16,1)
    def forward(self,c):
        x=torch.cat([self.types(c.state_id),c.continuous],-1).transpose(1,2)
        return self.projection(x)

class Generator(nn.Module):
    def __init__(self,k=None):
        super().__init__();self.primitive=ConditionEncoder(k) if k else None
        self.input=nn.Linear(128,128*16)
        widths=[128,128,64,32,16,16];extra=16 if k else 0
        self.blocks=nn.ModuleList([nn.Sequential(nn.Conv1d(a+extra,b,5,padding=2),nn.GroupNorm(8,b),nn.ReLU()) for a,b in zip(widths[:-1],widths[1:])])
        self.output=nn.Conv1d(16+extra,1,5,padding=2)
    def forward(self,z,c=None):
        h=self.input(z).view(-1,128,16);cond=self.primitive(c) if self.primitive else None
        for block in self.blocks:
            h=F.interpolate(h,scale_factor=2,mode='nearest')
            if cond is not None:h=torch.cat([h,F.adaptive_avg_pool1d(cond,h.shape[-1])],1)
            h=block(h)
        if cond is not None:h=torch.cat([h,cond],1)
        return torch.tanh(self.output(h)).transpose(1,2)

class Critic(nn.Module):
    def __init__(self,k=None):
        super().__init__();self.primitive=ConditionEncoder(k) if k else None
        widths=[1,32,64,128,128,128];extra=16 if k else 0
        self.blocks=nn.ModuleList([nn.Conv1d(a+extra,b,4,stride=2,padding=1) for a,b in zip(widths[:-1],widths[1:])])
        self.output=nn.Linear(128*16,1)
    def forward(self,x,c=None):
        h=x.transpose(1,2);cond=self.primitive(c) if self.primitive else None
        for block in self.blocks:
            if cond is not None:h=torch.cat([h,F.adaptive_avg_pool1d(cond,h.shape[-1])],1)
            h=F.leaky_relu(block(h),.2)
        return self.output(h.flatten(1)).squeeze(-1)

def gradient_penalty(critic,real,fake,c=None,alpha=None):
    if alpha is None:alpha=torch.rand(len(real),1,1,device=real.device)
    interpolated=(alpha*real+(1-alpha)*fake.detach()).requires_grad_(True)
    score=critic(interpolated,c)
    grad=torch.autograd.grad(score,interpolated,grad_outputs=torch.ones_like(score),create_graph=True,only_inputs=True)[0]
    norms=grad.flatten(1).norm(2,dim=1)
    return ((norms-1)**2).mean(),norms

class WGAN_GP(nn.Module):
    def __init__(self,k,dictionary_id,conditional):
        super().__init__();self.k=k;self.dictionary_id=dictionary_id;self.conditional=conditional
        self.generator=Generator(k if conditional else None);self.critic=Critic(k if conditional else None)
        self.apply(self.initialize)
    @staticmethod
    def initialize(m):
        if isinstance(m,(nn.Conv1d,nn.Linear,nn.Embedding)):
            nn.init.normal_(m.weight,0,.02)
            if getattr(m,'bias',None) is not None:nn.init.zeros_(m.bias)
    def validate(self,c,n):
        if not self.conditional:
            if c is not None:raise ValueError('U must not receive primitive conditions')
            return
        if c is None or c.dictionary_id!=self.dictionary_id:raise ValueError('Missing or mismatched dictionary')
        assert c.state_id.shape==(n,512) and c.state_id.dtype==torch.long
        assert c.continuous.shape==(n,512,4) and torch.isfinite(c.continuous).all()
        assert c.state_id.min()>=0 and c.state_id.max()<self.k
        assert ((c.continuous[...,1:]>=0)&(c.continuous[...,1:]<=1)).all()
    @torch.no_grad()
    def sample(self,n,c=None,seed=0):
        self.validate(c,n);device=next(self.parameters()).device
        rng=torch.Generator(device=device).manual_seed(seed)
        z=torch.randn(n,128,device=device,generator=rng)
        return self.generator(z,c)
