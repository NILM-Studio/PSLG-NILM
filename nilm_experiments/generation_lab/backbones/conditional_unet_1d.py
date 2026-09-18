"""Project-built conditional one-dimensional U-Net."""
import math
import torch
from torch import nn
from torch.nn import functional as F

class Residual(nn.Module):
    def __init__(self,cin,cout):
        super().__init__()
        self.norm1=nn.GroupNorm(8,cin);self.c1=nn.Conv1d(cin,cout,3,padding=1)
        self.norm2=nn.GroupNorm(8,cout);self.c2=nn.Conv1d(cout,cout,3,padding=1)
        self.time=nn.Linear(128,2*cout);self.skip=nn.Conv1d(cin,cout,1) if cin!=cout else nn.Identity()
    def forward(self,x,time):
        h=self.c1(F.silu(self.norm1(x)));scale,bias=self.time(time).chunk(2,-1)
        h=self.norm2(h)*(1+scale[:,:,None])+bias[:,:,None]
        return self.skip(x)+self.c2(F.silu(h))


class ConditionalUNet(nn.Module):
    def __init__(self,length,base=32):
        super().__init__();self.states=nn.Embedding(7,16)
        self.inp=nn.Conv1d(19,base,3,padding=1)
        self.time=nn.Sequential(nn.Linear(128,128),nn.SiLU(),nn.Linear(128,128))
        self.condition=nn.ModuleList([nn.Conv1d(18,c,1) for c in (base,2*base,4*base)])
        self.enc=nn.ModuleList([nn.ModuleList([Residual(c,c),Residual(c,c)]) for c in (base,2*base,4*base)])
        self.down=nn.ModuleList([nn.Conv1d(base,2*base,4,stride=2,padding=1),nn.Conv1d(2*base,4*base,4,stride=2,padding=1)])
        self.up=nn.ModuleList([nn.Conv1d(4*base,2*base,3,padding=1),nn.Conv1d(2*base,base,3,padding=1)])
        self.dec=nn.ModuleList([nn.ModuleList([Residual(4*base,2*base),Residual(2*base,2*base)]),
                                nn.ModuleList([Residual(2*base,base),Residual(base,base)])])
        self.out=nn.Sequential(nn.GroupNorm(8,base),nn.SiLU(),nn.Conv1d(base,1,3,padding=1))
    def forward(self,x,t,z,known,mask):
        freq=torch.exp(-math.log(10000)*torch.arange(64,device=x.device)/63)
        phase=t.float()[:,None]*freq[None];time=self.time(torch.cat([phase.sin(),phase.cos()],-1))
        c=torch.cat([self.states(z),known,mask.float()],-1).transpose(1,2)
        h=self.inp(torch.cat([x.transpose(1,2),c],1));skips=[]
        for i,pair in enumerate(self.enc):
            h=h+self.condition[i](F.interpolate(c,size=h.shape[-1],mode='nearest'))
            for block in pair:h=block(h,time)
            skips.append(h)
            if i<2:h=self.down[i](h)
        for i,pair in enumerate(self.dec):
            skip=skips[1-i];h=self.up[i](F.interpolate(h,size=skip.shape[-1],mode='nearest'))
            h=torch.cat([h,skip],1)
            for block in pair:h=block(h,time)
        return self.out(h).transpose(1,2)
