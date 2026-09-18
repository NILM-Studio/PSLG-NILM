"""Shared x0 diffusion objective, schedule and sampler."""
import math
import torch
from torch import nn
from .backbones.diffusion_ts_adapter import ConditionalDiffusionTS
from .backbones.conditional_unet_1d import ConditionalUNet

class Diffusion(nn.Module):
    def __init__(self,backbone,length,fourier=False,unet_base=32):
        super().__init__();self.length=length;self.fourier=fourier
        self.net=ConditionalDiffusionTS(length) if backbone=='diffusion_ts' else ConditionalUNet(length,base=unet_base)
        v=torch.linspace(0,500,501,dtype=torch.float64)
        cumulative=torch.cos(((v/500+.008)/1.008)*math.pi/2)**2
        cumulative=cumulative/cumulative[0]
        beta=(1-cumulative[1:]/cumulative[:-1]).clamp(0,.999)
        alpha=1-beta
        self.register_buffer('alpha_bar',alpha.cumprod(0).float())
        self.register_buffer('loss_weight',(alpha.sqrt()*(1-alpha.cumprod(0)).sqrt()/beta/100).float())
    def loss(self,y,z,seed):
        gen=torch.Generator(device=y.device).manual_seed(seed)
        b,l,_=y.shape;t=torch.randint(0,500,(b,),device=y.device,generator=gen)
        noise=torch.randn(y.shape,device=y.device,generator=gen)
        a=self.alpha_bar[t,None,None]
        x=a.sqrt()*y+(1-a).sqrt()*noise
        lengths=torch.randint(0,2,(b,),device=y.device,generator=gen)*max(2,l//8)
        mask=(torch.arange(l,device=y.device)[None,:]<lengths[:,None])[...,None]
        known=y*mask
        pred=self.net(x,t,z,known,mask)
        unknown=(~mask).float()
        per=((pred-y).abs()*unknown).sum((1,2))/unknown.sum((1,2)).clamp_min(1)
        if self.fourier:
            whole=torch.where(mask,y,pred)
            delta=torch.fft.fft(whole,dim=1,norm='forward')-torch.fft.fft(y,dim=1,norm='forward')
            per=per+(l**.5/5)*(delta.real.abs()+delta.imag.abs()).mean((1,2))
        return (per*self.loss_weight[t]).mean()
    @torch.no_grad()
    def sample(self,z,known=None,mask=None,steps=50,seed=0):
        b,l=z.shape;device=z.device
        gen=torch.Generator(device=device).manual_seed(seed)
        x=torch.randn((b,l,1),device=device,generator=gen)
        known=torch.zeros_like(x) if known is None else known
        mask=torch.zeros_like(x,dtype=torch.bool) if mask is None else mask
        fixed_noise=torch.randn(x.shape,device=device,generator=gen)
        times=torch.linspace(499,0,steps,device=device).long().tolist()
        for j,t in enumerate(times):
            a=self.alpha_bar[t]
            x=torch.where(mask,a.sqrt()*known+(1-a).sqrt()*fixed_noise,x)
            pred=self.net(x,torch.full((b,),t,device=device,dtype=torch.long),z,known,mask)
            pred=torch.where(mask,known,pred)
            if j+1==len(times):x=pred
            else:
                an=self.alpha_bar[times[j+1]];noise=(x-a.sqrt()*pred)/(1-a).sqrt()
                x=an.sqrt()*pred+(1-an).sqrt()*noise
        return torch.where(mask,known,x)
