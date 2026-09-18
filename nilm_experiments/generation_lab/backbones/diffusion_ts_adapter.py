"""Project conditioning adapter for Diffusion-TS."""
import torch
from torch import nn

class ConditionalDiffusionTS(nn.Module):
    def __init__(self,length):
        super().__init__()
        if __package__.startswith('nilm_experiments.'):
            from ...third_party.diffusion_ts.transformer import Transformer
        else:
            from third_party.diffusion_ts.transformer import Transformer
        self.base=Transformer(n_feat=1,n_channel=length,n_layer_enc=1,n_layer_dec=2,
            n_embd=64,n_heads=4,max_len=length,conv_params=[1,0],attn_pdrop=0.,resid_pdrop=0.)
        self.states=nn.Embedding(7,16)
        self.condition=nn.Linear(18,64)

    def forward(self,x,t,z,known,mask):
        base=self.base
        emb=base.emb(x)+self.condition(torch.cat([self.states(z),known,mask.float()],-1))
        enc=base.encoder(base.pos_enc(emb),t)
        output,mean,trend,season=base.decoder(base.pos_dec(emb),t,enc)
        res=base.inverse(output);res_mean=res.mean(1,keepdim=True)
        season_error=base.combine_s(season.transpose(1,2)).transpose(1,2)+res-res_mean
        trend=base.combine_m(mean)+res_mean+trend
        return trend+season_error
