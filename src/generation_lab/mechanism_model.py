"""Two shared-budget diffusion denoisers with explicit valid-point masks.

These are project-built backbones, not claims of exact Diffusion-TS/UNet
reproduction. Neither uses normalization statistics over padded time points.
"""
import math
import torch
from torch import nn


class Denoiser(nn.Module):
    def __init__(self, backbone, width=64, n_states=5):
        super().__init__()
        self.backbone = backbone
        self.states = nn.Embedding(n_states + 2, 16, padding_idx=0)
        self.input = nn.Linear(20, width)
        self.time = nn.Sequential(nn.Linear(32, width), nn.SiLU(), nn.Linear(width, width))
        if backbone == 'masked_transformer':
            self.layers = nn.ModuleList([nn.TransformerEncoderLayer(width, 4, width * 2, dropout=0.,
                                        batch_first=True, norm_first=True) for _ in range(3)])
        elif backbone == 'masked_conv':
            self.layers = nn.ModuleList([nn.Conv1d(width, width, 3, padding=d, dilation=d)
                                        for d in (1, 2, 4, 8, 16, 32)])
        else:
            raise ValueError(backbone)
        self.out = nn.Linear(width, 1)

    def forward(self, x, t, z, mask, pos, duration):
        m = mask[..., None]
        freq = torch.exp(-math.log(10000) * torch.arange(16, device=x.device) / 15)
        phase = t[:, None] * freq
        temb = self.time(torch.cat((phase.sin(), phase.cos()), -1))
        h = self.input(torch.cat((x[..., None], self.states(z), pos[..., None],
                                  duration[..., None], mask[..., None].float()), -1))
        # Native sample position; does not warp block duration or oscillation rate.
        index = torch.arange(x.shape[1], device=x.device)[:, None]
        f = torch.exp(-math.log(10000) * torch.arange(h.shape[-1] // 2, device=x.device) / (h.shape[-1] // 2))
        pe = torch.cat(((index * f).sin(), (index * f).cos()), -1)
        h = (h + temb[:, None] + pe[None]) * m
        for layer in self.layers:
            if self.backbone == 'masked_transformer':
                h = layer(h, src_key_padding_mask=~mask) * m
            else:
                h = (h + torch.nn.functional.silu(layer(h.transpose(1, 2)).transpose(1, 2))) * m
        return self.out(h).squeeze(-1) * mask


class StateDiffusion(nn.Module):
    def __init__(self, backbone, width=64, n_states=5):
        super().__init__()
        self.net = Denoiser(backbone, width, n_states)
        u = torch.linspace(0, 1, 501)
        a = torch.cos((u + .008) / 1.008 * math.pi / 2).square()
        self.register_buffer('alpha', (a[1:] / a[0]).clamp(1e-5, .9999))

    def loss(self, batch, robust=False):
        y, z, mask, pos, duration = batch
        t = torch.randint(0, len(self.alpha), (len(y),), device=y.device)
        a = self.alpha[t, None]
        x = (a.sqrt() * y + (1-a).sqrt() * torch.randn_like(y)) * mask
        pred = self.net(x, t.float(), z, mask, pos, duration)
        point = (pred - y).abs()
        loss = ((point * mask).sum(1) / mask.sum(1)).mean()
        if robust:
            # Independent extra observation corruption with the SAME diffusion
            # noise: consistency discourages response to added sensor noise.
            corrupted = x + a.sqrt() * .03 * torch.randn_like(x) * mask
            second = self.net(corrupted, t.float(), z, mask, pos, duration)
            consistency = ((second-pred.detach()).abs() * mask).sum(1) / mask.sum(1)
            loss = loss + .2 * consistency.mean()
        return loss

    @torch.no_grad()
    def sample(self, batch, steps=50):
        _, z, mask, pos, duration = batch
        x = torch.randn(z.shape, device=z.device) * mask
        times = torch.linspace(len(self.alpha)-1, 0, steps, device=z.device).long()
        for i, t in enumerate(times):
            a = self.alpha[t]
            pred = self.net(x, t.expand(len(z)).float(), z, mask, pos, duration)
            if i == len(times)-1:
                x = pred
            else:
                an = self.alpha[times[i+1]]
                eps = (x-a.sqrt()*pred) / (1-a).sqrt()
                x = (an.sqrt()*pred + (1-an).sqrt()*eps) * mask
        return x
