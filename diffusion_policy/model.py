"""1D temporal U-Net with FiLM conditioning (Diffusion Policy, Chi et al., RSS 2023).

Re-implemented from the paper and its public state-based Push-T notebook
(https://github.com/real-stanford/diffusion_policy); same structure:
residual Conv1d blocks with GroupNorm + Mish, FiLM (per-channel scale and bias)
from [diffusion-step embedding, flattened observation history].
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=x.device) * -emb)
        emb = x[:, None].float() * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class Conv1dBlock(nn.Module):
    def __init__(self, cin, cout, k, n_groups=8):
        super().__init__()
        self.block = nn.Sequential(nn.Conv1d(cin, cout, k, padding=k // 2), nn.GroupNorm(n_groups, cout), nn.Mish())

    def forward(self, x):
        return self.block(x)


class CondResBlock1D(nn.Module):
    def __init__(self, cin, cout, cond_dim, k=5, n_groups=8):
        super().__init__()
        self.blocks = nn.ModuleList([Conv1dBlock(cin, cout, k, n_groups), Conv1dBlock(cout, cout, k, n_groups)])
        self.cout = cout
        self.cond_encoder = nn.Sequential(nn.Mish(), nn.Linear(cond_dim, 2 * cout))   # FiLM scale and bias
        self.residual = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, cond):
        out = self.blocks[0](x)
        film = self.cond_encoder(cond).reshape(cond.shape[0], 2, self.cout, 1)
        out = film[:, 0] * out + film[:, 1]
        out = self.blocks[1](out)
        return out + self.residual(x)


class ConditionalUnet1D(nn.Module):
    def __init__(self, input_dim, global_cond_dim, step_embed_dim=256, down_dims=(256, 512, 1024),
                 kernel_size=5, n_groups=8):
        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]
        self.step_encoder = nn.Sequential(SinusoidalPosEmb(step_embed_dim),
                                          nn.Linear(step_embed_dim, 4 * step_embed_dim), nn.Mish(),
                                          nn.Linear(4 * step_embed_dim, step_embed_dim))
        cond_dim = step_embed_dim + global_cond_dim
        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        mid = all_dims[-1]
        self.mid = nn.ModuleList([CondResBlock1D(mid, mid, cond_dim, kernel_size, n_groups),
                                  CondResBlock1D(mid, mid, cond_dim, kernel_size, n_groups)])
        self.down = nn.ModuleList()
        for i, (din, dout) in enumerate(in_out):
            last = i >= len(in_out) - 1
            self.down.append(nn.ModuleList([
                CondResBlock1D(din, dout, cond_dim, kernel_size, n_groups),
                CondResBlock1D(dout, dout, cond_dim, kernel_size, n_groups),
                nn.Conv1d(dout, dout, 3, 2, 1) if not last else nn.Identity()]))
        self.up = nn.ModuleList()
        for i, (din, dout) in enumerate(reversed(in_out[1:])):
            last = i >= len(in_out) - 1
            self.up.append(nn.ModuleList([
                CondResBlock1D(2 * dout, din, cond_dim, kernel_size, n_groups),
                CondResBlock1D(din, din, cond_dim, kernel_size, n_groups),
                nn.ConvTranspose1d(din, din, 4, 2, 1) if not last else nn.Identity()]))
        self.final = nn.Sequential(Conv1dBlock(start_dim, start_dim, kernel_size, n_groups),
                                   nn.Conv1d(start_dim, input_dim, 1))

    def forward(self, sample, timestep, global_cond):
        """sample (B, H, input_dim), timestep (B,), global_cond (B, cond) -> (B, H, input_dim)."""
        x = sample.moveaxis(-1, -2)
        gf = torch.cat([self.step_encoder(timestep), global_cond], dim=-1)
        skips = []
        for r1, r2, ds in self.down:
            x = r2(r1(x, gf), gf)
            skips.append(x)
            x = ds(x)
        for m in self.mid:
            x = m(x, gf)
        for r1, r2, us in self.up:
            x = torch.cat([x, skips.pop()], dim=1)
            x = us(r2(r1(x, gf), gf))
        return self.final(x).moveaxis(-1, -2)
