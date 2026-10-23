"""DDPM training noise schedule, DDIM sampler, and EMA (own code).

Matches the Diffusion Policy low-dim defaults: 100 training steps, squared-
cosine ("squaredcos_cap_v2") beta schedule, epsilon prediction, samples
clipped to [-1, 1]. Inference uses deterministic DDIM (eta = 0) with
"leading" timestep spacing, as in diffusers' DDIMScheduler.
"""
from __future__ import annotations

import copy
import math

import torch


def cosine_betas(T: int, max_beta: float = 0.999) -> torch.Tensor:
    f = lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
    return torch.tensor([min(1 - f((i + 1) / T) / f(i / T), max_beta) for i in range(T)], dtype=torch.float64)


class DDPM:
    def __init__(self, T: int = 100, device="cpu"):
        self.T = T
        betas = cosine_betas(T)
        self.alphas_cumprod = torch.cumprod(1.0 - betas, 0).float().to(device)

    def add_noise(self, x0, noise, t):
        ab = self.alphas_cumprod[t].view(-1, *([1] * (x0.dim() - 1)))
        return ab.sqrt() * x0 + (1 - ab).sqrt() * noise

    @torch.no_grad()
    def ddim_sample(self, model, cond, shape, n_steps: int = 16, generator=None, noise=None):
        """Deterministic DDIM from pure noise. cond: (B, cond_dim)."""
        x = noise if noise is not None else torch.randn(shape, device=cond.device, generator=generator)
        ratio = self.T // n_steps
        steps = (torch.arange(n_steps) * ratio).flip(0).tolist()       # e.g. 90, 80, ..., 0
        for t in steps:
            tb = torch.full((shape[0],), t, device=cond.device, dtype=torch.long)
            eps = model(x, tb, cond)
            ab_t = self.alphas_cumprod[t]
            ab_prev = self.alphas_cumprod[t - ratio] if t - ratio >= 0 else torch.tensor(1.0, device=cond.device)
            x0 = ((x - (1 - ab_t).sqrt() * eps) / ab_t.sqrt()).clamp(-1, 1)
            eps = (x - ab_t.sqrt() * x0) / (1 - ab_t).sqrt()                # re-derive after clipping
            x = ab_prev.sqrt() * x0 + (1 - ab_prev).sqrt() * eps
        return x


class EMA:
    """Power-law EMA warm-up as in Diffusion Policy: decay = 1 - (1 + step)^-0.75, capped."""

    def __init__(self, model, power: float = 0.75, max_decay: float = 0.9999):
        self.model = copy.deepcopy(model).eval().requires_grad_(False)
        self.power, self.max_decay, self.step = power, max_decay, 0

    @torch.no_grad()
    def update(self, model):
        self.step += 1
        d = min(self.max_decay, max(0.0, 1 - (1 + self.step) ** -self.power))
        for pe, p in zip(self.model.parameters(), model.parameters()):
            pe.mul_(d).add_(p.detach(), alpha=1 - d)
        for be, b in zip(self.model.buffers(), model.buffers()):
            be.copy_(b)
