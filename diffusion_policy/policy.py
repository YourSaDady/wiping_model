"""Policies that map an observation history to a chunk of world-frame targets x_d.

DiffusionPolicy: loads a run directory written by diffusion_policy/train.py.
NearestNeighborPolicy: harness smoke test without training; returns the action chunk of the
nearest training window (normalized observation distance).
Both expose act(O, x_now, seeds, calls) -> (B, H, 3) world targets, the interface of
wipe/evaluate.py: O (B, To, 16) raw observations o_t = (x, xdot, f, p0, L, n).
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch

from .dataset import Normalizer, actions_from_frame, build_windows, featurize, load_steps, resolve_data
from .diffusion import DDPM
from .model import ConditionalUnet1D
from .train import pick_device


class DiffusionPolicy:
    def __init__(self, run_dir: str, ckpt: str | None = "best", device: str = "auto", ddim_steps: int | None = None):
        """ckpt: "best" (lowest validation chunk error), "last", a step number, or a path."""
        self.cfg = json.load(open(os.path.join(run_dir, "config.json")))
        self.on = Normalizer.from_state(self.cfg["obs_norm"])
        self.an = Normalizer.from_state(self.cfg["act_norm"])
        self.To, self.H, self.frame = self.cfg["To"], self.cfg["H"], self.cfg["frame"]
        self.device = pick_device(device)
        dims = tuple(int(d) for d in self.cfg["down_dims"].split(","))
        self.model = ConditionalUnet1D(3, self.To * self.cfg["obs_dim"], down_dims=dims).to(self.device).eval()
        self.ckpt = self.resolve(run_dir, ckpt)
        self.model.load_state_dict(torch.load(self.ckpt, map_location=self.device, weights_only=False)["ema"])
        self.ddpm = DDPM(self.cfg["diffusion_steps"], self.device)
        self.ddim_steps = ddim_steps or self.cfg["ddim_steps"]

    @staticmethod
    def resolve(run_dir, ckpt):
        if ckpt in (None, "last"):
            snaps = sorted((int(f[4:-3]), f) for f in os.listdir(run_dir)
                           if f.startswith("ema_") and f.endswith(".pt") and f[4:-3].isdigit())
            return os.path.join(run_dir, snaps[-1][1])
        if ckpt == "best":
            return os.path.join(run_dir, "ema_best.pt")
        if str(ckpt).isdigit():
            return os.path.join(run_dir, f"ema_{ckpt}.pt")
        return ckpt

    @torch.no_grad()
    def act(self, O: np.ndarray, x_now: np.ndarray, seeds, calls) -> np.ndarray:
        """seeds / calls (B,) ints fix the initial DDIM noise per (episode, inference call), so
        results do not depend on how environments are batched."""
        B = len(O)
        F = featurize(O, self.frame)
        cond = torch.tensor(self.on.norm(F).reshape(B, -1), device=self.device)
        noise = torch.stack([torch.randn((self.H, 3), generator=torch.Generator().manual_seed(
            int(s) * 100003 + int(c))) for s, c in zip(seeds, calls)]).to(self.device)
        pred = self.ddpm.ddim_sample(self.model, cond, (B, self.H, 3), self.ddim_steps, noise=noise)
        A = actions_from_frame(self.an.unnorm(pred.cpu().numpy()), O[:, -1, 13:16], self.frame)
        return A + x_now[:, None, :]


class NearestNeighborPolicy:
    def __init__(self, data: str, K: int, label: str, To: int = 2, H: int = 16, device: str = "auto"):
        O, A, _ = build_windows(load_steps(resolve_data(data, K), K, "train"), label, To, H, "world")
        self.on = Normalizer.fit(O)
        self.To, self.H, self.frame = To, H, "world"
        self.device = pick_device(device)
        self.keys = torch.tensor(self.on.norm(O).reshape(len(O), -1), device=self.device)
        self.A = A
        self.cfg = dict(K=K, label=label, To=To, H=H)

    @torch.no_grad()
    def act(self, O, x_now, seeds=None, calls=None):
        q = torch.tensor(self.on.norm(O.astype(np.float32)).reshape(len(O), -1), device=self.device)
        idx = torch.cdist(q, self.keys).argmin(1).cpu().numpy()
        return self.A[idx] + x_now[:, None, :]
