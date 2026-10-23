"""Policies that map an observation history to a chunk of world-frame targets.

DiffusionPolicy: loads a run directory from scripts/train_dp.py.
NearestNeighborPolicy: harness smoke test; returns the relative action chunk
of the nearest training window (normalized observation distance).
Both expose act(O, x_now, seeds, calls) -> (B, H, 3) world targets.
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch

from .config import GAINS_BY_KN
from .dp.dataset import Normalizer, build_windows
from .dp.diffusion import DDPM
from .dp.model import ConditionalUnet1D
from .labels import rebuild_target


class DiffusionPolicy:
    def __init__(self, run_dir: str, ckpt: str | None = None, device="cuda", ddim_steps: int | None = None):
        self.cfg = json.load(open(os.path.join(run_dir, "config.json")))
        self.on = Normalizer.from_state(self.cfg["obs_norm"])
        self.an = Normalizer.from_state(self.cfg["act_norm"])
        self.To, self.H, self.relative = self.cfg["To"], self.cfg["H"], self.cfg["relative"]
        self.obs_version = self.cfg.get("obs_version", 1)
        dims = tuple(int(d) for d in self.cfg["down_dims"].split(","))
        self.adim = len(self.an.lo)
        self.label = self.cfg["label"]
        self.deploy_gains = GAINS_BY_KN[self.cfg["K"]]    # gains used to rebuild "pf" targets
        self.model = ConditionalUnet1D(self.adim, len(self.on.lo) * self.To, down_dims=dims).to(device).eval()
        ckpt = ckpt or self.latest(run_dir)
        self.model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False)["ema"])
        self.ckpt = ckpt
        self.ddpm = DDPM(100, device)
        self.device = device
        self.ddim_steps = ddim_steps or self.cfg.get("ddim_steps", 16)
        self.use_graph = device.startswith("cuda") and os.environ.get("WIPE_NO_CUDA_GRAPH") != "1"
        self._graphs = {}

    def _sample(self, cond, noise):
        """DDIM sampling; on CUDA the 16-step loop is captured once per padded batch
        size as a CUDA graph (same arithmetic, ~1 launch instead of thousands)."""
        B = cond.shape[0]
        if not self.use_graph:
            return self.ddpm.ddim_sample(self.model, cond, noise.shape, self.ddim_steps, noise=noise)
        Bp = max(8, 1 << (B - 1).bit_length())
        if Bp not in self._graphs:
            sc = torch.zeros((Bp,) + cond.shape[1:], device=self.device)
            sn = torch.zeros((Bp,) + noise.shape[1:], device=self.device)
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(2):
                    self.ddpm.ddim_sample(self.model, sc, sn.shape, self.ddim_steps, noise=sn)
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                out = self.ddpm.ddim_sample(self.model, sc, sn.shape, self.ddim_steps, noise=sn)
            self._graphs[Bp] = (g, sc, sn, out)
        g, sc, sn, out = self._graphs[Bp]
        sc.zero_(); sn.zero_()
        sc[:B].copy_(cond); sn[:B].copy_(noise)
        g.replay()
        return out[:B].clone()

    @staticmethod
    def latest(run_dir):
        snaps = sorted((int(f[4:-3]), f) for f in os.listdir(run_dir) if f.startswith("ema_") and f.endswith(".pt"))
        return os.path.join(run_dir, snaps[-1][1])

    @torch.no_grad()
    def act(self, O: np.ndarray, x_now: np.ndarray, seeds, calls) -> np.ndarray:
        """O (B, To, 16) raw observations; x_now (B, 3); seeds/calls (B,) ints that
        fix the initial DDIM noise per (episode, inference call) so results do
        not depend on how environments are batched."""
        B = len(O)
        cond = torch.tensor(self.on.norm(O.astype(np.float32)).reshape(B, -1), device=self.device)
        noise = torch.stack([torch.randn((self.H, self.adim), generator=torch.Generator().manual_seed(
            int(s) * 100003 + int(c))) for s, c in zip(seeds, calls)]).to(self.device)
        pred = self._sample(cond, noise)
        A = self.an.unnorm(pred.cpu().numpy())
        if self.relative:
            A[..., :3] += x_now[:, None, :]
        if self.label == "pf":            # rebuild targets with the deployment gains, in each table's frame
            g = self.deploy_gains
            A = np.stack([rebuild_target(A[i], g.K(O[i, -1, 13:16]), g.D(O[i, -1, 13:16])) for i in range(B)])
        return A


class NearestNeighborPolicy:
    def __init__(self, train_paths, label, K_n, To=2, H=16, device="cuda", obs_version=1):
        O, A, _ = build_windows(train_paths, label, K_n, To, H, relative=True, obs_version=obs_version)
        self.on = Normalizer.fit(O)
        self.To, self.H, self.relative, self.obs_version = To, H, True, obs_version
        self.keys = torch.tensor(self.on.norm(O).reshape(len(O), -1), device=device)
        self.A = A
        self.device = device

    @torch.no_grad()
    def act(self, O, x_now, seeds=None, calls=None):
        q = torch.tensor(self.on.norm(O.astype(np.float32)).reshape(len(O), -1), device=self.device)
        idx = torch.cdist(q, self.keys).argmin(1).cpu().numpy()
        return self.A[idx] + x_now[:, None, :]
