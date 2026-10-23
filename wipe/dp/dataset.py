"""Diffusion Policy windows and min-max normalization.

A window at step t of an episode holds the observations at t-To+1..t (clamped
at 0) and the labels at t..t+H-1 (the last label is repeated past the end of
the episode, as in Diffusion Policy's sequence sampler). With relative actions
every label in the window is expressed relative to the current ball position
x_t: a_j = label_{t+j} - x_t.
"""
from __future__ import annotations

import numpy as np

from ..config import GAINS_BY_KN
from ..data import load_episode, obs_vector
from ..labels import make_labels


class Normalizer:
    """Per-dimension min-max to [-1, 1]; constant dimensions map to 0."""

    def __init__(self, lo, hi):
        self.lo, self.hi = np.asarray(lo, np.float32), np.asarray(hi, np.float32)
        rng = self.hi - self.lo
        self.scale = np.where(rng > 1e-6, 2.0 / np.maximum(rng, 1e-12), 1.0).astype(np.float32)
        self.mid = (0.5 * (self.hi + self.lo)).astype(np.float32)

    @staticmethod
    def fit(data):
        d = np.asarray(data).reshape(-1, np.asarray(data).shape[-1])
        return Normalizer(d.min(0), d.max(0))

    def norm(self, x):
        return (x - self.mid) * self.scale

    def unnorm(self, y):
        return y / self.scale + self.mid

    def state_dict(self):
        return {"lo": self.lo.tolist(), "hi": self.hi.tolist()}

    @staticmethod
    def from_state(d):
        return Normalizer(d["lo"], d["hi"])


def episode_arrays(ep, label: str, K_n: int, obs_version: int = 1):
    sc = ep["scene"]
    g = GAINS_BY_KN[K_n]
    n = sc.arr("n")
    obs = obs_vector(ep["x"], ep["xdot"], ep["f"], sc, ep["u"], obs_version)   # (T+1, D)
    lab = make_labels(ep["x"], ep["xdot"], ep["f"], ep["x_d"], g.K(n), g.D(n), label)   # (T, 3)
    return obs.astype(np.float32), lab.astype(np.float32)


def make_windows(obs, lab, x, To=2, H=16, relative=True):
    """obs (T+1, D), lab (T, 3), x (T+1, 3) -> O (T, To, D), A (T, H, 3)."""
    T = lab.shape[0]
    t = np.arange(T)
    oi = np.clip(t[:, None] + np.arange(-To + 1, 1)[None, :], 0, None)
    ai = np.clip(t[:, None] + np.arange(H)[None, :], None, T - 1)
    O = obs[oi]
    A = lab[ai]
    if relative:
        A = A - x[t][:, None, :]
    return O.astype(np.float32), A.astype(np.float32)


def build_windows(paths, label: str, K_n: int, To=2, H=16, relative=True, obs_version: int = 1):
    Os, As, meta = [], [], []
    for ei, p in enumerate(paths):
        ep = load_episode(p)
        obs, lab = episode_arrays(ep, label, K_n, obs_version)
        O, A = make_windows(obs, lab, ep["x"], To, H, relative)
        Os.append(O)
        As.append(A)
        meta.append(np.stack([np.full(len(O), ei), np.arange(len(O))], 1))
    return np.concatenate(Os), np.concatenate(As), np.concatenate(meta)
