"""CSV dataset, observation features, Diffusion Policy windows, normalization.

Depends only on numpy / pandas (no pydrake), so training runs on Colab.

Dataset layout (scripts/export_dataset.py, README "Data"):
  <root>/K{K_n}/{train,val,test}.csv   one row per 20 Hz step: observation + all four labels
  <root>/K{K_n}/episodes.csv            one row per episode: scene, table, expert metrics
  <root>/scenes/{val,test}.csv          closed-loop evaluation scenes
<root> is a local directory or a Hugging Face dataset repo id (downloaded on first use).

Observation (identical for every label, same order as wipe.data.obs_vector):
  o_t = (x_t, xdot_t, f_t, p0, L, n)                                    16 numbers, world frame
With frame="table" the policy sees the same information in the region frame (t1, t2, n):
  o_t = (R^T (x_t - p0), R^T xdot_t, R^T f_t, L, n)                     13 numbers
and predicts actions in that frame (an ablation; the default is the proposal's world frame).

Windows: at step t the policy gets observations t-To+1..t (clamped at 0) and the labels
t..t+H-1 (the last label repeats past the end of the episode). Actions are relative to the
current ball position: a_j = label_{t+j} - x_t.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

XYZ = ("x", "y", "z")


def cols(prefix: str):
    return [f"{prefix}_{c}" for c in XYZ]


OBS_COLS = cols("x") + cols("xdot") + cols("f") + cols("p0") + ["L"] + cols("n")
OBS_DIM = len(OBS_COLS)                               # 16
LABELS = ("a", "b", "c", "cp")
LABEL_COLS = {lab: cols(lab) for lab in LABELS}
STEP_COLS = (["episode_id", "step", "t", "phase"] + OBS_COLS + ["fn", "u"]
             + [c for lab in LABELS for c in LABEL_COLS[lab]])
PHASES = {0: "approach", 1: "press", 2: "sweep", 3: "lift", 4: "return"}
FRAMES = ("world", "table")


# ---------------------------------------------------------------- loading
def resolve_data(data: str, K: int | None = None) -> str:
    """Local dataset directory, or a Hugging Face dataset repo id ("user/name") that is
    downloaded (only the K{K} folder and the scene lists) into the HF cache."""
    if os.path.isdir(data):
        return data
    from huggingface_hub import snapshot_download
    pats = ["README.md", "stats.json", "scenes/*"] + ([f"K{K}/*"] if K is not None else ["K*/*"])
    return snapshot_download(repo_id=data, repo_type="dataset", allow_patterns=pats)


def load_steps(root: str, K: int, split: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(root, f"K{K}", f"{split}.csv"))


def load_episodes(root: str, K: int) -> pd.DataFrame:
    return pd.read_csv(os.path.join(root, f"K{K}", "episodes.csv"))


def episode_arrays(df: pd.DataFrame, label: str, max_episodes: int | None = None):
    """-> list of (episode_id, obs (T, 16), lab (T, 3)), in file order."""
    out = []
    for eid, g in df.groupby("episode_id", sort=False):
        g = g.sort_values("step")
        out.append((eid, g[OBS_COLS].to_numpy(np.float32), g[LABEL_COLS[label]].to_numpy(np.float32)))
        if max_episodes and len(out) >= max_episodes:
            break
    return out


# ---------------------------------------------------------------- frames
def table_rotation(n: np.ndarray) -> np.ndarray:
    """n (..., 3) -> R (..., 3, 3) with columns (t1, t2, n); t1 = world x projected onto the
    plane (the scenes' lane direction), t2 = n x t1."""
    n = np.asarray(n, np.float64)
    ex = np.broadcast_to(np.array([1.0, 0.0, 0.0]), n.shape)
    t1 = ex - (ex * n).sum(-1, keepdims=True) * n
    t1 /= np.linalg.norm(t1, axis=-1, keepdims=True)
    t2 = np.cross(n, t1)
    return np.stack([t1, t2, n], axis=-1)


def obs_dim(frame: str) -> int:
    return OBS_DIM if frame == "world" else 13


def featurize(O: np.ndarray, frame: str) -> np.ndarray:
    """Raw 16-number observations (..., 16) -> model features (..., obs_dim(frame))."""
    if frame == "world":
        return np.asarray(O, np.float32)
    O = np.asarray(O, np.float64)
    x, xd, f, p0, L, n = O[..., 0:3], O[..., 3:6], O[..., 6:9], O[..., 9:12], O[..., 12:13], O[..., 13:16]
    R = table_rotation(n)
    loc = lambda v: np.einsum("...ji,...j->...i", R, v)          # R^T v
    return np.concatenate([loc(x - p0), loc(xd), loc(f), L, n], -1).astype(np.float32)


def actions_to_frame(A: np.ndarray, n: np.ndarray, frame: str) -> np.ndarray:
    """Relative world-frame actions (B, H, 3), table normals (B, 3) -> model frame."""
    if frame == "world":
        return A
    return np.einsum("bji,bhj->bhi", table_rotation(n), A).astype(np.float32)


def actions_from_frame(A: np.ndarray, n: np.ndarray, frame: str) -> np.ndarray:
    if frame == "world":
        return A
    return np.einsum("bij,bhj->bhi", table_rotation(n), A).astype(np.float32)


# ---------------------------------------------------------------- windows
def make_windows(obs: np.ndarray, lab: np.ndarray, To: int = 2, H: int = 16):
    """obs (T, 16), lab (T, 3) of one episode -> O (T, To, 16) raw, A (T, H, 3) relative world."""
    T = len(lab)
    t = np.arange(T)
    oi = np.clip(t[:, None] + np.arange(-To + 1, 1)[None, :], 0, None)
    ai = np.clip(t[:, None] + np.arange(H)[None, :], None, T - 1)
    A = lab[ai] - obs[t, None, 0:3]
    return obs[oi].astype(np.float32), A.astype(np.float32)


def build_windows(df: pd.DataFrame, label: str, To: int = 2, H: int = 16, frame: str = "world",
                  max_episodes: int | None = None):
    """-> O (N, To, obs_dim) model features, A (N, H, 3) model-frame actions, ep (N,) episode index."""
    Os, As, E = [], [], []
    for ei, (_, obs, lab) in enumerate(episode_arrays(df, label, max_episodes)):
        O, A = make_windows(obs, lab, To, H)
        As.append(actions_to_frame(A, O[:, -1, 13:16], frame))
        Os.append(featurize(O, frame))
        E.append(np.full(len(O), ei))
    return np.concatenate(Os), np.concatenate(As), np.concatenate(E)


# ---------------------------------------------------------------- normalization
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
