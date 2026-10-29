"""Observations, the action labels of the proposal, and the .npz demo format.

For step t of an episode (t = 0..N-1; state sample N exists):

  obs   o_t = (x_t, xdot_t, f_t, p0, L, n)                      16 dims
  (a)   label_a_measured   = x_{t+1}
  (b)   label_b_commanded  = x_{d,t}
  (c)   label_c_force_comp = x_{t+1} - K^-1 f_{t+1}
  (c')  label_c_damp       = x_{t+1} + K^-1 (D xdot_{t+1} - f_{t+1})
  (d)   label_d_pose_force = [x_{t+1}, f_{t+1}]   (stretch goal; target x - K'^-1 f)

K and D are the 3x3 world matrices of that episode (they differ across tilts).
f is the force the table applies to the tool, so x - K^-1 f sits below the
surface while pressing.

File layout (`save_demos`): per-step arrays concatenated over episodes,
`episode_ends[i]` = exclusive end row of episode i (Diffusion-Policy style),
per-episode arrays (K, D, p0, L, n, success, ...) with one row per episode,
`ep_index` mapping each step to its episode, and `meta` = JSON list of
{config, metrics} per episode.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

OBS_NAMES = ["x"] * 3 + ["xdot"] * 3 + ["f"] * 3 + ["p0"] * 3 + ["L"] + ["n"] * 3
OBS_DIM = len(OBS_NAMES)  # 16
LABELS = ["label_a_measured", "label_b_commanded", "label_c_force_comp", "label_c_damp", "label_d_pose_force"]


def make_labels(ep: dict) -> dict:
    K, D = np.asarray(ep["K"]), np.asarray(ep["D"])
    Kinv = np.linalg.inv(K)
    x1, v1, f1 = ep["x"][1:], ep["xdot"][1:], ep["f"][1:]
    return {
        "label_a_measured": x1.copy(),
        "label_b_commanded": ep["x_d"].copy(),
        "label_c_force_comp": x1 - f1 @ Kinv.T,
        "label_c_damp": x1 + (v1 @ D.T - f1) @ Kinv.T,
        "label_d_pose_force": np.concatenate([x1, f1], axis=1),
    }


def make_obs(ep: dict) -> np.ndarray:
    cfg = ep["cfg"]
    N = len(ep["x_d"])
    p0 = cfg.region.p0(cfg.table)
    n = cfg.table.normal()
    ctx = np.concatenate([p0, [cfg.region.L], n])
    return np.concatenate([ep["x"][:N], ep["xdot"][:N], ep["f"][:N], np.tile(ctx, (N, 1))], axis=1)


def episode_arrays(ep: dict) -> dict:
    """All per-step arrays (N rows) for one logged episode."""
    N = len(ep["x_d"])
    out = dict(
        obs=make_obs(ep),
        t=ep["t"][:N], x=ep["x"][:N], xdot=ep["xdot"][:N], f=ep["f"][:N], f_n=ep["f_n"][:N],
        x_d=ep["x_d"], u=ep["u"][:N], phase=ep["phase"], q=ep["q"][:N], v=ep["v"][:N],
        x_next=ep["x"][1:], xdot_next=ep["xdot"][1:], f_next=ep["f"][1:], u_next=ep["u"][1:],
    )
    out.update(make_labels(ep))
    return out


class DemoWriter:
    """Accumulate episodes and write one .npz."""

    def __init__(self):
        self.steps = []
        self.eps = []

    def add(self, ep: dict):
        self.steps.append(episode_arrays(ep))
        cfg = ep["cfg"]
        self.eps.append(dict(
            K=ep["K"], D=ep["D"], p0=cfg.region.p0(cfg.table), L=cfg.region.L, n=cfg.table.normal(),
            k_e=cfg.table.k_e, tilt=cfg.table.tilt, mu=cfg.table.mu, f_des=cfg.expert.f_des,
            speed=cfg.expert.speed, success=ep["metrics"]["success"], u_T=ep["metrics"]["u_T"],
            meta=dict(config=cfg.to_dict(), metrics=ep["metrics"]),
        ))

    def __len__(self):
        return len(self.eps)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not self.steps:
            raise ValueError("no episodes to save")
        data = {k: np.concatenate([s[k] for s in self.steps]) for k in self.steps[0]}
        lens = [len(s["x_d"]) for s in self.steps]
        data["episode_ends"] = np.cumsum(lens).astype(np.int64)
        data["ep_index"] = np.repeat(np.arange(len(lens)), lens).astype(np.int32)
        for k in ["K", "D", "p0", "L", "n", "k_e", "tilt", "mu", "f_des", "speed", "success", "u_T"]:
            data[k] = np.asarray([e[k] for e in self.eps])
        data["meta"] = np.array(json.dumps([e["meta"] for e in self.eps]))
        data["obs_names"] = np.array(OBS_NAMES)
        np.savez_compressed(path, **data)
        return path


def load_demos(path) -> dict:
    """Load a demo file; `meta` comes back as a list of dicts."""
    with np.load(path, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    d["meta"] = json.loads(str(d["meta"]))
    return d


def episode_slices(d: dict):
    ends = d["episode_ends"]
    starts = np.concatenate([[0], ends[:-1]])
    return [slice(int(a), int(b)) for a, b in zip(starts, ends)]
