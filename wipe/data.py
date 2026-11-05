"""Demo storage and the observation vector shared by every policy.

Episodes are stored one per file: data/demos/K{K_n}/{table}/{index:03d}.npz for
the demo split, data/expert_{split}/K{K_n}/{table}/{index:03d}.npz for expert runs
on val/test/dev scenes (offline test set and expert reference).
The observation is identical for every label:
  o_t = (x_t, xdot_t, f_t, p0, L, n)  -> 16 numbers
x is the ball-centre position, xdot its velocity, f the contact force on the
tool (world frame), p0 the region corner, L its side, n the table normal.
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np

from .scene import Scene

OBS_DIM = 16
DATA_ROOT = os.environ.get("WIPE_DATA", os.path.join(os.path.dirname(os.path.dirname(__file__)), "data"))


def obs_vector(x, xdot, f, scene: Scene) -> np.ndarray:
    """x, xdot, f: (..., 3) -> (..., 16)."""
    x = np.asarray(x, float)
    lead = x.shape[:-1]
    const = np.concatenate([scene.arr("p0"), [scene.L], scene.arr("n")])
    return np.concatenate([x, np.asarray(xdot, float), np.asarray(f, float),
                           np.broadcast_to(const, lead + (7,))], axis=-1)


def split_dir(K_n: int, split: str = "demo", root: str = DATA_ROOT) -> str:
    return os.path.join(root, "demos" if split == "demo" else f"expert_{split}", f"K{K_n}")


def demo_path(K_n: int, table: str, index: int, root: str = DATA_ROOT, split: str = "demo") -> str:
    return os.path.join(split_dir(K_n, split, root), table, f"{index:03d}.npz")


def save_episode(path: str, ep: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arrays = {k: v for k, v in ep.items() if isinstance(v, np.ndarray)}
    arrays["metrics"] = np.array(json.dumps(ep["metrics"]))
    arrays["scene"] = np.array(json.dumps(ep["scene"]))
    for k in ("expert_done",):
        if k in ep:
            arrays[k] = np.array(ep[k])
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def load_episode(path: str) -> dict:
    with np.load(path, allow_pickle=False) as z:
        ep = {k: z[k] for k in z.files}
    ep["metrics"] = json.loads(str(ep["metrics"]))
    ep["scene"] = Scene.from_dict(json.loads(str(ep["scene"])))
    return ep


def list_demos(K_n: int, table: str, root: str = DATA_ROOT, split: str = "demo"):
    return sorted(glob.glob(os.path.join(split_dir(K_n, split, root), table, "[0-9][0-9][0-9].npz")))


def split_paths(K_n: int, tables, n_train: int = 32, n_val: int = 4, root: str = DATA_ROOT):
    """Train / validation demo files, split by episode: per table, the first
    n_train kept demos train and the next n_val validate."""
    man = json.load(open(os.path.join(root, "demos", f"K{K_n}", "manifest.json")))
    tr, va = [], []
    for t in tables:
        kept = man[t]["kept"]
        assert len(kept) >= n_train + n_val, (t, len(kept))
        tr += [demo_path(K_n, t, i, root) for i in kept[:n_train]]
        va += [demo_path(K_n, t, i, root) for i in kept[n_train:n_train + n_val]]
    return tr, va
