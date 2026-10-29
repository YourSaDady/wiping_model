"""Deterministic scene sampling.

A scene fixes everything that varies between episodes on one table: the table
height and position, the wiping region (corner p0, side L), the wiping speed v
used by the scripted expert and baselines, and the start pose above p0.
Scenes are generated from (table name, split, index) so demo, validation and
test scenes never share a seed.
"""
from __future__ import annotations

import zlib
from dataclasses import dataclass, asdict

import numpy as np

from .config import TABLES, TASK, SPLIT_CODE, TableSpec


@dataclass(frozen=True)
class Scene:
    table: str
    split: str
    index: int
    tilt_deg: float
    k_e: float
    c: tuple            # point on the table surface at the region centre [m]
    p0: tuple           # region corner on the surface [m]
    L: float            # region side [m]
    t1: tuple           # lane direction (in-plane)
    t2: tuple           # step direction (in-plane)
    n: tuple            # surface normal
    v: float            # wiping speed for the expert / baselines [m/s]
    start: tuple        # start position of the ball centre [m]

    @property
    def spec(self) -> TableSpec:
        return TABLES[self.table]

    @property
    def seed(self) -> int:
        return scene_seed(self.table, self.split, self.index)

    def arr(self, key) -> np.ndarray:
        return np.asarray(getattr(self, key), float)

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d):
        return Scene(**{k: (tuple(v) if isinstance(v, (list, np.ndarray)) else v) for k, v in d.items()})

    # ---- region geometry helpers
    @property
    def lane_spacing(self) -> float:
        return self.L / (TASK.n_lanes - 1)

    def to_plane(self, p) -> np.ndarray:
        """World point(s) -> (u1, u2, h) in the region frame (h along n from the surface)."""
        d = np.asarray(p, float) - self.arr("p0")
        return np.stack([d @ self.arr("t1"), d @ self.arr("t2"), d @ self.arr("n")], axis=-1)

    def from_plane(self, u1, u2, h=0.0) -> np.ndarray:
        return self.arr("p0") + u1 * self.arr("t1") + u2 * self.arr("t2") + h * self.arr("n")


def scene_seed(table: str, split: str, index: int) -> int:
    ss = np.random.SeedSequence([zlib.crc32(table.encode()), SPLIT_CODE[split], int(index)])
    return int(ss.generate_state(1)[0])


def make_scene(table: str, split: str, index: int, tilt_deg: float | None = None, k_e: float | None = None,
               v: float | None = None, L: float | None = None) -> Scene:
    """Scene `index` of `split` on `table`. The optional overrides replace the table's tilt / k_e or
    the sampled speed / region side after sampling, so the rest of the scene (position, height,
    start pose) stays the one drawn from the same seed (used by scripts/watch_expert.py)."""
    spec = TABLES[table]
    if tilt_deg is not None or k_e is not None:
        spec = TableSpec(spec.name, spec.tilt_deg if tilt_deg is None else float(tilt_deg),
                         spec.k_e if k_e is None else float(k_e), spec.split)
    rng = np.random.default_rng(scene_seed(table, split, index))
    n = spec.normal
    t1 = np.array([1.0, 0.0, 0.0])            # lanes run along world x (fixed in the table frame)
    t2 = np.cross(n, t1)                        # (t1, t2, n) is right-handed
    # Surface point at the region centre. Height is randomized; for a tilted table
    # this is the height of the region centre.
    c = np.array([rng.uniform(0.52, 0.60), rng.uniform(-0.04, 0.04), rng.uniform(0.06, 0.16)])
    L_s = rng.uniform(0.13, 0.17)
    L = L_s if L is None else float(L)
    p0 = c - 0.5 * L * t1 - 0.5 * L * t2
    v_s = rng.uniform(0.05, 0.12)
    v = v_s if v is None else float(v)
    h0 = rng.uniform(0.04, 0.07)                # start height of the ball bottom above the surface
    j1, j2 = rng.uniform(-0.01, 0.01, size=2)
    from .config import BALL_RADIUS
    start = p0 + j1 * t1 + j2 * t2 + (h0 + BALL_RADIUS) * n
    return Scene(table=table, split=split, index=int(index), tilt_deg=spec.tilt_deg, k_e=spec.k_e,
                 c=tuple(c), p0=tuple(p0), L=float(L), t1=tuple(t1), t2=tuple(t2), n=tuple(n),
                 v=float(v), start=tuple(start))
