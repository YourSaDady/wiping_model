"""Boustrophedon (back-and-forth) lane pattern over the L x L region.

Lanes run along t1 (u) and are stacked along t2 (v). With patch radius r and
lane spacing s < 2r, the margin m = min(s/2, 0.6 r) from each edge keeps every
point of the square within r of the path:
  * between lanes the worst point is s'/2 <= s/2 < r away,
  * at an edge it is m < r away, at a corner sqrt(2) m < r (m <= 0.6 r).
"""

from __future__ import annotations

import math

import numpy as np

from .config import ExpertConfig, RegionConfig


def plan_lanes(region: RegionConfig, expert: ExpertConfig) -> np.ndarray:
    """Waypoints (M x 2) in table coordinates (u, v)."""
    L, s, m = region.L, expert.lane_spacing, expert.margin()
    if s >= 2 * expert.patch_r:
        raise ValueError(f"lane spacing s={s} must be < 2r={2 * expert.patch_r}")
    span = L - 2 * m
    n_lanes = max(1, int(math.ceil(span / s - 1e-9)) + 1)
    vs = np.linspace(m, L - m, n_lanes) if n_lanes > 1 else np.array([0.5 * L])
    u0, u1 = m, L - m
    pts = []
    for i, v in enumerate(vs):
        a, b = (u0, u1) if i % 2 == 0 else (u1, u0)
        pts += [(a, v), (b, v)]
    return np.asarray(pts) + region.corner_uv()


class PathFollower:
    """Constant-speed traversal of a polyline; `advance(ds)` returns (point, unit tangent, done)."""

    def __init__(self, waypoints: np.ndarray):
        self.w = np.asarray(waypoints, float)
        seg = np.diff(self.w, axis=0)
        self.seg_len = np.linalg.norm(seg, axis=1)
        self.cum = np.concatenate([[0.0], np.cumsum(self.seg_len)])
        self.length = float(self.cum[-1])
        self.s = 0.0

    def point(self, s: float):
        s = min(max(s, 0.0), self.length)
        i = int(np.searchsorted(self.cum, s, side="right") - 1)
        i = min(i, len(self.seg_len) - 1)
        tang = (self.w[i + 1] - self.w[i]) / max(self.seg_len[i], 1e-12)
        return self.w[i] + (s - self.cum[i]) * tang, tang

    def advance(self, ds: float):
        self.s = min(self.length, self.s + ds)
        p, t = self.point(self.s)
        return p, t, self.s >= self.length - 1e-12
