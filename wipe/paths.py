"""Boustrophedon lane path over the L x L region, in region coordinates (u1, u2).

Lane i runs along t1 at u2 = i * s, alternating direction; consecutive lanes
are joined by a step of length s along t2 at the region edge. The path starts
at the given in-plane point (where contact was made) and first moves to the
corner p0 = (0, 0).
"""
from __future__ import annotations

import numpy as np

from .config import TASK


class LanePath:
    def __init__(self, scene, start_uv=(0.0, 0.0), n_lanes: int = TASK.n_lanes):
        L = scene.L
        s = L / (n_lanes - 1)
        pts = [tuple(start_uv), (0.0, 0.0)]
        for i in range(n_lanes):
            a, b = (0.0, L) if i % 2 == 0 else (L, 0.0)
            pts.append((a, i * s))
            pts.append((b, i * s))
        P = np.array(pts, float)
        # drop zero-length segments
        keep = [0] + [i for i in range(1, len(P)) if np.linalg.norm(P[i] - P[i - 1]) > 1e-9]
        self.P = P[keep]
        seg = np.linalg.norm(np.diff(self.P, axis=0), axis=1)
        self.cum = np.concatenate([[0.0], np.cumsum(seg)])
        self.length = float(self.cum[-1])
        self.corners = self.cum[1:-1].copy()       # arc lengths of interior waypoints

    def speed_scale(self, s: float, slow_dist: float = 0.0, min_scale: float = 1.0) -> float:
        """Speed multiplier that dips to min_scale within slow_dist of a corner."""
        if slow_dist <= 0 or len(self.corners) == 0:
            return 1.0
        d = float(np.min(np.abs(self.corners - s)))
        return float(np.clip(d / slow_dist, min_scale, 1.0))

    def at(self, s: float):
        s = float(np.clip(s, 0.0, self.length))
        i = int(np.searchsorted(self.cum, s, side="right") - 1)
        i = min(i, len(self.P) - 2)
        seg = self.cum[i + 1] - self.cum[i]
        a = 0.0 if seg <= 0 else (s - self.cum[i]) / seg
        u = self.P[i] + a * (self.P[i + 1] - self.P[i])
        return float(u[0]), float(u[1])
