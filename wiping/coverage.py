"""Dirt grid over the region and the uncovered fraction u_t.

A cell is cleaned once the contact point passes within the patch radius r of
its center while the normal force is at least f_min. Between two consecutive
samples that both satisfy the force condition we clean along the whole segment,
so the 20 Hz sampling does not leave gaps.
"""

from __future__ import annotations

import numpy as np

from .config import EpisodeConfig


def _seg_dist(pts, a, b):
    ab = b - a
    den = float(ab @ ab)
    if den < 1e-16:
        return np.linalg.norm(pts - a, axis=1)
    t = np.clip((pts - a) @ ab / den, 0.0, 1.0)
    return np.linalg.norm(pts - (a + t[:, None] * ab), axis=1)


class CoverageGrid:
    def __init__(self, cfg: EpisodeConfig):
        reg = cfg.region
        n = max(2, int(round(reg.L / reg.cell)))
        c = (np.arange(n) + 0.5) * (reg.L / n)
        uu, vv = np.meshgrid(c, c, indexing="ij")
        self.n = n
        self.centers = np.stack([uu.ravel(), vv.ravel()], 1) + reg.corner_uv()  # table (u, v)
        self.clean = np.zeros(len(self.centers), bool)
        self.r = cfg.expert.patch_r
        self.f_min = cfg.f_min()
        self.table = cfg.table
        self._last = None  # last good contact point (u, v)

    def contact_uv(self, p_tcp):
        """Contact point of the ball tool = tcp projected onto the table plane."""
        return self.table.to_table(p_tcp)[:2]

    def update(self, p_tcp, f) -> float:
        """Add one sample (tcp position, contact force on the tool). Returns u_t."""
        fn = float(np.asarray(f) @ self.table.normal())
        if fn >= self.f_min:
            uv = self.contact_uv(p_tcp)
            a = self._last if self._last is not None else uv
            self.clean |= _seg_dist(self.centers, a, uv) <= self.r
            self._last = uv
        else:
            self._last = None
        return self.uncovered()

    def uncovered(self) -> float:
        return float(1.0 - self.clean.mean())

    def image(self):
        """n x n bool array (index [i_u, i_v]) of cleaned cells."""
        return self.clean.reshape(self.n, self.n)


def coverage_from_log(cfg: EpisodeConfig, p, f) -> np.ndarray:
    """Recompute u_t for a whole logged trajectory (T x 3 arrays)."""
    g = CoverageGrid(cfg)
    return np.array([g.update(pk, fk) for pk, fk in zip(p, f)])
