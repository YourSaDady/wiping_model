"""Coverage metric on the L x L region.

A grid cell (size TASK.grid_res) counts as wiped once its centre lies within
r_wipe of the tool's contact point at a sample where the normal force is at
least f_min. Samples come at 100 Hz (SUBSTEPS per policy step); at the fastest
wiping speed (12 cm/s) consecutive samples are 1.2 mm apart, below the cell size.
The contact point is the ball centre projected onto the undeformed surface.
"""
from __future__ import annotations

import numpy as np

from .config import TASK, TaskParams


class Coverage:
    def __init__(self, L: float, task: TaskParams = TASK):
        self.task = task
        self.res = task.grid_res
        self.n = int(np.ceil(L / self.res))
        self.centers = (np.arange(self.n) + 0.5) * (L / self.n)
        self.cell = L / self.n
        self.grid = np.zeros((self.n, self.n), dtype=bool)
        rr = int(np.ceil(task.r_wipe / self.cell)) + 1
        di, dj = np.meshgrid(np.arange(-rr, rr + 1), np.arange(-rr, rr + 1), indexing="ij")
        self._di, self._dj = di.ravel(), dj.ravel()

    def add(self, u1: float, u2: float, fn: float) -> None:
        if fn < self.task.f_min:
            return
        i0 = int(np.floor(u1 / self.cell))
        j0 = int(np.floor(u2 / self.cell))
        i = i0 + self._di
        j = j0 + self._dj
        ok = (i >= 0) & (i < self.n) & (j >= 0) & (j < self.n)
        if not ok.any():
            return
        i, j = i[ok], j[ok]
        d2 = (self.centers[i] - u1) ** 2 + (self.centers[j] - u2) ** 2
        m = d2 <= self.task.r_wipe ** 2
        self.grid[i[m], j[m]] = True

    @property
    def uncovered(self) -> float:
        return 1.0 - float(self.grid.mean())
