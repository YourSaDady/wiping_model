"""Coverage metric on the L x L region.

Each grid cell (size TASK.grid_res) accumulates a wiping dose from every sample
at which its centre lies within r_wipe of the tool's contact point. The cell is
wiped once its dose reaches w_min. Two ways to measure the dose
(TASK.coverage_mode):

  "work"   friction work: each sample adds P dt, with P = max(0, -f_t . v_t) the
           power dissipated by the tangential contact force f_t on the sliding
           tool (Archard: worn volume ~ normal load x sliding distance). Pressing
           without sliding, or sliding without pressing, adds nothing; a cell
           under a pass at f_des collects mu f_des x (chord of the patch).
  "force"  the proposal's rule: a sample with f_n >= f_min wipes every cell
           under the patch at once (dose set to w_min).

Samples come at 100 Hz (SUBSTEPS per policy step); at the fastest wiping speed
(12 cm/s) consecutive samples are 1.2 mm apart, below the cell size. The contact
point is the ball centre projected onto the undeformed surface.
"""
from __future__ import annotations

import numpy as np

from .config import TASK, TaskParams


class Coverage:
    def __init__(self, L: float, task: TaskParams = TASK):
        if task.coverage_mode not in ("work", "force"):
            raise ValueError(f"unknown coverage_mode {task.coverage_mode!r}")
        self.task = task
        self.res = task.grid_res
        self.n = int(np.ceil(L / self.res))
        self.centers = (np.arange(self.n) + 0.5) * (L / self.n)
        self.cell = L / self.n
        self.dose = np.zeros((self.n, self.n))      # [J] in "work" mode
        rr = int(np.ceil(task.r_wipe / self.cell)) + 1
        di, dj = np.meshgrid(np.arange(-rr, rr + 1), np.arange(-rr, rr + 1), indexing="ij")
        self._di, self._dj = di.ravel(), dj.ravel()

    def _under_patch(self, u1: float, u2: float):
        i0 = int(np.floor(u1 / self.cell))
        j0 = int(np.floor(u2 / self.cell))
        i = i0 + self._di
        j = j0 + self._dj
        ok = (i >= 0) & (i < self.n) & (j >= 0) & (j < self.n)
        i, j = i[ok], j[ok]
        d2 = (self.centers[i] - u1) ** 2 + (self.centers[j] - u2) ** 2
        m = d2 <= self.task.r_wipe ** 2
        return i[m], j[m]

    def add(self, u1: float, u2: float, fn: float, power: float = 0.0, dt: float = 0.0) -> None:
        """One sample: contact point (u1, u2) in region coordinates, normal force fn,
        friction power (>= 0) and the time the sample stands for ("work" mode)."""
        if self.task.coverage_mode == "force":
            if fn < self.task.f_min:
                return
            i, j = self._under_patch(u1, u2)
            self.dose[i, j] = np.maximum(self.dose[i, j], self.task.w_min)
        else:
            w = max(power, 0.0) * dt
            if w <= 0.0:
                return
            i, j = self._under_patch(u1, u2)
            self.dose[i, j] += w

    @property
    def progress(self) -> np.ndarray:
        """Per-cell fraction of the required dose, clipped to [0, 1]."""
        return np.minimum(self.dose / self.task.w_min, 1.0)

    @property
    def grid(self) -> np.ndarray:
        """Boolean map of wiped cells."""
        return self.dose >= self.task.w_min

    @property
    def uncovered(self) -> float:
        return 1.0 - float(self.grid.mean())
