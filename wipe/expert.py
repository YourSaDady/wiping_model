"""Privileged scripted wiping expert (runs at the 20 Hz policy rate, ZOH).

Knows the table plane and the region. Phases:
  approach: target moves along -n at 2 cm/s, 6 mm/s once the ball is within
            2 cm of the surface, until the measured normal force exceeds f_contact;
  press:    integrate the normal-force error into the target depth,
            h_d <- h_d - (f_des - f_n) / (K_n * tau) * dt, until the force settles;
  sweep:    in-plane target follows the lane path at speed v (slowing to
            CORNER_SCALE * v near corners), force integrator on;
  lift:     target rises along n at 3 cm/s until the ball is 2 cm above the surface.
"""
from __future__ import annotations

import numpy as np

from .config import BALL_RADIUS, CORNER_DIST, CORNER_SCALE, DT_POLICY, TASK, Gains, TaskParams
from .paths import LanePath

APPROACH, PRESS, SWEEP, LIFT, DONE = "approach", "press", "sweep", "lift", "done"
PHASE_ID = {APPROACH: 0, PRESS: 1, SWEEP: 2, LIFT: 3, DONE: 4}


class ScriptedExpert:
    def __init__(self, scene, gains: Gains, task: TaskParams = TASK, tau: float = 0.3,
                 f_contact: float = 1.0, v_fast: float = 0.02, v_slow: float = 0.006,
                 slow_zone: float = 0.02, v_lift: float = 0.03, lift_clear: float = 0.02,
                 corner_dist: float = CORNER_DIST, corner_scale: float = CORNER_SCALE):
        self.scene, self.gains, self.task = scene, gains, task
        self.n = scene.arr("n")
        self.k_int = 1.0 / (gains.K_n * tau)
        self.f_contact, self.v_fast, self.v_slow, self.slow_zone = f_contact, v_fast, v_slow, slow_zone
        self.v_lift, self.lift_clear = v_lift, lift_clear
        self.corner_dist, self.corner_scale = corner_dist, corner_scale
        u = scene.to_plane(scene.arr("start"))
        self.u1, self.u2, self.h = u          # target in region coords (h of the ball centre)
        self.phase = APPROACH
        self.t_phase = 0.0
        self.settle = 0
        self.s = 0.0
        self.path = None

    def act(self, obs) -> np.ndarray:
        dt = DT_POLICY
        x, f = obs["x"], obs["f"]
        u = self.scene.to_plane(x)
        gap = u[2] - BALL_RADIUS
        fn = float(self.n @ f)
        f_des = self.task.f_des
        self.t_phase += dt
        if self.phase == APPROACH:
            if fn >= self.f_contact:
                self._enter(PRESS)
                self.path = LanePath(self.scene, (self.u1, self.u2))
            else:
                self.h -= (self.v_fast if gap > self.slow_zone else self.v_slow) * dt
        if self.phase in (PRESS, SWEEP):
            self.h -= (f_des - fn) * self.k_int * dt
        if self.phase == PRESS:
            self.settle = self.settle + 1 if abs(fn - f_des) < 1.0 else 0
            if (self.settle >= 4 and self.t_phase >= 0.3) or self.t_phase >= 3.0:
                self._enter(SWEEP)
        elif self.phase == SWEEP:
            self.s += self.scene.v * self.path.speed_scale(self.s, self.corner_dist, self.corner_scale) * dt
            self.u1, self.u2 = self.path.at(self.s)
            if self.s >= self.path.length:
                self._enter(LIFT)
        elif self.phase == LIFT:
            self.h += self.v_lift * dt
            if gap > self.lift_clear:
                self._enter(DONE)
        return self.scene.from_plane(self.u1, self.u2, self.h)

    def _enter(self, phase):
        self.phase, self.t_phase, self.settle = phase, 0.0, 0

    @property
    def done(self) -> bool:
        return self.phase == DONE
