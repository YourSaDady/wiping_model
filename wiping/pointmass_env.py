"""Point-mass test double with the same interface as DrakeWipeEnv.

NOT the experiment's simulator. It models the tool as a gravity-compensated
point mass m_eff under the same impedance law F = K (x_d - x) - D v, against a
Hunt-Crossley plane with regularized Coulomb friction. It runs ~50x faster than
Drake and needs only numpy, so it is used to unit-test the scripted program,
coverage, logging, labels and the dataset writer, and to iterate on expert
logic before paying for full rigid-body sims.
"""

from __future__ import annotations

import numpy as np

from .config import EpisodeConfig


class PointMassWipeEnv:
    def __init__(self, cfg: EpisodeConfig, start_tcp: np.ndarray, meshcat=None, substep: float = 2e-4,
                 v_slip: float = 2e-3):
        self.cfg = cfg
        self.K, self.D = cfg.controller.world_matrices(cfg.table)
        self.m = cfg.controller.m_eff
        self.n = cfg.table.normal()
        self.c = cfg.table.origin()
        self.h = substep
        self.v_slip = v_slip
        self.x = np.asarray(start_tcp, float).copy()
        self.v = np.zeros(3)
        self.t = 0.0
        self.x_d = self.x.copy()
        self._f = np.zeros(3)
        self.meshcat = None

    # ------------------------------------------------------------------
    def _contact(self, x, v):
        tab = self.cfg.table
        w = float((x - self.c) @ self.n)  # tcp height above the plane
        if w >= 0.0:
            return np.zeros(3)
        d, ddot = -w, -float(v @ self.n)
        fn = max(0.0, tab.k_e * d * (1.0 + tab.dissipation * ddot))
        vt = v - (v @ self.n) * self.n
        sp = np.linalg.norm(vt)
        ft = -tab.mu * fn * vt / max(sp, self.v_slip)
        return fn * self.n + ft

    def observe(self):
        return dict(t=self.t, x=self.x.copy(), xdot=self.v.copy(), f=self._f.copy(), q=np.zeros(7), v=np.zeros(7))

    def step(self, x_d):
        self.x_d = np.asarray(x_d, float)
        n_sub = int(round(1.0 / self.cfg.sim.rate / self.h))
        for _ in range(n_sub):
            fc = self._contact(self.x, self.v)
            F = self.K @ (self.x_d - self.x) - self.D @ self.v + fc
            self.v = self.v + self.h * F / self.m
            self.x = self.x + self.h * self.v
            self.t += self.h
        self._f = self._contact(self.x, self.v)

    def close(self):
        pass
