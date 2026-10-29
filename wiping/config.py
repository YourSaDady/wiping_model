"""Configuration for one wiping episode: table, region, controller, expert, sim.

Everything the scripted program and the simulator need is in `EpisodeConfig`,
a plain dataclass tree that round-trips through JSON (`to_dict` / `from_dict`),
so one dict per episode is enough to reproduce it.

Conventions (used everywhere in the package)
-------------------------------------------
* World frame: iiwa base at the origin, z up.
* Table frame: origin `c` (a point on the top surface), unit normal `n`
  (pointing out of the table), tangents `t1`, `t2` with t1 x t2 = n.
  t1 is world x projected onto the table plane, so for a flat table
  (t1, t2, n) = (x, y, z).
* f is the contact force the TABLE applies to the TOOL, world frame,
  so f_n = n . f >= 0 while pressing.
* Stiffness / damping are specified in the table frame (tangential, normal)
  and rotated into world 3x3 matrices: K = R diag(Kt, Kt, Kn) R^T.
"""

from __future__ import annotations

import dataclasses as dc
import json
import math
from typing import Optional

import numpy as np


# --------------------------------------------------------------------------
# Table
# --------------------------------------------------------------------------
@dc.dataclass
class TableConfig:
    height: float = 0.10  # height of the table-frame origin c (m)
    center_xy: tuple = (0.55, 0.0)  # xy of c (m)
    tilt: float = 0.0  # tilt of the top surface (rad)
    tilt_dir: float = 0.0  # azimuth of the normal's horizontal part (rad)
    k_e: float = 2.0e4  # contact (point) stiffness of the table/tool pair (N/m)
    dissipation: float = 20.0  # Hunt-Crossley dissipation (s/m)
    mu: float = 0.5  # Coulomb friction (static = dynamic)
    size: tuple = (0.6, 0.7, 0.05)  # box size along t1, t2, n (m)

    # ---- frame helpers -----------------------------------------------------
    def normal(self) -> np.ndarray:
        st = math.sin(self.tilt)
        return np.array([st * math.cos(self.tilt_dir), st * math.sin(self.tilt_dir), math.cos(self.tilt)])

    def frame(self) -> np.ndarray:
        """3x3 rotation R = [t1 t2 n] (columns), table -> world."""
        n = self.normal()
        ex = np.array([1.0, 0.0, 0.0])
        t1 = ex - (ex @ n) * n
        t1 /= np.linalg.norm(t1)
        t2 = np.cross(n, t1)
        return np.column_stack([t1, t2, n])

    def origin(self) -> np.ndarray:
        return np.array([self.center_xy[0], self.center_xy[1], self.height])

    def to_world(self, uvw) -> np.ndarray:
        """Table-frame coordinates (u, v, w) -> world point. Works on (...,3)."""
        uvw = np.asarray(uvw, float)
        return self.origin() + uvw @ self.frame().T

    def to_table(self, p) -> np.ndarray:
        """World point(s) -> table-frame coordinates (u, v, w); w = height above surface."""
        p = np.asarray(p, float)
        return (p - self.origin()) @ self.frame()

    def box_center(self) -> np.ndarray:
        return self.origin() - 0.5 * self.size[2] * self.normal()


# --------------------------------------------------------------------------
# Wiping region (an L x L square in the table plane)
# --------------------------------------------------------------------------
@dc.dataclass
class RegionConfig:
    center_uv: tuple = (0.0, 0.0)  # region center in table coordinates (m)
    L: float = 0.12  # side length (m)
    cell: float = 0.005  # dirt-grid resolution (m)

    def corner_uv(self) -> np.ndarray:
        return np.asarray(self.center_uv, float) - 0.5 * self.L

    def p0(self, table: TableConfig) -> np.ndarray:
        """World position of the region corner (min u, min v) on the surface."""
        c = self.corner_uv()
        return table.to_world([c[0], c[1], 0.0])


# --------------------------------------------------------------------------
# Controller (Cartesian impedance, stiffness form)
# --------------------------------------------------------------------------
@dc.dataclass
class ControllerConfig:
    K_t: float = 1000.0  # tangential stiffness (N/m)
    K_n: float = 500.0  # normal stiffness (N/m) -- the K of the proposal
    zeta_t: float = 1.0  # tangential damping ratio (free-space)
    zeta_n: float = 0.7  # normal damping ratio, sized for CONTACT: K_n + k_e
    # Nominal tcp mass used to size D (kg). The iiwa's operational-space inertia at the
    # tcp, with the orientation held, is ~15-30 kg along n and ~20-90 kg tangentially
    # over the workspace (Lambda = (J M^-1 J^T)^-1); with the old 3 kg the "zeta" were
    # ~0.2 in reality and the tool rang on every lane corner.
    m_eff: float = 25.0
    D_t: Optional[float] = None  # explicit override (N s/m)
    D_n: Optional[float] = None  # explicit override (N s/m)
    D_n_max: float = 2000.0  # cap on D_n (reached only for k_e >~ 8e4)
    K_rot: float = 100.0  # orientation stiffness (N m/rad)
    D_rot: float = 2.8  # orientation damping (N m s/rad)
    K_null: float = 5.0  # null-space posture stiffness (N m/rad)
    D_null: float = 1.0  # null-space posture damping
    gravity_comp: bool = True
    coriolis_comp: bool = True

    def gains_table(self, k_e: float):
        """(Kt, Kn, Dt, Dn) in the table frame."""
        Dt = self.D_t if self.D_t is not None else 2.0 * self.zeta_t * math.sqrt(self.K_t * self.m_eff)
        if self.D_n is not None:
            Dn = self.D_n
        else:
            # Damping sized for the contact mode (controller spring + table spring in
            # parallel acting on m_eff). Choosing it for free space (K_n only) makes the
            # tool bounce on stiff tables -- see the proposal.
            Dn = min(self.D_n_max, 2.0 * self.zeta_n * math.sqrt((self.K_n + k_e) * self.m_eff))
        return self.K_t, self.K_n, Dt, Dn

    def world_matrices(self, table: TableConfig):
        """3x3 world-frame K and D."""
        Kt, Kn, Dt, Dn = self.gains_table(table.k_e)
        R = table.frame()
        K = R @ np.diag([Kt, Kt, Kn]) @ R.T
        D = R @ np.diag([Dt, Dt, Dn]) @ R.T
        return K, D


# --------------------------------------------------------------------------
# Scripted wiping program
# --------------------------------------------------------------------------
@dc.dataclass
class ExpertConfig:
    f_des: float = 10.0  # desired normal force (N)
    speed: float = 0.06  # wiping speed along the lanes (m/s)
    patch_r: float = 0.015  # wiping-patch radius r (m): dirt within r is cleaned
    lane_spacing: float = 0.02  # s < 2 r
    edge_margin: Optional[float] = None  # default min(s/2, 0.6 r)
    hover: float = 0.03  # start/finish height of the tcp above the surface (m)
    approach_speed: float = 0.02  # m/s, far from the table
    approach_speed_slow: float = 0.01  # m/s, last `slow_zone` metres
    slow_zone: float = 0.01
    approach_force_frac: float = 0.5  # approach target stops at the static depth for this * f_des
    f_contact: float = 1.0  # contact detection threshold (N)
    force_tau: float = 0.25  # time constant of the normal-force integral loop (s)
    force_filter: float = 0.5  # EMA weight on the newest normal-force sample
    press_settle_tol: float = 0.1  # |f_n - f_des| <= tol * f_des ...
    press_settle_time: float = 0.3  # ... for this long ends the press phase (s)
    press_min_time: float = 0.5
    press_timeout: float = 3.0
    friction_ff: bool = True  # lead the tangential target by mu f_n / K_t
    damping_ff: bool = True  # ... plus D_t v / K_t (the controller damps absolute velocity)
    ref_omega: float = 8.0  # bandwidth of the 2nd-order lane-reference filter (rad/s); 0 = raw path
    lift_speed: float = 0.05
    lift_height: float = 0.04  # above the surface (m)
    hold_after_lift: float = 0.3
    f_abort: float = 40.0  # back off if the force ever exceeds this (N)

    def margin(self) -> float:
        if self.edge_margin is not None:
            return self.edge_margin
        return min(0.5 * self.lane_spacing, 0.6 * self.patch_r)


# --------------------------------------------------------------------------
# Simulation / logging
# --------------------------------------------------------------------------
@dc.dataclass
class SimConfig:
    dt: float = 1e-3  # plant time step (s)
    rate: float = 20.0  # expert / policy / logging rate (Hz)
    T_max: Optional[float] = None  # None -> estimated from the plan (+ margin)
    tool_radius: float = 0.02  # ball tool radius (m)
    tool_mass: float = 0.2
    backend: str = "drake"  # "drake" or "pointmass" (fast test double)


@dc.dataclass
class SuccessConfig:
    eps_u: float = 0.1  # uncovered fraction at the end must be <= eps_u
    f_min_frac: float = 0.5  # f_min = 0.5 f_des: wiping counts only above this
    f_max_frac: float = 2.0  # max_t f_n <= f_max = 2 f_des


@dc.dataclass
class EpisodeConfig:
    table: TableConfig = dc.field(default_factory=TableConfig)
    region: RegionConfig = dc.field(default_factory=RegionConfig)
    controller: ControllerConfig = dc.field(default_factory=ControllerConfig)
    expert: ExpertConfig = dc.field(default_factory=ExpertConfig)
    sim: SimConfig = dc.field(default_factory=SimConfig)
    success: SuccessConfig = dc.field(default_factory=SuccessConfig)
    seed: int = 0
    name: str = "default"

    # ---- derived ----------------------------------------------------------
    def f_min(self) -> float:
        return self.success.f_min_frac * self.expert.f_des

    def f_max(self) -> float:
        return self.success.f_max_frac * self.expert.f_des

    # ---- (de)serialization -----------------------------------------------
    def to_dict(self) -> dict:
        return dc.asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @staticmethod
    def from_dict(d: dict) -> "EpisodeConfig":
        sub = dict(table=TableConfig, region=RegionConfig, controller=ControllerConfig,
                   expert=ExpertConfig, sim=SimConfig, success=SuccessConfig)
        kw = {}
        for k, v in d.items():
            if k in sub:
                cls = sub[k]
                fields = {f.name: f for f in dc.fields(cls)}
                vv = {kk: (tuple(x) if isinstance(x, list) else x) for kk, x in v.items() if kk in fields}
                kw[k] = cls(**vv)
            else:
                kw[k] = v
        return EpisodeConfig(**kw)

    def replace(self, **over) -> "EpisodeConfig":
        """Nested override, e.g. cfg.replace(table__tilt=0.1, expert__speed=0.05)."""
        d = self.to_dict()
        for key, val in over.items():
            parts = key.split("__")
            tgt = d
            for p in parts[:-1]:
                tgt = tgt[p]
            if parts[-1] not in tgt:
                raise KeyError(key)
            tgt[parts[-1]] = val
        return EpisodeConfig.from_dict(d)
