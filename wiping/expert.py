"""The scripted wiping program (privileged expert).

It knows the table plane (n, t1, t2), the region and the controller gains.
At every control step (20 Hz) it maps the observation to a commanded target
x_d for the impedance controller:

  0 APPROACH  hover above the first lane start, descend along -n
              (fast, then slow in the last `slow_zone`), until f_n > f_contact.
              The target never goes deeper than the static depth for
              approach_force_frac * f_des, and at contact it is re-anchored to
              x - K^-1 f. Both remove the free-space tracking lag D_n v / K_n,
              which would otherwise turn into a force spike at contact.
  1 PRESS     integral loop on the commanded depth h along n:
                  h <- h - (f_des - f_n) / (K_eff tau) dt,
              K_eff = K_n k_e / (K_n + k_e) (controller and table springs in
              series), tau = force_tau. Ends once |f_n - f_des| stays within
              tol for `press_settle_time`.
  2 WIPE      the same force loop, while the tangential reference moves along
              the boustrophedon lanes at speed v, smoothed by a critically damped
              2nd-order filter (ref_omega) that rounds the lane corners and ramps
              the speed up from rest. The target leads the reference by
              (mu f_n dir(v) + D_t v) / K_t, the steady-state lag of the tool
              behind its target while sliding.
  3 LIFT      raise h at lift_speed to lift_height, hold.
  4 DONE

This is the answer to the proposal's footnote "find a way to achieve this in
Drake": the force loop runs in the scripted program at the policy rate, on the
contact force read from MultibodyPlant's contact-results port, and only moves
the impedance target -- so the logged x_d is exactly the command a policy would
have to output.
"""

from __future__ import annotations

import numpy as np

from .config import EpisodeConfig
from .lanes import PathFollower, plan_lanes

APPROACH, PRESS, WIPE, LIFT, DONE = 0, 1, 2, 3, 4
PHASE_NAMES = ["approach", "press", "wipe", "lift", "done"]


class ScriptedWiper:
    def __init__(self, cfg: EpisodeConfig):
        self.cfg = cfg
        e, tab = cfg.expert, cfg.table
        self.dt = 1.0 / cfg.sim.rate
        self.n = tab.normal()
        self.K, self.D = cfg.controller.world_matrices(tab)
        self.Kinv = np.linalg.inv(self.K)
        Kt, Kn, Dt, _ = cfg.controller.gains_table(tab.k_e)
        self.Kt, self.Kn, self.Dt = Kt, Kn, Dt
        self.K_eff = Kn * tab.k_e / (Kn + tab.k_e)
        self.ki = 1.0 / (self.K_eff * e.force_tau)
        self.waypoints = plan_lanes(cfg.region, e)
        self.path = PathFollower(self.waypoints)
        self.reset()

    # ------------------------------------------------------------------
    def reset(self):
        self.phase = APPROACH
        self.t_phase = 0.0
        self.t_settled = None
        self.t_lifted = None
        self.h =self.cfg.expert.hover  # commanded tcp height above the surface (along n)
        self.uv = self.waypoints[0].copy()  # smoothed reference contact point (table u, v)
        self.uv_dot = np.zeros(2)  # its velocity
        self.uv_ff = np.zeros(2)  # tangential lead added to the target
        self.fn_f = 0.0  # filtered normal force
        self.t = 0.0

    def start_tcp(self) -> np.ndarray:
        """World position of the tcp at t = 0 (the env starts the arm there)."""
        u, v = self.waypoints[0]
        return self.cfg.table.to_world([u, v, self.cfg.expert.hover])

    def expected_duration(self) -> float:
        e = self.cfg.expert
        t_app = (e.hover - e.slow_zone) / e.approach_speed + e.slow_zone / e.approach_speed_slow + 1.0
        t_wipe = self.path.length / e.speed + (8.0 / e.ref_omega if e.ref_omega > 0 else 0.0)
        t_lift = (e.lift_height + 0.05) / e.lift_speed + e.hold_after_lift
        return t_app + e.press_timeout + t_wipe + t_lift

    def _smooth_reference(self, p_path):
        """Critically damped 2nd-order filter from the lane path to the reference uv.

        Rounds the 90 deg lane corners and ramps the speed up from rest, so the
        tangential friction never flips in one step (each flip kicks the normal
        force through the arm's coupled inertia).
        """
        w = self.cfg.expert.ref_omega
        if w <= 0:
            self.uv_dot = (p_path - self.uv) / self.dt
            self.uv = p_path.copy()
            return
        n_sub = 10
        h = self.dt / n_sub
        for _ in range(n_sub):
            acc = w * w * (p_path - self.uv) - 2.0 * w * self.uv_dot
            self.uv_dot = self.uv_dot + h * acc
            self.uv = self.uv + h * self.uv_dot

    def _tangential_lead(self, fn):
        """Target lead along the motion: (mu f_n dir(v) + D_t v) / K_t.

        In steady sliding the tool trails its target by friction / K_t plus
        D_t v / K_t (the controller damps absolute velocity); leading the target
        by that much keeps the tool on the planned lanes. dir(v) is regularized
        below v_eps so the lead fades in/out with the motion.
        """
        e, tab = self.cfg.expert, self.cfg.table
        lead = np.zeros(2)
        sp = float(np.linalg.norm(self.uv_dot))
        if e.friction_ff:
            v_eps = 0.25 * e.speed
            lead += tab.mu * max(fn, 0.0) / self.Kt * self.uv_dot / max(sp, v_eps)
        if e.damping_ff:
            lead += self.Dt * self.uv_dot / self.Kt
        return lead

    def _to(self, phase, t):
        self.phase, self.t_phase = phase, t

    # ------------------------------------------------------------------
    def __call__(self, obs: dict):
        """obs needs t, x (tcp, world), f (force of the table on the tool, world)."""
        e, tab, dt = self.cfg.expert, self.cfg.table, self.dt
        t = float(obs["t"])
        x, f = np.asarray(obs["x"]), np.asarray(obs["f"])
        fn = float(f @ self.n)
        a = e.force_filter
        self.fn_f = a * fn + (1 - a) * self.fn_f

        if self.phase == APPROACH:
            v = e.approach_speed if self.h > e.slow_zone else e.approach_speed_slow
            # never command deeper than the static depth for f0 = approach_force_frac f_des:
            # the free-space lag D_n v / K_n would otherwise turn into a force spike at contact
            h_floor = -e.approach_force_frac * e.f_des * (1.0 / self.Kn + 1.0 / tab.k_e)
            self.h = max(h_floor, self.h - v * dt)
            if fn > e.f_contact:
                # re-anchor: the target that is consistent with the current force
                x_anchor = x - self.Kinv @ f
                self.h = float(tab.to_table(x_anchor)[2])
                self.fn_f = fn
                self._to(PRESS, t)

        elif self.phase in (PRESS, WIPE):
            self.h -= self.ki * (e.f_des - self.fn_f) * dt
            self.h = float(np.clip(self.h, -0.15, e.hover))
            if self.phase == PRESS:
                ok = abs(self.fn_f - e.f_des) <= e.press_settle_tol * e.f_des
                if ok and self.t_settled is None:
                    self.t_settled = t
                if not ok:
                    self.t_settled = None
                long_enough = t - self.t_phase >= e.press_min_time
                settled = self.t_settled is not None and t - self.t_settled >= e.press_settle_time
                if long_enough and (settled or t - self.t_phase >= e.press_timeout):
                    self._to(WIPE, t)
            else:
                p_path, _, done = self.path.advance(e.speed * dt)
                self._smooth_reference(p_path)
                self.uv_ff = self._tangential_lead(self.fn_f)
                settled = (np.linalg.norm(self.uv - p_path) < 1e-3
                           and np.linalg.norm(self.uv_dot) < 0.1 * e.speed)
                if done and settled:
                    self._to(LIFT, t)

        if self.phase in (LIFT, DONE):
            self.uv_dot = np.zeros(2)
            self.uv_ff = np.zeros(2)

        if self.phase == LIFT:
            self.h = min(e.lift_height, self.h + e.lift_speed * dt)
            if self.h >= e.lift_height and fn < e.f_contact:
                if self.t_lifted is None:
                    self.t_lifted = t
                if t - self.t_lifted >= e.hold_after_lift:
                    self._to(DONE, t)

        if fn > e.f_abort and self.phase != LIFT:
            self.h += 0.002  # back off; never seen with the default gains

        self.t = t
        uvw = np.array([self.uv[0] + self.uv_ff[0], self.uv[1] + self.uv_ff[1], self.h])
        return tab.to_world(uvw), self.phase
