"""Torque-level controllers for the iiwa (run at the 1 kHz plant rate).

ImpedanceController: Cartesian stiffness control in operational-space form
(textbook Ch. 8), written by us because Drake has no Cartesian impedance
controller for the iiwa:

    tau = J^T [K_rot e_R - D_rot w ; K (x_d - x) - D xdot] - tau_g + N tau_null

J is the spatial-velocity Jacobian of the ball centre, tau_g is Drake's
gravity generalized force (so -tau_g compensates gravity), and N projects a
joint-centering term into the null space of J.

HybridForcePositionController: classical hybrid force/position control. Along
the table normal it regulates the measured contact force with feedforward + PI
(after a velocity-servoed approach that does not need the table height);
in the table plane it tracks the lane path with the same stiffness as the
impedance controller.
"""
from __future__ import annotations

import numpy as np
from pydrake.all import (AbstractValue, BasicVector, ContactResults, JacobianWrtVariable, LeafSystem)

from .config import CORNER_DIST, CORNER_SCALE, DT_SIM, Gains, TaskParams
from .paths import LanePath


def _vee_orientation_error(R_d: np.ndarray, R: np.ndarray) -> np.ndarray:
    """sin(theta) * axis of R_d R^T, expressed in world. Exact direction, and
    magnitude ~ angle for the small errors we see."""
    E = R_d @ R.T
    return 0.5 * np.array([E[2, 1] - E[1, 2], E[0, 2] - E[2, 0], E[1, 0] - E[0, 1]])


class _CartesianBase(LeafSystem):
    def __init__(self, plant, tool_body, gains: Gains, n: np.ndarray, R_target: np.ndarray,
                 q_null: np.ndarray):
        super().__init__()
        self.plant = plant
        self.ctx = plant.CreateDefaultContext()     # controller's own plant context
        self.tool_frame = tool_body.body_frame()
        self.world = plant.world_frame()
        self.gains = gains
        self.n = np.asarray(n, float)
        self.K = gains.K(self.n)
        self.D = gains.D(self.n)
        self.R_target = np.asarray(R_target, float)
        self.q_null = np.asarray(q_null, float)
        nq = plant.num_positions()
        self.nq = nq
        self.state_in = self.DeclareVectorInputPort("state", 2 * nq)
        self._zero3 = np.zeros(3)

    def _kinematics(self, context):
        xs = self.state_in.Eval(context)
        q, v = xs[:self.nq], xs[self.nq:]
        self.plant.SetPositionsAndVelocities(self.ctx, xs)
        X = self.plant.CalcRelativeTransform(self.ctx, self.world, self.tool_frame)
        J = self.plant.CalcJacobianSpatialVelocity(self.ctx, JacobianWrtVariable.kV, self.tool_frame,
                                                   self._zero3, self.world, self.world)
        V = J @ v
        return q, v, X.translation(), X.rotation().matrix(), J, V

    def _torque(self, q, v, R, J, V, F_trans):
        g = self.gains
        e_R = _vee_orientation_error(self.R_target, R)
        F = np.concatenate([g.K_rot * e_R - g.D_rot * V[:3], F_trans])
        tau_g = self.plant.CalcGravityGeneralizedForces(self.ctx)
        JJt = J @ J.T
        N = np.eye(self.nq) - J.T @ np.linalg.solve(JJt, J)
        tau_null = N @ (g.K_null * (self.q_null - q) - g.D_null * v)
        return J.T @ F - tau_g + tau_null


class ImpedanceController(_CartesianBase):
    """Inputs: plant state, x_d (3). Output: joint torques."""

    def __init__(self, plant, tool_body, gains, n, R_target, q_null):
        super().__init__(plant, tool_body, gains, n, R_target, q_null)
        self.xd_in = self.DeclareVectorInputPort("x_d", 3)
        self.DeclareVectorOutputPort("tau", self.nq, self._calc)

    def _calc(self, context, output):
        q, v, p, R, J, V = self._kinematics(context)
        x_d = self.xd_in.Eval(context)
        F_trans = self.K @ (x_d - p) - self.D @ V[3:]
        output.SetFromVector(self._torque(q, v, R, J, V, F_trans))


class ToolForceSensor(LeafSystem):
    """Sum of contact forces the environment applies to the tool body, in world."""

    def __init__(self, plant, tool_body):
        super().__init__()
        self.tool_index = tool_body.index()
        self.cr_in = self.DeclareAbstractInputPort("contact_results", AbstractValue.Make(ContactResults()))
        self.DeclareVectorOutputPort("force", 3, self._calc)

    def _calc(self, context, output):
        cr = self.cr_in.Eval(context)
        f = np.zeros(3)
        for i in range(cr.num_point_pair_contacts()):
            info = cr.point_pair_contact_info(i)
            fi = info.contact_force()          # force on body B, expressed in world
            if info.bodyB_index() == self.tool_index:
                f += fi
            elif info.bodyA_index() == self.tool_index:
                f -= fi
        output.SetFromVector(f)


class PeakForceMonitor(LeafSystem):
    """Tracks max over 1 kHz samples of the normal force n.f (discrete state)."""

    def __init__(self, n):
        super().__init__()
        self.n = np.asarray(n, float)
        self.f_in = self.DeclareVectorInputPort("force", 3)
        self.DeclareDiscreteState(1)
        self.DeclarePeriodicDiscreteUpdateEvent(DT_SIM, 0.0, self._update)

    def _update(self, context, discrete_state):
        fn = float(self.n @ self.f_in.Eval(context))
        old = context.get_discrete_state_vector().GetAtIndex(0)
        discrete_state.get_mutable_vector().SetAtIndex(0, max(old, fn))


class HybridForcePositionController(_CartesianBase):
    """Classical hybrid force/position baseline (1 kHz, no demonstrations).

    Knows the region (p0, L, t1, t2), the normal n and the lane speed, but not
    the table height: it approaches along -n with a velocity servo until the
    measured normal force exceeds f_contact, then regulates the force with
    feedforward + PI while tracking the lanes in the plane, then lifts.

    Discrete state: [mode, I_f, s_path, t_mode, u1_ref, u2_ref, settle_count, f_ref]
    mode 0 approach, 1 press, 2 sweep, 3 lift, 4 done.
    """
    APPROACH, PRESS, SWEEP, LIFT, DONE = range(5)

    def __init__(self, plant, tool_body, gains, scene, task: TaskParams, q_null, R_target,
                 v_approach=0.008, f_contact=1.0, k_p=0.3, k_i=6.0, ramp_time=0.5,
                 v_lift=0.03, lift_time=1.5):
        super().__init__(plant, tool_body, gains, scene.arr("n"), R_target, q_null)
        self.scene = scene
        self.task = task
        self.t1, self.t2 = scene.arr("t1"), scene.arr("t2")
        self.P_t = np.eye(3) - np.outer(self.n, self.n)
        self.v_app, self.f_contact, self.k_p, self.k_i = v_approach, f_contact, k_p, k_i
        self.ramp_time, self.v_lift, self.lift_time = ramp_time, v_lift, lift_time
        self.f_in = self.DeclareVectorInputPort("force", 3)
        u_start = scene.to_plane(scene.arr("start"))
        self.path = None   # built at contact (starts from the contact location)
        self._u_start = u_start
        x0 = np.array([self.APPROACH, 0.0, 0.0, 0.0, u_start[0], u_start[1], 0.0, 0.0])
        self.DeclareDiscreteState(x0)
        self.DeclarePeriodicDiscreteUpdateEvent(DT_SIM, 0.0, self._update)
        self.DeclareVectorOutputPort("tau", self.nq, self._calc)
        self.DeclareVectorOutputPort("mode", 1, lambda c, o: o.SetAtIndex(
            0, c.get_discrete_state_vector().GetAtIndex(0)))

    def _plane_target(self, s):
        u1, u2 = self.path.at(s)
        return u1, u2

    def _update(self, context, discrete_state):
        st = context.get_discrete_state_vector().CopyToVector()
        mode, I_f, s, t_mode, u1r, u2r, settle, f_ref = st
        mode = int(mode)
        xs = self.state_in.Eval(context)
        self.plant.SetPositionsAndVelocities(self.ctx, xs)
        fn = float(self.n @ self.f_in.Eval(context))
        dt = DT_SIM
        t_mode += dt
        f_des = self.task.f_des
        if mode == self.APPROACH:
            if fn >= self.f_contact:
                p = self.plant.CalcRelativeTransform(self.ctx, self.world, self.tool_frame).translation()
                u = self.scene.to_plane(p)
                u1r, u2r = u[0], u[1]
                self.path = LanePath(self.scene, (u1r, u2r))
                mode, t_mode, f_ref, I_f = self.PRESS, 0.0, fn, 0.0
        elif mode in (self.PRESS, self.SWEEP):
            # reference ramps from the force at contact to f_des
            f_ref = min(f_des, f_ref + (f_des - self.f_contact) * dt / self.ramp_time)
            e = f_ref - fn
            I_f = float(np.clip(I_f + e * dt, -10.0 / self.k_i, 10.0 / self.k_i))
            if mode == self.PRESS:
                settle = settle + 1 if abs(fn - f_des) < 1.0 and f_ref >= f_des else 0
                if (settle >= 200 and t_mode >= 0.3) or t_mode >= 3.0:
                    mode, t_mode, s = self.SWEEP, 0.0, 0.0
            else:
                s += self.scene.v * self.path.speed_scale(s, CORNER_DIST, CORNER_SCALE) * dt
                if s >= self.path.length:
                    s = self.path.length
                    mode, t_mode = self.LIFT, 0.0
        elif mode == self.LIFT:
            if t_mode >= self.lift_time:
                mode = self.DONE
        discrete_state.get_mutable_vector().SetFromVector(
            np.array([mode, I_f, s, t_mode, u1r, u2r, settle, f_ref]))

    def _calc(self, context, output):
        q, v, p, R, J, V = self._kinematics(context)
        mode, I_f, s, t_mode, u1r, u2r, settle, f_ref = context.get_discrete_state_vector().CopyToVector()
        mode = int(mode)
        g = self.gains
        vel = V[3:]
        vn = float(self.n @ vel)
        if mode == self.SWEEP:
            u1r, u2r = self._plane_target(s)
        elif mode in (self.LIFT, self.DONE) and self.path is not None:
            u1r, u2r = self._plane_target(self.path.length)
        target = self.scene.from_plane(u1r, u2r, 0.0)
        F_plane = self.P_t @ (g.K_t * (target - p) - g.D_t * vel)
        if mode == self.APPROACH:
            F_n = g.D_n * (-self.v_app - vn)
        elif mode in (self.PRESS, self.SWEEP):
            fn = float(self.n @ self.f_in.Eval(context))
            F_n = -(f_ref + self.k_p * (f_ref - fn) + self.k_i * I_f) - g.D_n * vn
        else:
            F_n = g.D_n * ((self.v_lift if mode == self.LIFT else 0.0) - vn)
        F_trans = F_plane + F_n * self.n
        output.SetFromVector(self._torque(q, v, R, J, V, F_trans))
