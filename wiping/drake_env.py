"""Drake scene: iiwa14 + ball tool, a (tilted) box table with point-contact
stiffness k_e, and a Cartesian impedance controller with known K and D.

Controller (stiffness form of operational-space impedance control, Ch. 8):

    tau = J^T [ K_rot e_R - D_rot w ;  K (x_d - x) - D xdot ]  - tau_g(q) + C(q, v) v
          + (I - J^T J^+T) (K_null (q_nom - q) - D_null v)

with J the 6x7 spatial-velocity Jacobian of the tcp. There is no inertia
shaping, so in static contact K (x_d - x) + f = 0 holds exactly with the K we
chose -- that is what makes labels (b) and (c) comparable. K and D are 3x3
world matrices built from table-frame gains (tangential, normal).

The tcp is the bottom of the ball (ball center + r along the tool axis); the
controller holds the tool axis along -n, so the tcp is the contact point.
Contact uses Drake's point-contact model. Each of the two geometries gets
point_stiffness = 2 k_e, so the series combination seen by the tool is k_e.
"""

from __future__ import annotations

import numpy as np
from pydrake.all import (
    AddContactMaterial,
    AddMultibodyPlantSceneGraph,
    Box,
    ContactModel,
    CoulombFriction,
    Cylinder,
    DiagramBuilder,
    DiscreteContactApproximation,
    FixedOffsetFrame,
    InverseKinematics,
    JacobianWrtVariable,
    LeafSystem,
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    Parser,
    ProximityProperties,
    RigidTransform,
    RotationMatrix,
    Simulator,
    Solve,
    SpatialInertia,
    Sphere,
    UnitInertia,
)

from .config import ControllerConfig, EpisodeConfig

IIWA_URL = "package://drake_models/iiwa_description/sdf/iiwa14_no_collision.sdf"
L7_FLANGE_Z = 0.045  # link-7 frame to the flange (approx., visual only)
L7_BALL_Z = 0.12  # ball center in the link-7 frame


class CartesianImpedanceController(LeafSystem):
    def __init__(self, plant, iiwa, tcp, K, D, R_d, q_nom, cc: ControllerConfig):
        super().__init__()
        self.plant, self.iiwa, self.tcp = plant, iiwa, tcp
        self.ctx = plant.CreateDefaultContext()
        self.K, self.D = np.asarray(K, float), np.asarray(D, float)
        self.R_d = R_d
        self.q_nom = np.asarray(q_nom, float)
        self.cc = cc
        self.state = self.DeclareVectorInputPort("iiwa_state", 14)
        self.target = self.DeclareVectorInputPort("x_d", 3)
        self.DeclareVectorOutputPort("tau", 7, self.calc)

    def calc(self, context, out):
        s = self.state.Eval(context)
        q, v = s[:7], s[7:]
        x_d = self.target.Eval(context)
        plant, ctx, cc = self.plant, self.ctx, self.cc
        plant.SetPositions(ctx, self.iiwa, q)
        plant.SetVelocities(ctx, self.iiwa, v)
        X = self.tcp.CalcPoseInWorld(ctx)
        J = plant.CalcJacobianSpatialVelocity(
            ctx, JacobianWrtVariable.kV, self.tcp, np.zeros(3), plant.world_frame(), plant.world_frame()
        )
        V = J @ v
        F = self.K @ (x_d - X.translation()) - self.D @ V[3:]
        aa = (self.R_d @ X.rotation().inverse()).ToAngleAxis()
        M = cc.K_rot * aa.angle() * aa.axis() - cc.D_rot * V[:3]
        tau = J.T @ np.concatenate([M, F])
        if cc.gravity_comp:
            tau -= plant.CalcGravityGeneralizedForces(ctx)
        if cc.coriolis_comp:
            tau += plant.CalcBiasTerm(ctx)
        N = np.eye(7) - J.T @ np.linalg.pinv(J).T
        tau += N @ (cc.K_null * (self.q_nom - q) - cc.D_null * v)
        out.SetFromVector(tau)


def _props(k_e, mu, d):
    p = ProximityProperties()
    AddContactMaterial(dissipation=d, point_stiffness=2.0 * k_e, friction=CoulombFriction(mu, mu), properties=p)
    return p


def solve_ik(plant, tcp, p_tcp, tool_dir):
    """Joint angles with the tcp at p_tcp and the tool axis (link-7 z) along tool_dir."""
    ik = InverseKinematics(plant)
    ik.AddPositionConstraint(tcp, np.zeros(3), plant.world_frame(), p_tcp - 1e-4, p_tcp + 1e-4)
    ik.AddAngleBetweenVectorsConstraint(tcp, [0, 0, 1], plant.world_frame(), np.asarray(tool_dir, float), 0.0, 1e-3)
    q_nom = np.array([np.arctan2(p_tcp[1], p_tcp[0]), 0.6, 0.0, -1.7, 0.0, 0.85, 0.0])
    ik.get_mutable_prog().AddQuadraticErrorCost(np.eye(7) * 0.1, q_nom, ik.q())
    rng = np.random.default_rng(0)  # fixed restarts: a config always maps to the same q0
    for i in range(30):
        guess = q_nom if i == 0 else q_nom + rng.normal(0, 0.3, 7)
        ik.get_mutable_prog().SetInitialGuess(ik.q(), guess)
        res = Solve(ik.prog())
        if res.is_success():
            return res.GetSolution(ik.q())
    raise RuntimeError(f"IK failed for tcp {p_tcp} (is the region reachable?)")


class DrakeWipeEnv:
    """One episode of physics. `step(x_d)` holds x_d for 1/rate s (zero-order hold)."""

    def __init__(self, cfg: EpisodeConfig, start_tcp: np.ndarray, meshcat=None):
        self.cfg = cfg
        tab, sim = cfg.table, cfg.sim
        r = sim.tool_radius
        b = DiagramBuilder()
        plant, scene_graph = AddMultibodyPlantSceneGraph(b, time_step=sim.dt)
        plant.set_contact_model(ContactModel.kPoint)
        plant.set_discrete_contact_approximation(DiscreteContactApproximation.kLagged)
        iiwa = Parser(plant).AddModelsFromUrl(IIWA_URL)[0]
        plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("iiwa_link_0", iiwa))
        l7 = plant.GetBodyByName("iiwa_link_7", iiwa)

        # ball tool (+ a visual-only stick to the flange)
        tool = plant.AddRigidBody("tool", iiwa, SpatialInertia(sim.tool_mass, np.zeros(3), UnitInertia.SolidSphere(r)))
        plant.WeldFrames(l7.body_frame(), tool.body_frame(), RigidTransform([0.0, 0.0, L7_BALL_Z]))
        props = _props(tab.k_e, tab.mu, tab.dissipation)
        plant.RegisterCollisionGeometry(tool, RigidTransform(), Sphere(r), "tool", props)
        plant.RegisterVisualGeometry(tool, RigidTransform(), Sphere(r), "tool", [0.95, 0.55, 0.1, 1.0])
        stick = L7_BALL_Z - L7_FLANGE_Z - 0.5 * r
        plant.RegisterVisualGeometry(tool, RigidTransform([0, 0, -0.5 * r - 0.5 * stick]), Cylinder(0.008, stick),
                                     "stick", [0.3, 0.3, 0.3, 1.0])

        # table
        X_table = RigidTransform(RotationMatrix(tab.frame()), tab.box_center())
        box = Box(*tab.size)
        plant.RegisterCollisionGeometry(plant.world_body(), X_table, box, "table", props)
        plant.RegisterVisualGeometry(plant.world_body(), X_table, box, "table", [0.72, 0.72, 0.76, 1.0])

        tcp = plant.AddFrame(FixedOffsetFrame("tcp", l7.body_frame(), RigidTransform([0.0, 0.0, L7_BALL_Z + r])))
        plant.Finalize()

        n = tab.normal()
        q0 = solve_ik(plant, tcp, np.asarray(start_tcp, float), -n)
        ctx0 = plant.CreateDefaultContext()
        plant.SetPositions(ctx0, iiwa, q0)
        R_d = tcp.CalcPoseInWorld(ctx0).rotation()

        self.K, self.D = cfg.controller.world_matrices(tab)
        ctrl = b.AddSystem(CartesianImpedanceController(plant, iiwa, tcp, self.K, self.D, R_d, q0, cfg.controller))
        b.Connect(plant.get_state_output_port(iiwa), ctrl.state)
        b.Connect(ctrl.get_output_port(), plant.get_actuation_input_port(iiwa))
        b.ExportInput(ctrl.target, "x_d")

        self.meshcat, self.visualizer = meshcat, None
        if meshcat is not None:
            self.visualizer = MeshcatVisualizer.AddToBuilder(b, scene_graph, meshcat, MeshcatVisualizerParams())

        self.diagram = b.Build()
        self.plant, self.iiwa, self.tcp, self.tool = plant, iiwa, tcp, tool
        self.q0 = q0
        self.sim = Simulator(self.diagram)
        self.root = self.sim.get_mutable_context()
        self.pctx = plant.GetMyMutableContextFromRoot(self.root)
        plant.SetPositions(self.pctx, iiwa, q0)
        self.port = self.diagram.get_input_port(0)
        self.port.FixValue(self.root, np.asarray(start_tcp, float))
        self.sim.Initialize()

    # ------------------------------------------------------------------
    def contact_force_on_tool(self) -> np.ndarray:
        """Sum of point-contact forces the table applies to the tool (world)."""
        cr = self.plant.get_contact_results_output_port().Eval(self.pctx)
        f = np.zeros(3)
        for i in range(cr.num_point_pair_contacts()):
            info = cr.point_pair_contact_info(i)
            if info.bodyB_index() == self.tool.index():
                f += info.contact_force()
            elif info.bodyA_index() == self.tool.index():
                f -= info.contact_force()
        return f

    def observe(self) -> dict:
        q = self.plant.GetPositions(self.pctx, self.iiwa)
        v = self.plant.GetVelocities(self.pctx, self.iiwa)
        x = self.tcp.CalcPoseInWorld(self.pctx).translation()
        xdot = self.tcp.CalcSpatialVelocityInWorld(self.pctx).translational()
        return dict(t=self.root.get_time(), x=x, xdot=xdot, f=self.contact_force_on_tool(), q=q, v=v)

    def step(self, x_d):
        self.port.FixValue(self.root, np.asarray(x_d, float))
        self.sim.AdvanceTo(self.root.get_time() + 1.0 / self.cfg.sim.rate)

    # ---- recording for meshcat replay -----------------------------------
    def start_recording(self):
        if self.visualizer is not None:
            self.visualizer.StartRecording()

    def stop_recording(self):
        if self.visualizer is not None:
            self.visualizer.StopRecording()
            self.visualizer.PublishRecording()

    def close(self):
        pass
