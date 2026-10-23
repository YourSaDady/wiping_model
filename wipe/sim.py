"""Drake wiping environment: iiwa14 + ball tool + tilted compliant table.

WipeEnv runs at the 20 Hz policy rate with zero-order hold on the commanded
target x_d. Each policy step advances the 1 kHz plant in SUBSTEPS chunks and
samples the tool for the coverage metric at 100 Hz; the peak normal force is
tracked at 1 kHz by PeakForceMonitor.

Contact: point contact with the lagged discrete approximation. Drake combines
the point stiffnesses of the two geometries in series, so both get 2 k_e and
the pair has stiffness k_e. Hunt-Crossley dissipation is the same on both.
"""
from __future__ import annotations

import numpy as np
from pydrake.all import (AddMultibodyPlant, Box, CoulombFriction, Cylinder, DiagramBuilder,
                         InverseKinematics, MultibodyPlantConfig, Parser, ProximityProperties,
                         RigidTransform, RollPitchYaw, RotationMatrix, Simulator, Solve, SpatialInertia,
                         Sphere, UnitInertia, AddContactMaterial, IllustrationProperties)

from .config import (BALL_RADIUS, DT_POLICY, DT_SIM, FRICTION, HC_DISSIPATION, SUBSTEPS, TASK, TOOL_MASS,
                     TOOL_OFFSET, Gains, MAIN_GAINS, TaskParams)
from .controllers import (HybridForcePositionController, ImpedanceController, PeakForceMonitor,
                          ToolForceSensor)
from .coverage import Coverage
from .scene import Scene

IIWA_URL = "package://drake_models/iiwa_description/sdf/iiwa14_no_collision.sdf"
Q_NOMINAL = np.array([0.0, 0.6, 0.0, -1.6, 0.0, 0.95, 0.0])
TABLE_SIZE = (0.5, 0.5, 0.10)


def add_robot_and_tool(plant, k_e: float | None):
    model = Parser(plant).AddModelsFromUrl(IIWA_URL)[0]
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("iiwa_link_0", model))
    # Tool body frame origin = ball centre (the controlled point). The rod's COM
    # sits halfway between flange and ball.
    rod_len = TOOL_OFFSET - BALL_RADIUS - 0.045
    com = np.array([0.0, 0.0, -0.5 * (TOOL_OFFSET - 0.045)])
    I = UnitInertia.SolidCylinder(0.012, TOOL_OFFSET - 0.045, [0, 0, 1]).ShiftFromCenterOfMass(-com)
    tool = plant.AddRigidBody("tool", model, SpatialInertia(TOOL_MASS, com, I))
    plant.WeldFrames(plant.GetFrameByName("iiwa_link_7", model), tool.body_frame(),
                     RigidTransform([0.0, 0.0, TOOL_OFFSET]))
    if plant.geometry_source_is_registered():
        plant.RegisterVisualGeometry(tool, RigidTransform(), Sphere(BALL_RADIUS), "ball_vis",
                                     np.array([0.9, 0.45, 0.35, 1.0]))
        plant.RegisterVisualGeometry(tool, RigidTransform([0, 0, -BALL_RADIUS - 0.5 * rod_len]),
                                     Cylinder(0.008, rod_len), "rod_vis", np.array([0.6, 0.6, 0.6, 1.0]))
        if k_e is not None:
            props = ProximityProperties()
            AddContactMaterial(dissipation=HC_DISSIPATION, point_stiffness=2.0 * k_e,
                               friction=CoulombFriction(FRICTION, FRICTION), properties=props)
            plant.RegisterCollisionGeometry(tool, RigidTransform(), Sphere(BALL_RADIUS), "ball", props)
    return model, tool


def table_pose(scene: Scene) -> RigidTransform:
    R = RotationMatrix.MakeXRotation(np.deg2rad(scene.tilt_deg))
    c = scene.arr("c")
    return RigidTransform(R, c - 0.5 * TABLE_SIZE[2] * scene.arr("n"))


def add_table(plant, scene: Scene):
    props = ProximityProperties()
    AddContactMaterial(dissipation=HC_DISSIPATION, point_stiffness=2.0 * scene.k_e,
                       friction=CoulombFriction(FRICTION, FRICTION), properties=props)
    X = table_pose(scene)
    plant.RegisterCollisionGeometry(plant.world_body(), X, Box(*TABLE_SIZE), "table", props)
    # colour encodes stiffness: soft = light green, hard = dark grey
    s = np.clip((np.log10(scene.k_e) - 2.3) / 3.3, 0, 1)
    rgba = np.array([0.55 - 0.25 * s, 0.75 - 0.4 * s, 0.55 - 0.25 * s, 1.0])
    plant.RegisterVisualGeometry(plant.world_body(), X, Box(*TABLE_SIZE), "table_vis", rgba)


def solve_start_ik(plant, tool, scene: Scene, n_restarts: int = 8):
    """IK for the start pose: ball centre at scene.start, tool +z along -n, yaw free."""
    world = plant.world_frame()
    n = scene.arr("n")
    rng = np.random.default_rng(scene.seed % (2 ** 32))
    best, best_cost = None, np.inf
    for k in range(n_restarts):
        ik = InverseKinematics(plant, with_joint_limits=True)
        q = ik.q()
        p = scene.arr("start")
        ik.AddPositionConstraint(tool.body_frame(), np.zeros(3), world, p - 1e-5, p + 1e-5)
        ik.AddAngleBetweenVectorsConstraint(world, -n, tool.body_frame(), np.array([0.0, 0.0, 1.0]), 0.0, 1e-4)
        ik.prog().AddQuadraticErrorCost(np.eye(7), Q_NOMINAL, q)
        q0 = Q_NOMINAL + (0 if k == 0 else 1) * rng.normal(0, 0.3, 7)
        ik.prog().SetInitialGuess(q, q0)
        res = Solve(ik.prog())
        if res.is_success():
            qs = res.GetSolution(q)
            c = float(np.sum((qs - Q_NOMINAL) ** 2))
            if c < best_cost:
                best, best_cost = qs, c
    if best is None:
        raise RuntimeError(f"start IK failed for scene {scene.table}/{scene.split}/{scene.index}")
    return best


class WipeEnv:
    """One episode on one scene.

    controller = "impedance": step(x_d) holds the target x_d for DT_POLICY.
    controller = "hybrid": the hybrid baseline runs on its own; call step(None).
    """

    def __init__(self, scene: Scene, gains: Gains = MAIN_GAINS, controller: str = "impedance",
                 task: TaskParams = TASK, record_q: bool = False, hybrid_kwargs=None):
        self.scene, self.gains, self.task, self.kind = scene, gains, task, controller
        self.n = scene.arr("n")
        builder = DiagramBuilder()
        cfg = MultibodyPlantConfig(time_step=DT_SIM, contact_model="point",
                                   discrete_contact_approximation="lagged")
        plant, sg = AddMultibodyPlant(cfg, builder)
        self.model, self.tool = add_robot_and_tool(plant, scene.k_e)
        add_table(plant, scene)
        plant.Finalize()
        self.plant, self.scene_graph = plant, sg

        self.q0 = solve_start_ik(plant, self.tool, scene)
        ctx0 = plant.CreateDefaultContext()
        plant.SetPositions(ctx0, self.q0)
        X0 = plant.CalcRelativeTransform(ctx0, plant.world_frame(), self.tool.body_frame())
        self.R0 = X0.rotation().matrix()

        sensor = builder.AddSystem(ToolForceSensor(plant, self.tool))
        builder.Connect(plant.get_contact_results_output_port(), sensor.get_input_port())
        if controller == "impedance":
            ctrl = builder.AddSystem(ImpedanceController(plant, self.tool, gains, self.n, self.R0, self.q0))
            builder.ExportInput(ctrl.xd_in, "x_d")
        elif controller == "hybrid":
            ctrl = builder.AddSystem(HybridForcePositionController(
                plant, self.tool, gains, scene, task, self.q0, self.R0, **(hybrid_kwargs or {})))
            builder.Connect(sensor.get_output_port(), ctrl.f_in)
        else:
            raise ValueError(controller)
        builder.Connect(plant.get_state_output_port(self.model), ctrl.state_in)
        builder.Connect(ctrl.get_output_port(0), plant.get_actuation_input_port(self.model))
        peak = builder.AddSystem(PeakForceMonitor(self.n))
        builder.Connect(sensor.get_output_port(), peak.f_in)
        self.ctrl, self.sensor, self.peak = ctrl, sensor, peak
        self.diagram = builder.Build()

        self.simulator = Simulator(self.diagram)
        self.root = self.simulator.get_mutable_context()
        self.pctx = plant.GetMyMutableContextFromRoot(self.root)
        plant.SetPositions(self.pctx, self.model, self.q0)
        plant.SetVelocities(self.pctx, self.model, np.zeros(7))
        self.sensor_ctx = sensor.GetMyContextFromRoot(self.root)
        self.peak_ctx = peak.GetMyContextFromRoot(self.root)
        if controller == "impedance":
            self.xd_port = self.diagram.get_input_port(0)
            self.xd_port.FixValue(self.root, np.asarray(scene.start, float))
        self.simulator.Initialize()

        self.k = 0
        self.coverage = Coverage(scene.L, task)
        self.record_q = record_q
        self.q_log = [self.q0.copy()] if record_q else None
        self.samples = []     # 100 Hz: (t, u1, u2, h, fn)
        self._sample()

    # ------------------------------------------------------------------ state
    def tool_state(self):
        X = self.plant.EvalBodyPoseInWorld(self.pctx, self.tool)
        V = self.plant.EvalBodySpatialVelocityInWorld(self.pctx, self.tool)
        return X.translation().copy(), V.translational().copy()

    def force(self):
        return self.sensor.get_output_port().Eval(self.sensor_ctx).copy()

    def observe(self) -> dict:
        x, xd = self.tool_state()
        return dict(t=self.k * DT_POLICY, x=x, xdot=xd, f=self.force(),
                    q=self.plant.GetPositions(self.pctx, self.model).copy(),
                    u=self.coverage.uncovered)

    def _sample(self):
        x, _ = self.tool_state()
        u = self.scene.to_plane(x)
        fn = float(self.n @ self.force())
        self.coverage.add(u[0], u[1], fn)
        self.samples.append((self.root.get_time(), u[0], u[1], u[2] - BALL_RADIUS, fn))

    # ------------------------------------------------------------------ stepping
    def step(self, x_d=None) -> dict:
        if self.kind == "impedance":
            self.xd_port.FixValue(self.root, np.asarray(x_d, float))
        for i in range(1, SUBSTEPS + 1):
            self.simulator.AdvanceTo(self.k * DT_POLICY + i * DT_POLICY / SUBSTEPS)
            self._sample()
            if self.record_q:
                self.q_log.append(self.plant.GetPositions(self.pctx, self.model).copy())
        self.k += 1
        return self.observe()

    @property
    def peak_force(self) -> float:
        return float(self.peak_ctx.get_discrete_state_vector().GetAtIndex(0))

    def hybrid_mode(self) -> int:
        return int(self.ctrl.GetOutputPort("mode").Eval(self.ctrl.GetMyContextFromRoot(self.root))[0])

    # ------------------------------------------------------------------ metrics
    def metrics(self) -> dict:
        S = np.array(self.samples)
        L, r = self.scene.L, self.task.r_wipe
        inreg = (S[:, 1] >= -r) & (S[:, 1] <= L + r) & (S[:, 2] >= -r) & (S[:, 2] <= L + r)
        near = S[:, 3] <= 0.005
        w = inreg & near
        contact = S[:, 4] > 0.0
        fn_w = S[w, 4]
        u = self.coverage.uncovered
        peak = self.peak_force
        return dict(
            uncovered=u, peak_fn=peak,
            success=bool(u <= self.task.eps_u and peak <= self.task.f_max),
            mean_fn_wipe=float(fn_w.mean()) if w.any() else 0.0,
            std_fn_wipe=float(fn_w.std()) if w.any() else 0.0,
            frac_wipe_ok=float((fn_w >= self.task.f_min).mean()) if w.any() else 0.0,
            n_wipe_samples=int(w.sum()),
            ever_contact=bool(contact.any()),
            first_contact_t=float(S[contact, 0][0]) if contact.any() else float("nan"),
            duration=float(self.k * DT_POLICY),
        )
