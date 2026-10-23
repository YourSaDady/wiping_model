"""Render a recorded rollout to mp4: Drake VTK camera view + normal force + coverage map.

Input: a trace .npz written by eval_policy.py / eval_baselines.py with --record-q
(contains scene, q_hi at 100 Hz, samples (t, u1, u2, gap, fn) at 100 Hz).

Usage:
  python scripts/render_rollout.py --trace results/raw/X_test_traces/tilt00_ke2e4_000.npz \
      --title "label (b) x_d" --out results/videos/b.mp4 [--speed 2] [--fps 25]
  python scripts/render_rollout.py --grid a.npz,b.npz,c.npz,cp.npz --titles "(a),(b),(c),(c')" --out grid.mp4
"""
import argparse
import json
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from pydrake.all import (AddMultibodyPlant, Box, CameraInfo, ClippingRange, DepthRange, DepthRenderCamera,
                         DiagramBuilder, MakeRenderEngineVtk, MultibodyPlantConfig, RenderCameraCore,
                         RenderEngineVtkParams, RgbdSensor, RigidTransform, RollPitchYaw, SceneGraph)

from wipe.config import TASK
from wipe.coverage import Coverage
from wipe.scene import Scene
from wipe.sim import add_robot_and_tool, add_table, table_pose

W, H = 640, 480


class Renderer:
    def __init__(self, scene: Scene):
        b = DiagramBuilder()
        plant, sg = AddMultibodyPlant(MultibodyPlantConfig(time_step=1e-3), b)
        self.model, _ = add_robot_and_tool(plant, None)
        add_table(plant, scene)
        # region outline: thin dark strips on the surface
        X_T = table_pose(scene)
        L = scene.L
        p0 = scene.arr("p0")
        for k, (a, bb, sx, sy) in enumerate([(0.5, 0, L, 0.003), (0.5, 1, L, 0.003), (0, 0.5, 0.003, L), (1, 0.5, 0.003, L)]):
            c = p0 + a * L * scene.arr("t1") + bb * L * scene.arr("t2") + 0.0006 * scene.arr("n")
            plant.RegisterVisualGeometry(plant.world_body(), RigidTransform(X_T.rotation(), c), Box(sx, sy, 0.001),
                                         f"region{k}", np.array([0.1, 0.1, 0.1, 1.0]))
        plant.Finalize()
        params = RenderEngineVtkParams()
        params.backend = "EGL"
        sg.AddRenderer("vtk", MakeRenderEngineVtk(params))
        core = RenderCameraCore("vtk", CameraInfo(W, H, np.deg2rad(40)), ClippingRange(0.05, 10), RigidTransform())
        c = scene.arr("c")
        eye = c + np.array([0.62, -0.38, 0.36])
        X_WC = look_at(eye, c + np.array([-0.04, 0.0, 0.05]))
        self.sensor = b.AddSystem(RgbdSensor(SceneGraph.world_frame_id(), X_WC,
                                             DepthRenderCamera(core, DepthRange(0.05, 10))))
        b.Connect(sg.get_query_output_port(), self.sensor.query_object_input_port())
        self.plant = plant
        self.diagram = b.Build()
        self.ctx = self.diagram.CreateDefaultContext()
        self.pctx = plant.GetMyMutableContextFromRoot(self.ctx)
        self.sctx = self.sensor.GetMyContextFromRoot(self.ctx)

    def render(self, q):
        self.plant.SetPositions(self.pctx, self.model, q)
        return np.array(self.sensor.color_image_output_port().Eval(self.sctx).data)[..., :3].copy()


def look_at(eye, target, up=np.array([0, 0, 1.0])):
    z = target - eye
    z /= np.linalg.norm(z)                      # camera +z looks forward
    x = np.cross(z, up); x /= np.linalg.norm(x)  # camera +x to the right
    y = np.cross(z, x)                          # camera +y down
    from pydrake.all import RotationMatrix
    return RigidTransform(RotationMatrix(np.stack([x, y, z], 1)), eye)


def load(path):
    z = np.load(path, allow_pickle=False)
    d = {k: z[k] for k in z.files}
    d["scene"] = Scene.from_dict(json.loads(str(d["scene"])))
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace")
    ap.add_argument("--grid", default="")
    ap.add_argument("--title", default="")
    ap.add_argument("--titles", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--speed", type=float, default=2.0)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--t-end", type=float, default=None)
    args = ap.parse_args()
    paths = args.grid.split(",") if args.grid else [args.trace]
    titles = args.titles.split(",") if args.titles else [args.title]
    tr = [load(p) for p in paths]
    rends = [Renderer(t["scene"]) for t in tr]
    t_end = args.t_end or max(t["samples"][-1, 0] for t in tr)
    times = np.arange(0, t_end, args.speed / args.fps)
    ncol = len(tr)
    fig = plt.figure(figsize=(4.4 * ncol, 6.0), dpi=100)
    gs = fig.add_gridspec(3, ncol, height_ratios=[3.2, 1.1, 1.5], hspace=0.45)
    axes = []
    for j, t in enumerate(tr):
        a_img = fig.add_subplot(gs[0, j]); a_img.axis("off")
        im = a_img.imshow(np.zeros((H, W, 3), np.uint8))
        a_img.set_title(titles[j] if j < len(titles) else "", fontsize=11)
        a_f = fig.add_subplot(gs[1, j])
        S = t["samples"]
        a_f.plot(S[:, 0], S[:, 4], color="0.8", lw=0.8)
        (lf,) = a_f.plot([], [], color="#2e86ab", lw=1.2)
        for yv, ls in ((TASK.f_des, "-"), (TASK.f_min, ":"), (TASK.f_max, "--")):
            a_f.axhline(yv, color="k", lw=0.5, ls=ls)
        a_f.set_xlim(0, t_end); a_f.set_ylim(-0.5, max(22, S[:, 4].max() * 1.05))
        a_f.set_ylabel("f_n [N]", fontsize=8); a_f.tick_params(labelsize=7)
        a_c = fig.add_subplot(gs[2, j])
        cov = Coverage(t["scene"].L)
        ci = a_c.imshow(cov.grid.T.astype(float), origin="lower", cmap="Greens", vmin=0, vmax=1.4,
                        extent=[0, t["scene"].L * 100, 0, t["scene"].L * 100])
        (tp,) = a_c.plot([], [], "o", color="#d1495b", ms=4)
        a_c.set_xlim(-2, t["scene"].L * 100 + 2); a_c.set_ylim(-2, t["scene"].L * 100 + 2)
        a_c.set_xlabel("u1 [cm]", fontsize=8); a_c.set_ylabel("u2 [cm]", fontsize=8); a_c.tick_params(labelsize=7)
        txt = a_f.text(0.01, 1.04, "", transform=a_f.transAxes, fontsize=8, family="monospace")
        axes.append(dict(im=im, lf=lf, ci=ci, tp=tp, cov=cov, txt=txt, k=0))
    fig.subplots_adjust(left=0.10, right=0.98, top=0.95, bottom=0.07, wspace=0.3)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    writer = FFMpegWriter(fps=args.fps, bitrate=2400)
    with writer.saving(fig, args.out, dpi=100):
        for tt in times:
            for t, r, ax in zip(tr, rends, axes):
                S = t["samples"]
                qi = min(int(round(tt * 100)), len(t["q_hi"]) - 1)
                ax["im"].set_data(r.render(t["q_hi"][qi]))
                while ax["k"] < len(S) and S[ax["k"], 0] <= tt + 1e-9:
                    ax["cov"].add(S[ax["k"], 1], S[ax["k"], 2], S[ax["k"], 4])
                    ax["k"] += 1
                k = max(ax["k"] - 1, 0)
                ax["lf"].set_data(S[:k + 1, 0], S[:k + 1, 4])
                ax["ci"].set_data(ax["cov"].grid.T.astype(float))
                ax["tp"].set_data([S[k, 1] * 100], [S[k, 2] * 100])
                ax["txt"].set_text(f"t={tt:5.1f}s  f_n={S[k, 4]:5.1f}N  uncovered={ax['cov'].uncovered:.2f}")
            writer.grab_frame()
    print("wrote", args.out)


if __name__ == "__main__":
    main()
