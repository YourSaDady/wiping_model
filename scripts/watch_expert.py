"""Run the scripted expert on one scene with a live Meshcat view and a live top view of the wiped region.

    python scripts/watch_expert.py                                  # tilt00_ke2e4, dev scene 0, real time
    python scripts/watch_expert.py tilt20_ke1e5 3 --rate 0.5        # a preset table, scene 3, half speed
    python scripts/watch_expert.py --tilt 15 --ke 5e4 --v 0.08 --fdes 12 --mu 0.4
    python scripts/watch_expert.py --no-meshcat --rate 0 --save-topview top.png

The table and the scene come from a preset (positional `table index`, see wipe/config.py TABLES);
the options override single parameters of that scene and keep the rest (position, height, start pose):

    --tilt  table tilt about world x [deg]        --ke    normal contact stiffness k_e [N/m]
    --mu    Coulomb friction of ball and table    --v     wiping speed along the lanes [m/s]
    --fdes  normal force the expert holds [N]     --L     side of the square region [m]
    --K     controller normal stiffness K_n (250 / 500 / 1000 N/m)

The success test is never changed by these options: f_min, f_max, eps_u, w_min, r_wipe and the time
limit are the task's (wipe/config.py TASK, T_MAX), so a scene that the expert cannot wipe shows up as a
failure, with the reason printed at the end and shown in the top view.

Meshcat (open the printed URL): the robot, the dirty region (pink), the planned lanes (blue) and the
wiped cells (white), updated while the episode runs. The episode is also recorded, so after it ends it
can be replayed with Meshcat's Animation controls.

Top view (matplotlib): the coverage grid in region coordinates, coloured by each cell's wiping dose
(wipe/coverage.py; README "Success criterion"). Any pink shade = not wiped yet, from deep pink (no dose)
to pale pink (almost w_min); white = wiped (dose >= w_min). Also the lane path, the tool trail, and the
patch of radius r_wipe around the contact point (green while it is wiping, red otherwise). The panel on
the right lists the scene parameters and the live success checks (u vs eps_u, 1 kHz peak f_n vs f_max,
time vs the limit).
"""
import argparse
import os
import sys
import time
from dataclasses import replace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wipe.config import DT_POLICY, FRICTION, GAINS_BY_KN, T_MAX, TABLES, TASK  # noqa: E402
from wipe.expert import ScriptedExpert  # noqa: E402
from wipe.paths import LanePath  # noqa: E402
from wipe.scene import make_scene  # noqa: E402
from wipe.sim import WipeEnv  # noqa: E402


# ---------------------------------------------------------------------------- Meshcat overlays
def draw_region(meshcat, scene, lane_path):
    from pydrake.all import Box, RigidTransform, RotationMatrix, Rgba

    R = RotationMatrix(np.column_stack([scene.arr("t1"), scene.arr("t2"), scene.arr("n")]))
    L = scene.L
    meshcat.SetObject("/wipe/dirt", Box(L, L, 0.001), Rgba(0.85, 0.11, 0.38, 0.9))   # same pink as the top view
    meshcat.SetTransform("/wipe/dirt", RigidTransform(R, scene.from_plane(0.5 * L, 0.5 * L, 0.0006)))
    P = lane_path.P[1:]                      # skip the approach start point
    pts = np.array([scene.from_plane(u1, u2, 0.002) for u1, u2 in P])
    meshcat.SetLine("/wipe/lanes", pts.T.copy(), 2.0, Rgba(0.1, 0.4, 0.9, 1.0))
    meshcat.Delete("/wipe/cleaned")


def draw_cleaned(meshcat, scene, cov):
    """Cleaned cells as a flat white triangle mesh 1.5 mm above the surface.

    Contiguous cleaned cells of each grid column are merged into one quad, so the mesh stays at a few hundred
    triangles. (A point cloud renders each cell as a screen-facing sprite, which looks like a field of little
    upright squares when the table is seen from the side.)
    """
    from pydrake.all import Rgba

    g, c = cov.grid, cov.cell
    verts, faces = [], []
    for i in range(g.shape[0]):
        col = np.concatenate([[False], g[i], [False]]).astype(np.int8)
        d = np.diff(col)
        for j0, j1 in zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]):
            k = len(verts)
            for u1, u2 in [(i * c, j0 * c), ((i + 1) * c, j0 * c), ((i + 1) * c, j1 * c), (i * c, j1 * c)]:
                verts.append(scene.from_plane(u1, u2, 0.0015))
            faces += [(k, k + 1, k + 2), (k, k + 2, k + 3)]
    if not faces:
        return
    meshcat.SetTriangleMesh("/wipe/cleaned", np.array(verts).T, np.array(faces, dtype=np.int32).T,
                            Rgba(0.95, 0.97, 1.0, 1.0))


# ---------------------------------------------------------------------------- matplotlib top view
class TopView:
    def __init__(self, scene, cov, lane_path, task, params):
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap

        self.plt, self.scene, self.cov, self.task, self.params = plt, scene, cov, task, params
        plt.ion()
        self.fig = plt.figure(figsize=(12, 6))
        self.ax = self.fig.add_axes([0.06, 0.09, 0.46, 0.82])
        cax = self.fig.add_axes([0.535, 0.09, 0.013, 0.82])
        self.tax = self.fig.add_axes([0.61, 0.02, 0.38, 0.96])
        self.tax.axis("off")
        L, r = scene.L, task.r_wipe
        # dirt shade from deep pink (no dose) to pale pink (almost done); dose >= w_min is drawn white ("over")
        cmap = LinearSegmentedColormap.from_list("dirt", ["#d81b60", "#f06292", "#fbd9e6"])
        cmap.set_over("#ffffff")
        self.im = self.ax.imshow(self._level(), origin="lower", extent=[0, L, 0, L], vmin=0.0, vmax=1.0 - 1e-9,
                                 cmap=cmap, interpolation="nearest")
        cb = self.fig.colorbar(self.im, cax=cax, extend="max")
        cb.set_label("wiping progress  (white = wiped)" if task.coverage_mode == "force"
                     else "friction work / w_min  (white = wiped)")
        self.ax.add_patch(plt.Rectangle((0, 0), L, L, fill=False, color="k", lw=1))
        self.ax.set_facecolor("#d9d9d9")          # grey outside the region, so white cells stand out
        P = lane_path.P[1:]
        self.ax.plot(P[:, 0], P[:, 1], "b-", lw=0.8, alpha=0.6, label="lane path")
        (self.trail,) = self.ax.plot([], [], color="0.3", lw=0.8, label="tool")
        (self.dot,) = self.ax.plot([], [], "o", ms=8, mfc="none", mec="r", mew=2, label="contact point")
        self.patch = plt.Circle((0, 0), r, fill=False, color="r", ls="--", lw=1)
        self.ax.add_patch(self.patch)
        self.ax.set_xlim(-2 * r, L + 2 * r)
        self.ax.set_ylim(-2 * r, L + 2 * r)
        self.ax.set_aspect("equal")
        self.ax.set_xlabel("u1 along t1 (m)")
        self.ax.set_ylabel("u2 along t2 (m)")
        self.ax.legend(loc="upper right", fontsize=7)
        self.txt = self.tax.text(0.0, 1.0, "", va="top", ha="left", family="monospace", fontsize=8.5,
                                 transform=self.tax.transAxes)
        self.verdict = self.tax.text(0.0, 0.02, "", va="bottom", ha="left", fontsize=11, weight="bold",
                                     transform=self.tax.transAxes)
        self.uv = []
        self.fig.canvas.draw()
        plt.show(block=False)

    def _level(self):
        return (self.cov.dose / self.task.w_min).T

    def update(self, t, phase, x, fn, p_fric, u, peak, fn_mean):
        u1, u2, _ = self.scene.to_plane(x)
        self.uv.append((u1, u2))
        uv = np.array(self.uv)
        self.im.set_data(self._level())
        self.trail.set_data(uv[:, 0], uv[:, 1])
        self.dot.set_data([u1], [u2])
        wiping = fn >= self.task.f_min if self.task.coverage_mode == "force" else p_fric > 0.0
        self.dot.set_markeredgecolor("g" if wiping else "r")
        self.patch.center = (u1, u2)
        self.patch.set_color("g" if wiping else "r")
        self.ax.set_title(f"t = {t:5.2f} s   phase: {phase}", fontsize=11)
        tk = self.task
        ok = lambda c: "ok " if c else "BAD"   # noqa: E731
        self.txt.set_text(
            self.params
            + "\n\nlive\n"
            + f"  f_n now          {fn:6.2f} N\n"
            + f"  f_n mean (sweep) {fn_mean:6.2f} N\n"
            + f"  friction power   {p_fric:6.3f} W\n"
            + "\nsuccess checks\n"
            + f"  {ok(u <= tk.eps_u)} uncovered u   {u:5.3f}  (<= {tk.eps_u})\n"
            + f"  {ok(peak <= tk.f_max)} peak f_n    {peak:5.1f} N (<= {tk.f_max:g} N, 1 kHz)\n"
            + f"  {ok(t < T_MAX)} time        {t:5.1f} s (< {T_MAX:g} s)")
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def set_verdict(self, ok, reasons):
        self.verdict.set_text("SUCCESS" if ok else "FAIL: " + "; ".join(reasons))
        self.verdict.set_color("green" if ok else "red")
        self.verdict.set_wrap(True)
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def save(self, path):
        self.fig.savefig(path, dpi=120)


def param_text(scene, gains, mu, f_des, task):
    return ("scene\n"
            f"  table preset     {scene.table} / {scene.split} {scene.index}\n"
            f"  tilt             {scene.tilt_deg:6.1f} deg\n"
            f"  k_e              {scene.k_e:8.3g} N/m\n"
            f"  mu (ball/table)  {mu:6.2f}\n"
            f"  region L         {100 * scene.L:6.1f} cm, lanes s = {100 * scene.lane_spacing:.1f} cm\n"
            "expert / controller\n"
            f"  wiping speed v   {scene.v:6.3f} m/s\n"
            f"  f_des            {f_des:6.1f} N\n"
            f"  K_n / K_t        {gains.K_n:g} / {gains.K_t:g} N/m\n"
            f"  D_n / D_t        {gains.D_n:g} / {gains.D_t:g} N s/m\n"
            "task (success test)\n"
            f"  r_wipe {100 * task.r_wipe:g} cm, w_min {task.w_min:g} J ({task.coverage_mode})\n"
            f"  f_min {task.f_min:g} N, f_max {task.f_max:g} N, eps_u {task.eps_u:g}")


def failure_reasons(m, task, expert_done):
    r = []
    if m["uncovered"] > task.eps_u:
        r.append(f"uncovered {m['uncovered']:.2f} > {task.eps_u}")
    if m["peak_fn"] > task.f_max:
        r.append(f"peak f_n {m['peak_fn']:.1f} N > {task.f_max:g} N")
    if not expert_done:
        r.append(f"not finished in {T_MAX:g} s")
    return r


# ---------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("table", nargs="?", default="tilt00_ke2e4", choices=sorted(TABLES))
    ap.add_argument("index", nargs="?", type=int, default=0)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--tilt", type=float, default=None, help="table tilt [deg] (overrides the preset)")
    ap.add_argument("--ke", type=float, default=None, help="contact stiffness k_e [N/m] (overrides the preset)")
    ap.add_argument("--mu", type=float, default=FRICTION, help=f"Coulomb friction (default {FRICTION})")
    ap.add_argument("--v", type=float, default=None, help="wiping speed [m/s] (default: sampled per scene)")
    ap.add_argument("--fdes", type=float, default=TASK.f_des, help=f"expert normal force [N] (default {TASK.f_des})")
    ap.add_argument("--L", type=float, default=None, help="region side [m] (default: sampled per scene)")
    ap.add_argument("--K", type=int, default=500, choices=sorted(GAINS_BY_KN), help="controller K_n [N/m]")
    ap.add_argument("--rate", type=float, default=1.0, help="target realtime rate (0 = as fast as possible)")
    ap.add_argument("--update-every", type=int, default=2, help="redraw every N policy steps (20 Hz)")
    ap.add_argument("--no-meshcat", action="store_true")
    ap.add_argument("--no-topview", action="store_true")
    ap.add_argument("--save-topview", default=None, help="save the final top view to this png")
    args = ap.parse_args()

    scene = make_scene(args.table, args.split, args.index, tilt_deg=args.tilt, k_e=args.ke, v=args.v, L=args.L)
    gains = GAINS_BY_KN[args.K]
    task = TASK                                   # success test: never changed here
    expert_task = replace(TASK, f_des=args.fdes)  # only the expert's force set point changes
    meshcat = None
    if not args.no_meshcat:
        from pydrake.all import StartMeshcat

        meshcat = StartMeshcat()
        print("Meshcat:", meshcat.web_url(), "(open it now; the episode starts in 3 s)")
        time.sleep(3.0)

    try:
        env = WipeEnv(scene, gains, task=task, meshcat=meshcat, mu=args.mu)
    except RuntimeError as e:                     # start pose unreachable (very steep tilts)
        sys.exit(f"FAIL before the episode: {e}")
    ex = ScriptedExpert(scene, gains, task=expert_task)
    env.simulator.set_target_realtime_rate(args.rate)
    # the expert builds its LanePath at contact from the contact point; for display, build it from the start
    lanes = LanePath(scene, tuple(scene.to_plane(scene.arr("start"))[:2]))
    n = scene.arr("n")

    if meshcat is not None:
        # the visualizer already sent the robot and table geometry when WipeEnv initialized the simulator;
        # deleting /drake/visualizer here would remove them for good (later publishes only move them)
        draw_region(meshcat, scene, lanes)
        meshcat.SetCameraPose(np.array([1.4, -1.1, 0.9]), scene.arr("c"))
        env.visualizer.StartRecording()
        env.diagram.ForcedPublish(env.root)
    params = param_text(scene, gains, args.mu, args.fdes, task)
    print(params)
    view = None if args.no_topview else TopView(scene, env.coverage, lanes, task, params)

    obs = env.observe()
    max_steps = int(round(T_MAX / DT_POLICY))
    last_phase, fn_sweep = None, []

    def redraw():
        fn = float(n @ obs["f"])
        if meshcat is not None:
            draw_cleaned(meshcat, scene, env.coverage)
        if view is not None:
            view.update(obs["t"], ex.phase, obs["x"], fn, env.samples[-1][5], obs["u"], env.peak_force,
                        float(np.mean(fn_sweep)) if fn_sweep else 0.0)

    while not ex.done and env.k < max_steps:
        xd = ex.act(obs)
        if ex.done:
            break
        if ex.phase != last_phase:
            print(f"  t={obs['t']:6.2f}s  -> {ex.phase}")
            last_phase = ex.phase
        obs = env.step(xd)
        if ex.phase == "sweep":
            fn_sweep.append(float(n @ obs["f"]))
        if env.k % args.update_every == 0:
            redraw()

    m = env.metrics()
    redraw()
    reasons = failure_reasons(m, task, ex.done)
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items()})
    print("SUCCESS" if m["success"] else "FAIL: " + "; ".join(reasons))
    if meshcat is not None:
        env.visualizer.StopRecording()
        env.visualizer.PublishRecording()
    if view is not None:
        view.set_verdict(m["success"], reasons)
        if args.save_topview:
            view.save(args.save_topview)
            print("top view saved to", args.save_topview)
    if (meshcat is not None or view is not None) and sys.stdin.isatty():
        input("Done. Meshcat has a replayable recording (Animation controls). Press Enter to quit.")


if __name__ == "__main__":
    main()
