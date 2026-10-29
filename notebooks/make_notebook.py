"""Generates wiping_demo.ipynb (keeps the notebook diff-able as plain Python).

    python notebooks/make_notebook.py
"""

import json
from pathlib import Path

cells = []


def md(s):
    cells.append(dict(cell_type="markdown", metadata={}, source=s.strip("\n")))


def code(s):
    cells.append(dict(cell_type="code", metadata={}, execution_count=None, outputs=[], source=s.strip("\n")))


md(r"""
# Scripted wiping program: demo

6.4212 Fall 2026, group 17. This notebook walks through week 1 (Oct 30): the impedance-controlled iiwa with a ball
tool, tables with adjustable tilt and contact stiffness $k_e$, the scripted wiping program, and the per-step log
$(x_t, \dot x_t, f_t, x_{d,t}, u_t)$ with labels (a), (b), (c).

The notebook uses the same package as the command-line scripts (`../wiping`). With pydrake installed it uses the
Drake backend and Meshcat. Without pydrake it falls back to the point-mass test double, so every cell still runs
(the numbers then come from the test double, not from Drake).
""")

code(r"""
import sys, importlib.util
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))

from wiping import make_config, run_episode, PRESETS, PHASE_NAMES, DemoWriter, load_demos, make_labels
from wiping.lanes import plan_lanes
from wiping.plotting import plot_episode
from wiping.expert import WIPE, PRESS, APPROACH, LIFT

BACKEND = "drake" if importlib.util.find_spec("pydrake") else "pointmass"
print("backend:", BACKEND)
""")

code(r"""
meshcat = None
if BACKEND == "drake":
    from pydrake.all import StartMeshcat
    meshcat = StartMeshcat()   # open the printed URL
""")

md(r"""
## 1. One table and the plan

`make_config(preset)` returns an `EpisodeConfig` (table, region, controller, expert, sim, success test). Any field can
be overridden with `section__field=value`. The plan is a set of back-and-forth lanes with spacing $s < 2r$ over the
$L \times L$ region.
""")

code(r"""
cfg = make_config("flat_medium", backend=BACKEND)
print(cfg.table)
print(cfg.controller.gains_table(cfg.table.k_e), "  (K_t, K_n, D_t, D_n)")
wp = plan_lanes(cfg.region, cfg.expert)
c = cfg.region.corner_uv()
fig, ax = plt.subplots(figsize=(4, 4))
ax.add_patch(plt.Rectangle(c, cfg.region.L, cfg.region.L, color="#8c5a2b", alpha=0.3))
ax.plot(wp[:, 0], wp[:, 1], "b.-")
for p in wp[::2]:
    ax.add_patch(plt.Circle(p, cfg.expert.patch_r, fill=False, color="b", alpha=0.3))
ax.set_aspect("equal"); ax.set_title(f"{len(wp)//2} lanes, s={cfg.expert.lane_spacing}, r={cfg.expert.patch_r}")
""")

md(r"""
## 2. Run the scripted program

Phases: approach along $-\hat n$, press (integral loop on the commanded depth until $f_n \approx f_{des}$), wipe the
lanes at speed $v$ while the loop keeps holding the force, then lift. With Meshcat the run is recorded; use the
Animation controls in the Meshcat page to replay it.
""")

code(r"""
ep = run_episode(cfg, meshcat=meshcat, record=meshcat is not None, verbose=True)
ep["metrics"]
""")

code(r"""
plot_episode(ep);
""")

md(r"""
## 3. What gets logged, and the labels

Each step stores $x_t, \dot x_t, f_t, x_{d,t}, u_t$, the phase and the joint state. The labels are built from the same
log: (a) $x_{t+1}$, (b) $x_{d,t}$, (c) $x_{t+1} - K^{-1} f_{t+1}$, (c') $x_{t+1} + K^{-1}(D\dot x_{t+1} - f_{t+1})$.

This is the offline P2 check: how far (c) and (c') are from the true command (b) in each phase.
""")

code(r"""
lab = make_labels(ep)
n = cfg.table.normal()
b = lab["label_b_commanded"]
rows = []
for name in ["label_a_measured", "label_c_force_comp", "label_c_damp"]:
    err = np.linalg.norm(lab[name] - b, axis=1) * 1e3
    rows.append([name] + [np.median(err[ep["phase"] == p]) if (ep["phase"] == p).any() else np.nan
                          for p in (APPROACH, PRESS, WIPE, LIFT)])
print(f"{'median |label - x_d| (mm)':24s}" + "".join(f"{PHASE_NAMES[p]:>10s}" for p in (APPROACH, PRESS, WIPE, LIFT)))
for r in rows:
    print(f"{r[0]:24s}" + "".join(f"{v:10.2f}" for v in r[1:]))
""")

md(r"""
## 4. Open-loop replay of the labels (preview of P1)

Feed each label back as the target sequence from the same start. Label (a) puts the target on the surface, so the
spring hardly stretches and the wiping force collapses toward the quasi-static ratio $K/(K+k_e)$. (b) reproduces the
demo, and (c) comes close to it.
""")

code(r"""
wipe = ep["phase"] == WIPE
fn_demo = ep["f_n"][:-1][wipe].mean()
K_n = cfg.controller.K_n
print(f"demo wipe f_n = {fn_demo:.2f} N;  predicted (a): {fn_demo * K_n / (K_n + cfg.table.k_e):.2f} N")
for name in ["label_a_measured", "label_b_commanded", "label_c_force_comp", "label_c_damp"]:
    rp = run_episode(cfg, policy=lab[name])
    print(f"  replay {name:20s} wipe f_n = {rp['f_n'][:-1][wipe].mean():6.2f} N   u_T = {rp['u'][-1]:.3f}")
""")

md(r"""
## 5. Reliability across tables

The same program, unchanged, on the flat tables and the training tilts (one nominal episode each; use
`scripts/collect_demos.py --mode presets` for jittered repeats).
""")

code(r"""
names = ["flat_soft", "flat_medium", "flat_hard", "tilt10_medium", "tilt10_hard", "tilt15_soft"]
results = {}
for nm in names:
    e = run_episode(make_config(nm, backend=BACKEND), verbose=True)
    results[nm] = e
fig, ax = plt.subplots(figsize=(9, 3.5))
for nm, e in results.items():
    ax.plot(e["t"], e["f_n"], lw=1, label=nm)
ax.axhline(cfg.expert.f_des, color="k", ls="--", lw=0.8)
ax.set(xlabel="t (s)", ylabel="f_n (N)"); ax.legend(fontsize=7, ncol=3)
""")

md(r"""
Any table that is not a preset: override the fields directly. `table__tilt` / `table__tilt_dir` are in radians,
`table__k_e` in N/m; the region, friction and wiping speed work the same way.
""")

code(r"""
custom = make_config("flat_medium", backend=BACKEND, table__k_e=3.0e4, table__mu=0.6, table__tilt=np.radians(8),
                     table__tilt_dir=np.radians(45), region__L=0.10, expert__speed=0.07, name="custom")
e = run_episode(custom, verbose=True)
plot_episode(e);
""")

md(r"""
## 6. Collect demonstrations

`DemoWriter` concatenates episodes Diffusion-Policy style (`episode_ends`) with `obs` (16-D:
$x, \dot x, f, p_0, L, \hat n$), every label, and the per-episode $K$, $D$ and config. For real collection use
`scripts/collect_demos.py --mode sample --n 200 --workers 6`.
""")

code(r"""
w = DemoWriter()
for e in results.values():
    w.add(e)
path = w.save(ROOT / "data" / f"notebook_demo_{BACKEND}.npz")
d = load_demos(path)
print(path)
for k in ["obs", "x_d", "u", "label_a_measured", "label_b_commanded", "label_c_force_comp", "label_d_pose_force",
          "episode_ends", "K"]:
    print(f"  {k:20s} {d[k].shape}")
print("obs columns:", [str(s) for s in d["obs_names"]])
""")

nb = dict(cells=cells, metadata=dict(kernelspec=dict(display_name="Python 3", language="python", name="python3"),
                                     language_info=dict(name="python")), nbformat=4, nbformat_minor=5)
for i, c in enumerate(nb["cells"]):
    c["id"] = f"cell-{i}"
out = Path(__file__).with_name("wiping_demo.ipynb")
out.write_text(json.dumps(nb, indent=1))
print("wrote", out)
