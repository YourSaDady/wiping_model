"""Oct 30 milestone figure: one scripted-expert episode on a flat table (K_n = 500), showing
every logged 20 Hz signal: ball position x and commanded target x_d (region frame), velocity
xdot, contact force f, and uncovered fraction u. Also saves the raw episode log.

Outputs: results/figures/milestone_flat_log.png, results/milestone/flat_episode.npz
Usage:   python results/fig_milestone_log.py [table] [dev scene index]
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wipe.config import MAIN_GAINS, TASK
from wipe.expert import PHASE_ID
from wipe.rollout import run_expert
from wipe.scene import make_scene

table = sys.argv[1] if len(sys.argv) > 1 else "tilt00_ke2e4"
idx = int(sys.argv[2]) if len(sys.argv) > 2 else 0
sc = make_scene(table, "dev", idx)
ep = run_expert(sc, MAIN_GAINS)
m = ep["metrics"]
print(table, "dev", idx, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items()})

os.makedirs("results/milestone", exist_ok=True)
np.savez_compressed("results/milestone/flat_episode.npz",
                    **{k: ep[k] for k in ("t", "x", "xdot", "f", "q", "u", "x_d", "phase")},
                    scene=str(sc.to_dict()), metrics=str(m))

t, n = ep["t"], sc.arr("n")
x = sc.to_plane(ep["x"])                       # (u1, u2, h) of the ball centre
xd = sc.to_plane(ep["x_d"])
R = np.stack([sc.arr("t1"), sc.arr("t2"), n])  # world -> region frame rotation
v = ep["xdot"] @ R.T
f = ep["f"] @ R.T
td = t[:-1]                                    # x_d[k] is held on [t_k, t_k+1)

fig, ax = plt.subplots(5, 1, figsize=(11, 11), sharex=True)
for i, (lab, c) in enumerate([("u1 (lane dir.)", "C0"), ("u2 (step dir.)", "C2")]):
    ax[0].plot(t, x[:, i], color=c, label=f"x {lab}")
    ax[0].step(td, xd[:, i], where="post", color=c, ls="--", lw=1, label=f"x_d {lab}")
ax[0].set_ylabel("in-plane [m]")
ax[1].plot(t, 1e3 * x[:, 2], label="x (ball centre)")
ax[1].step(td, 1e3 * xd[:, 2], where="post", ls="--", lw=1, label="x_d")
ax[1].axhline(1e3 * 0.02, color="k", lw=0.6, ls=":", label="surface + R")
ax[1].set_ylabel("height h along n [mm]")
for i, lab in enumerate(["t1", "t2", "n"]):
    ax[2].plot(t, v[:, i], lw=1, label=f"xdot·{lab}")
ax[2].set_ylabel("velocity [m/s]")
ax[3].plot(t, f[:, 2], label="f_n")
ax[3].plot(t, np.hypot(f[:, 0], f[:, 1]), lw=1, label="|f_t|")
for y, ls in [(TASK.f_min, ":"), (TASK.f_des, "-"), (TASK.f_max, "--")]:
    ax[3].axhline(y, color="gray", lw=0.7, ls=ls)
ax[3].set_ylabel("contact force [N]")
ax[4].plot(t, ep["u"], color="k", label="u (uncovered fraction)")
ax[4].axhline(TASK.eps_u, color="gray", lw=0.7, ls="--")
ax[4].set_ylabel("uncovered u")
ax[4].set_xlabel("t [s]")
names = {v_: k_ for k_, v_ in PHASE_ID.items()}
ph = ep["phase"]
edges = np.flatnonzero(np.diff(ph)) + 1
for a in ax:
    for e in edges:
        a.axvline(td[e], color="gray", lw=0.5, alpha=0.6)
    a.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=7)
for k, (s_, e) in enumerate(zip(np.r_[0, edges], np.r_[edges, len(ph)])):
    ax[1].text(0.5 * (td[s_] + td[min(e, len(td) - 1)]), 50 - 8 * (k % 2), names[ph[s_]],
               fontsize=7, ha="center", color="gray")
ax[0].set_title(f"Scripted expert, {table} (dev scene {idx}), K_n = {MAIN_GAINS.K_n:.0f} N/m\n"
                f"success={m['success']}, uncovered={m['uncovered']:.3f}, peak 1 kHz f_n={m['peak_fn']:.1f} N",
                fontsize=10)
fig.tight_layout()
os.makedirs("results/figures", exist_ok=True)
fig.savefig("results/figures/milestone_flat_log.png", dpi=130)
print("wrote results/figures/milestone_flat_log.png, results/milestone/flat_episode.npz")
