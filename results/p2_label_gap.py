"""P2 (offline): how far labels (c) and (c') are from the commanded target (b),
by expert phase and by tool speed, on all demonstrations.

gap_c  = || x_{d,k} - (x_{k+1} - K^-1 f_{k+1}) ||
gap_cp = || x_{d,k} - (x_{k+1} + K^-1 (D xdot_{k+1} - f_{k+1})) ||
split into the component along n and the in-plane component.

Phases: approach, onset (first 0.5 s after the first contact sample), press
(rest of the press phase), sweep (straight: path speed = v), sweep-corner
(inside the corner slowdown), lift.

Outputs: results/tables/p2_gap_by_phase.csv, results/tables/p2_gap_by_speed.csv,
results/figures/p2_label_gap.png
Usage: python results/p2_label_gap.py
"""
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wipe.config import GAINS_BY_KN, TRAIN_TABLES
from wipe.data import list_demos, load_episode
from wipe.labels import make_labels

os.makedirs("results/tables", exist_ok=True)
rows = []
for K_n in (500, 250, 1000):
    g = GAINS_BY_KN[K_n]
    for t in TRAIN_TABLES:
        for p in list_demos(K_n, t.name):
            ep = load_episode(p)
            sc = ep["scene"]
            n = sc.arr("n")
            K, D = g.K(n), g.D(n)
            b = make_labels(ep["x"], ep["xdot"], ep["f"], ep["x_d"], K, D, "b")
            c = make_labels(ep["x"], ep["xdot"], ep["f"], ep["x_d"], K, D, "c")
            cp = make_labels(ep["x"], ep["xdot"], ep["f"], ep["x_d"], K, D, "cp")
            ph = ep["phase"].astype(int)
            fn = ep["f"][1:] @ n
            speed = np.linalg.norm(ep["xdot"][1:] - np.outer(ep["xdot"][1:] @ n, n), axis=1)
            T = len(b)
            first = int(np.argmax(fn > 0)) if (fn > 0).any() else T
            # path speed of the target in the sweep: corner slowdown shows as target speed < 0.98 v
            tgt_speed = np.r_[0, np.linalg.norm(np.diff(b - np.outer(b @ n, n), axis=0), axis=1) / 0.05]
            name = np.array(["approach", "press", "sweep", "lift"])[np.clip(ph, 0, 3)].astype(object)
            onset = (np.arange(T) >= first) & (np.arange(T) < first + 10)
            name[onset] = "onset"
            corner = (ph == 2) & (tgt_speed < 0.98 * sc.v)
            name[corner & ~onset] = "sweep-corner"
            for lab, arr in (("c", c), ("cp", cp)):
                d = b - arr
                dn = d @ n
                dt = np.linalg.norm(d - np.outer(dn, n), axis=1)
                rows.append(pd.DataFrame(dict(K_n=K_n, table=t.name, k_e=t.k_e, tilt=t.tilt_deg, v=sc.v,
                                              label=lab, phase=name, gap=np.linalg.norm(d, axis=1),
                                              gap_n=np.abs(dn), gap_t=dt, speed=speed, fn=fn)))
df = pd.concat(rows, ignore_index=True)

by_phase = df.groupby(["K_n", "label", "phase"]).agg(
    gap_mm=("gap", lambda s: 1e3 * s.mean()), gap_n_mm=("gap_n", lambda s: 1e3 * s.mean()),
    gap_t_mm=("gap_t", lambda s: 1e3 * s.mean()), gap_p95_mm=("gap", lambda s: 1e3 * s.quantile(0.95)),
    n=("gap", "size")).reset_index()
by_phase.to_csv("results/tables/p2_gap_by_phase.csv", index=False)
print(by_phase[by_phase.K_n == 500].to_string(index=False))

sw = df[(df.phase == "sweep")]
bins = np.arange(0.04, 0.135, 0.01)
sw = sw.assign(speed_bin=pd.cut(sw.speed, bins))
by_speed = sw.groupby(["K_n", "label", "speed_bin"], observed=True).agg(
    speed=("speed", "mean"), gap_mm=("gap", lambda s: 1e3 * s.mean()),
    gap_t_mm=("gap_t", lambda s: 1e3 * s.mean()), gap_n_mm=("gap_n", lambda s: 1e3 * s.mean()),
    n=("gap", "size")).reset_index()
by_speed.to_csv("results/tables/p2_gap_by_speed.csv", index=False)
print(by_speed[by_speed.K_n == 500].to_string(index=False))

fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
order = ["approach", "onset", "press", "sweep", "sweep-corner", "lift"]
bp = by_phase[by_phase.K_n == 500].set_index(["label", "phase"])
x = np.arange(len(order))
for j, (lab, col) in enumerate((("c", "#edae49"), ("cp", "#66a182"))):
    ax[0].bar(x + (j - 0.5) * 0.38, [bp.loc[(lab, p), "gap_mm"] for p in order], 0.38, color=col,
              label={"c": "(c) x - K^-1 f", "cp": "(c') x + K^-1 (D xdot - f)"}[lab])
ax[0].set_xticks(x, order, rotation=20)
ax[0].set_ylabel("mean ||x_d - label|| [mm]")
ax[0].set_title("Gap to the commanded target (b), by phase (K_n = 500)")
ax[0].legend(fontsize=8)
bs = by_speed[by_speed.K_n == 500]
for lab, col in (("c", "#edae49"), ("cp", "#66a182")):
    s = bs[bs.label == lab]
    ax[1].plot(1e2 * s.speed, s.gap_t_mm, "o-", color=col, label=f"{lab}: in-plane")
    ax[1].plot(1e2 * s.speed, s.gap_n_mm, "s--", color=col, alpha=0.6, label=f"{lab}: along n")
vv = np.linspace(4, 13, 10)
ax[1].plot(vv, 1e3 * GAINS_BY_KN[500].D_t / GAINS_BY_KN[500].K_t * vv / 1e2, "k:", lw=1,
           label="D_t v / K_t (dropped damping term)")
ax[1].set_xlabel("in-plane tool speed during straight sweep [cm/s]")
ax[1].set_ylabel("mean gap [mm]")
ax[1].set_title("Sweep: gap vs speed")
ax[1].legend(fontsize=7)
fig.tight_layout()
fig.savefig("results/figures/p2_label_gap.png", dpi=150)
print("saved results/figures/p2_label_gap.png")
