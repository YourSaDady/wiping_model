"""P2/P4 diagnostic: does a policy's commanded lead depend on the velocity it observes?

For validation windows of the sweep phase, perturb the observed tool velocity along a
direction e (lane direction t1, or the normal n) by ±dv in both observation frames and
measure the change of the first commanded target along e:
    G_e = d(target_0 - x_t) / d(xdot)  [s].
The controller applies K(x_d - x) - D xdot, so a policy that commands a lead G·xdot leaves
a net closed-loop damping of D - K G along e. G = D/K cancels the controller's damping;
G > D/K is negative damping (velocity runaway).

Usage: python results/p2_velocity_gain.py --runs runs/K500_b_s0,runs/K500_cp_s0,...
Output: results/tables/p2_velocity_gain.csv
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from wipe.config import GAINS_BY_KN
from wipe.data import load_episode
from wipe.dp.dataset import episode_arrays, make_windows
from wipe.policy import DiffusionPolicy

ap = argparse.ArgumentParser()
ap.add_argument("--runs", required=True)
ap.add_argument("--dv", type=float, default=0.01)
ap.add_argument("--n-eps", type=int, default=12)
args = ap.parse_args()
rows = []
for run in args.runs.split(","):
    sel = json.load(open(os.path.join(run, "selected.json")))
    pol = DiffusionPolicy(run, sel["ckpt"])
    cfg = pol.cfg
    g = GAINS_BY_KN[cfg["K"]]
    for p in cfg["val_paths"][::max(1, len(cfg["val_paths"]) // args.n_eps)][:args.n_eps]:
        ep = load_episode(p)
        sc = ep["scene"]
        obs, lab = episode_arrays(ep, cfg["label"], cfg["K"], cfg.get("obs_version", 1))
        O, _ = make_windows(obs, lab, ep["x"], cfg["To"], cfg["H"], cfg["relative"])
        idx = np.where(ep["phase"] == 2)[0][::6]
        x = ep["x"][idx]
        for name, e in (("t1", sc.arr("t1")), ("n", sc.arr("n"))):
            outs = []
            for s in (+1, -1):
                Op = O[idx].copy()
                Op[:, :, 3:6] += s * args.dv * e                     # velocity channel, both frames
                A = pol.act(Op, x, np.full(len(idx), 11), np.arange(len(idx)))
                outs.append((A[:, 0] - x) @ e)
            G = (outs[0] - outs[1]) / (2 * args.dv)
            D, K = (g.D_t, g.K_t) if name == "t1" else (g.D_n, g.K_n)
            rows.append(dict(run=os.path.basename(run), label=cfg["label"], K_n=cfg["K"], table=sc.table,
                             dir=name, G_s=float(np.median(G)), D_over_K=D / K,
                             net_damping=float(D - K * np.median(G))))
    print(run, flush=True)
df = pd.DataFrame(rows)
df.to_csv("results/tables/p2_velocity_gain.csv", index=False)
print(df.groupby(["run", "dir"]).agg(G_s=("G_s", "median"), D_over_K=("D_over_K", "first"),
                                     net_damping=("net_damping", "median")).round(3).to_string())
