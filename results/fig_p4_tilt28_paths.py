"""P4 detail: paths on the steeper unseen 28° tables.

In-plane contact-point path (region coordinates; colour = time, red = f_n < f_min) of the
seed-0 (b) and (c′) policies and the scripted expert on test scene 000 of both 28° OOD tables.
Also reports, over the 3 traced scenes x 3 seeds, the mean u2 of wiping samples (where the tool
actually wipes along the slope direction t2; the region spans u2 ∈ [0, L]).

Usage: python results/fig_p4_tilt28_paths.py
Outputs: results/figures/p4_tilt28_paths.png, results/tables/p4_tilt28_u2.csv
"""
import glob
import json

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wipe.config import TASK

methods = [("K500_b_s0", "(b) x_d"), ("K500_cp_s0", "(c') x + K^-1(D xdot - f)"), ("scripted_K500", "scripted expert")]
tables = ["tilt28_ke2e4", "tilt28_ke500"]
fig, ax = plt.subplots(2, 3, figsize=(13, 7.5))
for r, tab in enumerate(tables):
    for c, (m, title) in enumerate(methods):
        z = np.load(f"results/raw/{m}_test_traces/{tab}_000.npz")
        S, L = z["samples"], json.loads(str(z["scene"]))["L"] * 100
        ok = S[:, 4] >= TASK.f_min
        a = ax[r, c]
        a.scatter(S[ok, 1] * 100, S[ok, 2] * 100, s=1, c=S[ok, 0], cmap="viridis")
        a.scatter(S[~ok, 1] * 100, S[~ok, 2] * 100, s=1, c="r", alpha=0.15)
        a.plot([0, L, L, 0, 0], [0, 0, L, L, 0], "k--", lw=0.8)
        a.set_aspect("equal"); a.set_title(f"{title}\n{tab}, scene 000", fontsize=9)
        a.set_xlabel("u1 [cm] (lane direction)"); a.set_ylabel("u2 [cm] (up the slope)")
fig.tight_layout()
fig.savefig("results/figures/p4_tilt28_paths.png", dpi=150)
rows = []
for m in ["K500_b_s0", "K500_b_s1", "K500_b_s2", "K500_cp_s0", "K500_cp_s1", "K500_cp_s2", "scripted_K500"]:
    for tab in tables:
        for f in sorted(glob.glob(f"results/raw/{m}_test_traces/{tab}_*.npz")):
            S = np.load(f)["samples"]
            ok = S[:, 4] >= TASK.f_min
            if ok.any():
                rows.append(dict(method=m, table=tab, file=f.split("/")[-1], mean_u2_cm=100 * S[ok, 2].mean(),
                                 min_u2_cm=100 * S[ok, 2].min(), max_u2_cm=100 * S[ok, 2].max()))
t = pd.DataFrame(rows)
t.to_csv("results/tables/p4_tilt28_u2.csv", index=False)
print(t.groupby(["method", "table"])[["mean_u2_cm", "min_u2_cm", "max_u2_cm"]].mean().round(1).to_string())
print("saved results/figures/p4_tilt28_paths.png")
