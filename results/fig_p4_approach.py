"""P4 detail: how each label approaches the table.

Height of the ball bottom above the undeformed surface (100 Hz samples) vs time, for the
first 3 test scenes of every training table (K_n = 500): seed-0 policy of each label, the
scripted expert and the hybrid baseline. Thin lines: episodes; thick: median.

Usage: python results/fig_p4_approach.py
Output: results/figures/p4_approach.png, results/tables/p4_approach_speed.csv
"""
import glob
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import COLORS, LABEL_ORDER, LABEL_SHORT
from wipe.config import TRAIN_TABLES

SRC = {**{l: f"results/raw/K500_{l}_s0_test_traces" for l in LABEL_ORDER},
       "scripted": "results/raw/scripted_K500_test_traces", "hybrid": "results/raw/hybrid_K500_test_traces"}
train = {t.name for t in TRAIN_TABLES}
fig, axes = plt.subplots(1, 6, figsize=(17, 3.3), sharey=True)
rows = []
tgrid = np.arange(0, 50, 0.05)
for ax, (key, d) in zip(axes, SRC.items()):
    files = [f for f in sorted(glob.glob(os.path.join(d, "*.npz"))) if os.path.basename(f)[:-8] in train]
    curves = []
    for f in files:
        S = np.load(f)["samples"]
        g = np.interp(tgrid, S[:, 0], S[:, 3], right=np.nan) * 1e3
        curves.append(g)
        ax.plot(tgrid, g, color=COLORS[key], lw=0.4, alpha=0.35)
        # mean approach speed while the ball is 30 -> 5 mm above the surface
        i30 = np.argmax(S[:, 3] < 0.030); i5 = np.argmax(S[:, 3] < 0.005)
        if S[i5, 3] < 0.005 and i5 > i30:
            rows.append(dict(method=key, file=os.path.basename(f),
                             speed_mm_s=(25.0 / (S[i5, 0] - S[i30, 0])), t_5mm=S[i5, 0]))
    if curves:
        ax.plot(tgrid, np.nanmedian(np.array(curves), 0), color=COLORS[key], lw=2.2)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_title(f"{LABEL_SHORT.get(key, key)}  (n={len(files)})")
    ax.set_xlabel("time [s]"); ax.set_xlim(0, 50)
    ax.set_yscale("symlog", linthresh=1.0)
axes[0].set_ylabel("ball height above surface [mm]")
fig.suptitle("Approach to the table (first 3 test scenes per training table, K_n = 500, seed 0)", y=1.02)
fig.tight_layout()
fig.savefig("results/figures/p4_approach.png", dpi=150, bbox_inches="tight")
tab = pd.DataFrame(rows)
if len(tab):
    summ = tab.groupby("method").agg(speed_mm_s=("speed_mm_s", "median"), t_5mm=("t_5mm", "median"),
                                     n=("speed_mm_s", "size")).reset_index()
    summ.to_csv("results/tables/p4_approach_speed.csv", index=False)
    print(summ.round(2).to_string(index=False))
print("saved results/figures/p4_approach.png")
