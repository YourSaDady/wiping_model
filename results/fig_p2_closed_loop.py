"""P2 in closed loop, and the controller-stiffness sweep.

(a) Success per label on training-table (ID) and OOD test scenes, K_n = 500 (3 seeds pooled),
    with the two baselines; Wilson 95 % CI.
(b) How strongly each policy feeds its observed velocity into the commanded target along n
    (G_n, results/p2_velocity_gain.py) vs the controller's D_n/K_n: the remaining normal
    damping is D_n − K_n G_n.
(c) ID success vs controller stiffness K_n per label (seed 0 at every K_n, for comparability).

Usage: python results/fig_p2_closed_loop.py
Outputs: results/figures/p2_closed_loop.png, results/tables/ksweep_success.csv
"""
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

from common import COLORS, LABEL_ORDER, LABEL_SHORT, load, success_table

df = load("test")
df["key"] = np.where(df.family == "dp", df.label, df.family)
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2), gridspec_kw={"width_ratios": [1.3, 1, 1]})

k5 = df[df.K_n == 500]
keys = ["scripted", "hybrid"] + LABEL_ORDER
st = success_table(k5, ["key", "ood"])
x = np.arange(len(keys))
for j, ood in enumerate((False, True)):
    s = st[st.ood == ood].set_index("key").reindex(keys)
    xs = x + (j - 0.5) * 0.38
    ax[0].bar(xs, s.rate, 0.38, color=[COLORS[k] for k in keys], alpha=1.0 if not ood else 0.45,
              hatch=None if not ood else "//", edgecolor="k", lw=0.4)
    ax[0].errorbar(xs, s.rate, yerr=[s.rate - s.lo, s.hi - s.rate], fmt="none", ecolor="k", lw=0.8, capsize=2)
ax[0].set_xticks(x, [LABEL_SHORT.get(k, k) for k in keys])
ax[0].set_ylabel("success rate (Wilson 95% CI)")
ax[0].set_title("K_n = 500: solid = training tables (ID), hatched = OOD")
ax[0].set_ylim(0, 1.05)

vg = pd.read_csv("results/tables/p2_velocity_gain.csv")
vg = vg[vg.dir == "n"].groupby(["label", "K_n"]).G_s.median().reset_index()
for i, lab in enumerate(LABEL_ORDER):
    s = vg[vg.label == lab].sort_values("K_n")
    ax[1].plot(s.K_n, s.G_s, "o-", color=COLORS[lab], label=LABEL_SHORT[lab])
KK = np.array([250, 500, 1000])
ax[1].plot(KK, 250 / KK, "k--", lw=1, label="D_n / K_n (cancels all damping)")
ax[1].set_xscale("log"); ax[1].set_xticks(KK, KK); ax[1].xaxis.set_minor_formatter(NullFormatter())
ax[1].set_xlabel("controller stiffness K_n [N/m]")
ax[1].set_ylabel("G_n = d(target lead along n) / d(observed v_n) [s]")
ax[1].set_title("Velocity fed into the target (validation sweeps)")
ax[1].legend(fontsize=7)

s0 = df[(df.family == "dp") & (df.seed == 0) & (~df.ood)]
ks = success_table(s0, ["label", "K_n"])
ks.to_csv("results/tables/ksweep_success.csv", index=False)
ko = success_table(df[(df.family == "dp") & (df.seed == 0) & (df.ood)], ["label", "K_n"])
ko.to_csv("results/tables/ksweep_success_ood.csv", index=False)
for lab in LABEL_ORDER:
    s = ks[ks.label == lab].sort_values("K_n")
    ax[2].errorbar(s.K_n * (1 + 0.03 * LABEL_ORDER.index(lab)), s.rate, yerr=[s.rate - s.lo, s.hi - s.rate],
                   marker="o", capsize=2, color=COLORS[lab], label=LABEL_SHORT[lab])
ax[2].set_xscale("log"); ax[2].set_xticks(KK, KK); ax[2].xaxis.set_minor_formatter(NullFormatter())
ax[2].set_ylim(-0.03, 1.05)
ax[2].set_xlabel("controller stiffness K_n [N/m] (D_n = 250 fixed)")
ax[2].set_ylabel("ID success (seed 0, 600 scenes)")
ax[2].set_title("Controller-stiffness sweep")
ax[2].legend(fontsize=7)
fig.tight_layout()
fig.savefig("results/figures/p2_closed_loop.png", dpi=150)
print(ks[["label", "K_n", "k", "n", "rate", "lo", "hi"]].round(3).to_string(index=False))
print(ko[["label", "K_n", "k", "n", "rate"]].round(3).to_string(index=False))
print("saved results/figures/p2_closed_loop.png")
