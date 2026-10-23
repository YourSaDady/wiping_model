"""P1: label (a) keeps only K/(K + k_e) of the demonstrated force.

Left: mean normal force during wiping of each DP policy divided by the scripted expert's on
the same test scene (paired, same K_n), averaged per table, against K/(K + k_e). All K_n
values and all tables (training + OOD). Right: success rate vs k_e for each label at
K_n = 500 (training + OOD tables pooled by k_e, Wilson 95 % CI); the dotted line marks
k_e = K_n (f_des K/(K+k_e) = f_min).

Usage: python results/fig_p1_force_ratio.py [--split test]
Outputs: results/figures/p1_force_ratio.png, results/tables/p1_force_ratio.csv
"""
import argparse

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import COLORS, LABEL_ORDER, LABEL_SHORT, load, success_table

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="test")
args = ap.parse_args()
df = load(args.split)
exp = df[df.family == "scripted"][["K_n", "table", "index", "mean_fn_wipe"]].rename(
    columns={"mean_fn_wipe": "fn_expert"})
dp = df[df.family == "dp"].merge(exp, on=["K_n", "table", "index"], how="inner")
dp["ratio"] = dp.mean_fn_wipe / dp.fn_expert
tab = dp.groupby(["label", "K_n", "table", "k_e", "tilt", "ood"]).agg(
    ratio=("ratio", "mean"), ratio_sd=("ratio", "std"), n=("ratio", "size"),
    fn=("mean_fn_wipe", "mean"), fn_expert=("fn_expert", "mean"), success=("success", "mean")).reset_index()
tab["pred_a"] = tab.K_n / (tab.K_n + tab.k_e)
tab.to_csv("results/tables/p1_force_ratio.csv", index=False)
print(tab[tab.label == "a"].sort_values(["K_n", "k_e"]).round(4).to_string(index=False))

fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
mk = {250: "v", 500: "o", 1000: "^"}
for lab in LABEL_ORDER:
    for K, m in mk.items():
        s = tab[(tab.label == lab) & (tab.K_n == K)]
        if s.empty:
            continue
        ax[0].scatter(s.pred_a, s.ratio, marker=m, s=28, color=COLORS[lab], alpha=0.85,
                      edgecolors="k" if K == 500 else "none", linewidths=0.4,
                      label=f"{LABEL_SHORT[lab]} K_n={K}" if lab in ("a", "b") else None)
xx = np.logspace(-3.6, 0, 50)
ax[0].plot(xx, xx, "k--", lw=1, label="K/(K+k_e)")
ax[0].axhline(0.5, color="gray", lw=0.6, ls=":")
ax[0].text(1.5e-3, 0.53, "f_min / f_des", fontsize=7, color="gray")
ax[0].set_xscale("log"); ax[0].set_yscale("symlog", linthresh=1e-2)
ax[0].set_xlabel("predicted force ratio for (a): K_n / (K_n + k_e)")
ax[0].set_ylabel("policy / expert mean wiping force")
ax[0].set_title("P1: force kept by each label (all tables, all K_n)")
ax[0].legend(fontsize=7, ncol=2)

s5 = df[(df.K_n == 500) & (df.family == "dp")]
st = success_table(s5, ["label", "k_e"])
for lab in LABEL_ORDER:
    s = st[st.label == lab].sort_values("k_e")
    ax[1].errorbar(s.k_e * (1 + 0.04 * LABEL_ORDER.index(lab)), s.rate, yerr=[s.rate - s.lo, s.hi - s.rate],
                   marker="o", ms=4, capsize=2, color=COLORS[lab], label=LABEL_SHORT[lab])
for base in ("scripted", "hybrid"):
    s = success_table(df[(df.K_n == 500) & (df.family == base)], ["k_e"]).sort_values("k_e")
    if len(s):
        ax[1].plot(s.k_e, s.rate, ls="--", color=COLORS[base], lw=1, label=base)
ax[1].axvline(500, color="gray", ls=":", lw=1)
ax[1].set_xscale("log"); ax[1].set_ylim(-0.03, 1.03)
ax[1].set_xlabel("table stiffness k_e [N/m]"); ax[1].set_ylabel("success rate (Wilson 95% CI)")
ax[1].set_title("Success vs k_e (K_n = 500)")
ax[1].legend(fontsize=7)
fig.tight_layout()
fig.savefig("results/figures/p1_force_ratio.png", dpi=150)
print("saved results/figures/p1_force_ratio.png")
