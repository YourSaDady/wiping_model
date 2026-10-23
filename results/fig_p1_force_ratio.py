"""P1: label (a) keeps only K/(K + k_e) of the demonstrated force.

Left: mean normal force during wiping of each DP policy divided by the scripted expert's on
the same test scene (paired, same K_n), averaged per table, against K/(K + k_e). All K_n
values and all tables (training + OOD). Right: success rate vs k_e for each label at
K_n = 500 (lines: training tables pooled over tilts, Wilson 95 % CI; hollow diamonds: OOD
tables; dashed: baselines); the dotted line marks k_e = K_n (f_des K/(K+k_e) = f_min).

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
        ax[0].scatter(s.pred_a, np.maximum(s.ratio, 1e-3), marker=m, s=28, color=COLORS[lab], alpha=0.85,
                      edgecolors="k" if K == 500 else "none", linewidths=0.4,
                      label=f"{LABEL_SHORT[lab]} K_n={K}" if lab in ("a", "b") else None)
xx = np.logspace(-3.6, 0, 50)
ax[0].plot(xx, xx, "k--", lw=1, label="K/(K+k_e)")
ax[0].axhline(0.5, color="gray", lw=0.6, ls=":")
ax[0].text(1.5e-3, 0.53, "f_min / f_des", fontsize=7, color="gray")
ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_ylim(8e-4, 2)
ax[0].text(1.2e-3, 1.1e-3, "ratios < 1e-3 drawn at 1e-3", fontsize=7, color="gray")
ax[0].set_xlabel("predicted force ratio for (a): K_n / (K_n + k_e)")
ax[0].set_ylabel("policy / expert mean wiping force")
ax[0].set_title("P1: closed-loop force kept (all tables, all K_n)")
ax[0].legend(fontsize=7, ncol=2)

s5 = df[(df.K_n == 500)].copy()
s5["key"] = np.where(s5.family == "dp", s5.label, s5.family)
st = success_table(s5, ["key", "k_e", "ood", "table"])
for key in LABEL_ORDER + ["scripted", "hybrid"]:
    j = (LABEL_ORDER + ["scripted", "hybrid"]).index(key)
    idt = success_table(s5[(s5.key == key) & (~s5.ood)], ["k_e"]).sort_values("k_e")
    if idt.empty:
        continue
    dp = key in LABEL_ORDER
    ax[1].errorbar(idt.k_e * (1 + 0.04 * j), idt.rate, yerr=[idt.rate - idt.lo, idt.hi - idt.rate],
                   marker="o" if dp else None, ms=4, capsize=2, color=COLORS[key], ls="-" if dp else "--",
                   lw=1.5 if dp else 1, label=LABEL_SHORT.get(key, key) + " (training tables)")
    o = st[(st.key == key) & (st.ood)]
    ax[1].scatter(o.k_e * (1 + 0.04 * j), o.rate, marker="D", s=22, facecolors="none", edgecolors=COLORS[key],
                  label=None)
ax[1].scatter([], [], marker="D", facecolors="none", edgecolors="k", label="OOD tables (hollow)")
ax[1].axvline(500, color="gray", ls=":", lw=1)
ax[1].set_xscale("log"); ax[1].set_ylim(-0.03, 1.03)
ax[1].set_xlabel("table stiffness k_e [N/m]"); ax[1].set_ylabel("success rate (Wilson 95% CI)")
ax[1].set_title("Success vs k_e (K_n = 500; ID pooled over tilts 0/10/20°)")
ax[1].legend(fontsize=7)
fig.tight_layout()
fig.savefig("results/figures/p1_force_ratio.png", dpi=150)
print("saved results/figures/p1_force_ratio.png")
