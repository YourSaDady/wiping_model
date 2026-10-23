"""P3: target errors cost more force on stiffer surfaces (force error ≈ K_eff δ).

Left: closed-loop spread of the normal force during wiping (mean within-episode std of f_n)
against K_eff = K k_e/(K + k_e), per label, all K_n (marker) and tables (training + OOD).
Right: the same spread against the prediction K_eff · σ_δ, where σ_δ is the policy's
target error along n on held-out validation demos (scripts/action_spread.py, sweep phase).

Usage: python results/fig_p3_force_spread.py [--split test]
Outputs: results/figures/p3_force_spread.png, results/tables/p3_force_spread.csv
"""
import argparse
import json
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import COLORS, LABEL_ORDER, LABEL_SHORT, load

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="test")
args = ap.parse_args()
df = load(args.split)
# only episodes that actually wiped (force spread around f_des is meaningless for a tool in the air)
d = df[df.mean_fn_wipe > 2.0]
tab = d.groupby(["family", "label", "K_n", "table", "k_e", "tilt", "K_eff"], dropna=False).agg(
    sd=("std_fn_wipe", "mean"), sd_sd=("std_fn_wipe", "std"), n=("std_fn_wipe", "size"),
    fn=("mean_fn_wipe", "mean")).reset_index()
spread = {}
for m in df[df.family == "dp"].method.unique():
    p = os.path.join("runs", m, "action_spread.json")
    if os.path.exists(p):
        spread[m] = json.load(open(p))["std_sweep_mm"] / 1e3
sig = df[df.family == "dp"].drop_duplicates("method")[["method", "label", "K_n"]].copy()
sig["sigma_delta"] = sig.method.map(spread)
sig = sig.groupby(["label", "K_n"]).sigma_delta.mean().reset_index()
tab = tab.merge(sig, on=["label", "K_n"], how="left")
tab["pred_sd"] = tab.K_eff * tab.sigma_delta
tab.to_csv("results/tables/p3_force_spread.csv", index=False)
print(tab.round(4).to_string(index=False))

fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
mk = {250: "v", 500: "o", 1000: "^"}
for lab in LABEL_ORDER + ["scripted", "hybrid"]:
    s = tab[(tab.label == lab)] if lab in LABEL_ORDER else tab[tab.family == lab]
    for K, m in mk.items():
        ss = s[s.K_n == K]
        if ss.empty:
            continue
        ax[0].scatter(ss.K_eff, ss.sd, marker=m, s=26, color=COLORS[lab], alpha=0.85,
                      label=(LABEL_SHORT.get(lab, lab) if K == 500 else None))
        if lab in LABEL_ORDER:
            ax[1].scatter(ss.pred_sd, ss.sd, marker=m, s=26, color=COLORS[lab], alpha=0.85,
                          label=(LABEL_SHORT[lab] if K == 500 else None))
    if lab in LABEL_ORDER and len(s) > 2:
        c = np.polyfit(np.log(s.K_eff), s.sd, 1)
        xx = np.logspace(np.log10(s.K_eff.min()), np.log10(s.K_eff.max()), 20)
        ax[0].plot(xx, np.polyval(c, np.log(xx)), color=COLORS[lab], lw=0.8, alpha=0.6)
ax[0].set_xscale("log")
ax[0].set_xlabel("K_eff = K k_e / (K + k_e) [N/m]")
ax[0].set_ylabel("std of f_n during wiping [N]")
ax[0].set_title("P3: force spread vs K_eff (markers: K_n 250 ▽, 500 ○, 1000 △)")
ax[0].legend(fontsize=7)
lim = [0, max(1e-3, np.nanmax(tab.pred_sd.values) if tab.pred_sd.notna().any() else 1, tab.sd.max()) * 1.05]
ax[1].plot(lim, lim, "k--", lw=1, label="y = x")
ax[1].set_xlabel("predicted K_eff · σ_δ [N] (σ_δ from validation demos)")
ax[1].set_ylabel("measured std of f_n [N]")
ax[1].set_title("Measured vs predicted force spread")
ax[1].legend(fontsize=7)
fig.tight_layout()
fig.savefig("results/figures/p3_force_spread.png", dpi=150)
print("saved results/figures/p3_force_spread.png")
