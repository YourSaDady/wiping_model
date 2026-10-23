"""P3 with a steady-state force-spread metric (from the saved 100 Hz traces).

The per-episode `std_fn_wipe` in the evaluation metrics includes the contact-onset ramp and
the lift, which dominate it (≈ 2.7 N for every method). Here we keep only steady wiping:
in-region samples with the ball within 5 mm of the surface, from 1 s after f_n first reaches
0.8 f_des to 0.5 s before it last does. Two spreads per episode:
  sd     — std of f_n over those samples (includes slow variations, e.g. at lane corners);
  sd_hf  — std of f_n minus its 0.5 s moving average (fast fluctuations only).
Episodes: the first 3 test scenes of every table (traces), every method and K_n.

Usage: python results/p3_steady_spread.py
Outputs: results/tables/p3_steady_spread.csv, results/figures/p3_steady_spread.png
"""
import glob
import json
import os
import re

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import COLORS, LABEL_ORDER, LABEL_SHORT, k_eff
from wipe.config import TABLES, TASK

rows = []
for d in sorted(glob.glob("results/raw/*_test_traces")):
    name = os.path.basename(d)[:-len("_test_traces")]
    m = re.match(r"^K(\d+)_(a|b|cp|c)_s(\d+)$", name)
    mb = re.match(r"^(scripted|hybrid)_K(\d+)$", name)
    if m:
        K, key, seed = int(m.group(1)), m.group(2), int(m.group(3))
    elif mb:
        key, K, seed = mb.group(1), int(mb.group(2)), 0
    else:
        continue
    for f in sorted(glob.glob(os.path.join(d, "*.npz"))):
        z = np.load(f)
        S = z["samples"]
        sc = json.loads(str(z["scene"]))
        L, r = sc["L"], TASK.r_wipe
        t, u1, u2, gap, fn = S.T
        ok = fn >= 0.8 * TASK.f_des
        if ok.sum() < 200:
            continue
        t0, t1 = t[ok][0] + 1.0, t[ok][-1] - 0.5
        w = (t >= t0) & (t <= t1) & (u1 >= -r) & (u1 <= L + r) & (u2 >= -r) & (u2 <= L + r) & (gap <= 0.005)
        if w.sum() < 200:
            continue
        x = fn[w]
        ma = np.convolve(fn, np.ones(50) / 50, mode="same")[w]
        tab = TABLES[sc["table"]]
        rows.append(dict(method=name, key=key, K_n=K, seed=seed, table=tab.name, k_e=tab.k_e, tilt=tab.tilt_deg,
                         ood=tab.split == "ood", K_eff=k_eff(K, tab.k_e), mean=x.mean(), sd=x.std(),
                         sd_hf=(x - ma).std(), frac_low=(x < TASK.f_min).mean(), n=int(w.sum())))
df = pd.DataFrame(rows)
df.to_csv("results/tables/p3_steady_spread.csv", index=False)
summ = df.groupby(["key", "K_n", "k_e"]).agg(K_eff=("K_eff", "first"), sd=("sd", "mean"), sd_hf=("sd_hf", "mean"),
                                             mean=("mean", "mean"), n=("sd", "size")).reset_index()
print(summ.round(3).to_string(index=False))
# rank correlation of spread with K_eff per method family (training + OOD tables)
for key in LABEL_ORDER + ["scripted", "hybrid"]:
    s = df[df.key == key]
    if len(s) > 5:
        rho = s[["K_eff", "sd"]].corr(method="spearman").iloc[0, 1]
        rho_hf = s[["K_eff", "sd_hf"]].corr(method="spearman").iloc[0, 1]
        rk = s[["k_e", "sd"]].corr(method="spearman").iloc[0, 1]
        print(f"{key:9s} Spearman(K_eff, sd) = {rho:+.2f}   (K_eff, sd_hf) = {rho_hf:+.2f}   (k_e, sd) = {rk:+.2f}   n = {len(s)}")
        for K in (250, 500, 1000):
            sk = s[s.K_n == K]
            if len(sk) > 5:
                print(f"            K_n={K}: Spearman(K_eff, sd) = {sk[['K_eff', 'sd']].corr(method='spearman').iloc[0, 1]:+.2f}  n = {len(sk)}")

fig, ax = plt.subplots(2, 3, figsize=(14, 7), sharey="row")
for j, K in enumerate((250, 500, 1000)):
    for key in LABEL_ORDER + ["scripted", "hybrid"]:
        ss = summ[(summ.key == key) & (summ.K_n == K)].sort_values("k_e")
        if ss.empty:
            continue
        for i, col in enumerate(("sd", "sd_hf")):
            ax[i, j].plot(ss.k_e, ss[col], "o-", ms=4, lw=1.2, color=COLORS[key], label=LABEL_SHORT.get(key, key))
    ax[0, j].set_title(f"K_n = {K} N/m  (K_eff from {k_eff(K, 250):.0f} to {k_eff(K, 3e5):.0f})")
    for i in range(2):
        ax[i, j].set_xscale("log")
        ax[i, j].axvline(K, color="gray", lw=0.6, ls=":")
    ax[1, j].set_xlabel("table stiffness k_e [N/m]")
ax[0, 0].set_ylabel("steady-wiping std of f_n [N]")
ax[1, 0].set_ylabel("fast part (minus 0.5 s mov. avg) [N]")
ax[0, 1].legend(fontsize=7)
fig.suptitle("P3: steady-state force spread vs table stiffness (first 3 test scenes per table; dotted: k_e = K_n)",
             y=1.01)
fig.tight_layout()
fig.savefig("results/figures/p3_steady_spread.png", dpi=150, bbox_inches="tight")
print("saved results/figures/p3_steady_spread.png")
