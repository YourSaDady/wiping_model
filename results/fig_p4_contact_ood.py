"""P4: where the 1-D model breaks.

(i) Does each label make contact? Fraction of test episodes with any contact, with f_n ever
    ≥ f_min during wiping, and the time to first contact relative to the expert on the same
    scene (training tables, K_n = 500).
(ii) Success per label on the OOD tables, steeper-tilt tables highlighted.

Usage: python results/fig_p4_contact_ood.py [--split test]
Outputs: results/figures/p4_contact_ood.png, results/tables/p4_contact.csv, results/tables/p4_ood.csv
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
ap.add_argument("--K", type=int, default=500)
args = ap.parse_args()
df = load(args.split)
df = df[df.K_n == args.K]
exp = df[df.family == "scripted"][["table", "index", "first_contact_t"]].rename(columns={"first_contact_t": "t_exp"})
d = df.merge(exp, on=["table", "index"], how="left")
d["contact_delay"] = d.first_contact_t - d.t_exp
d["key"] = np.where(d.family == "dp", d.label, d.family)
idd = d[~d.ood]
cont = idd.groupby("key").agg(ever_contact=("ever_contact", "mean"),
                              pressed=("frac_wipe_ok", lambda s: (s > 0).mean()),
                              delay_med=("contact_delay", "median"),
                              delay_p90=("contact_delay", lambda s: s.quantile(0.9)),
                              n=("ever_contact", "size")).reset_index()
cont.to_csv("results/tables/p4_contact.csv", index=False)
print(cont.round(3).to_string(index=False))
ood = success_table(d[d.ood], ["key", "table", "tilt", "k_e"])
ood.to_csv("results/tables/p4_ood.csv", index=False)
print(ood.round(3).to_string(index=False))

keys = LABEL_ORDER + ["scripted", "hybrid"]
fig, ax = plt.subplots(1, 3, figsize=(14, 4.2), gridspec_kw={"width_ratios": [1, 1, 1.6]})
c = cont.set_index("key").reindex(keys)
x = np.arange(len(keys))
ax[0].bar(x - 0.2, c.ever_contact, 0.4, color=[COLORS[k] for k in keys], label="any contact")
ax[0].bar(x + 0.2, c.pressed, 0.4, color=[COLORS[k] for k in keys], alpha=0.45, hatch="//",
          label="f_n ≥ f_min while wiping")
ax[0].set_xticks(x, [LABEL_SHORT.get(k, k) for k in keys], rotation=20)
ax[0].set_ylim(0, 1.05); ax[0].set_title("Reaching the table (ID test scenes)")
ax[0].legend(fontsize=7)
for i, k in enumerate(keys):
    s = idd[idd.key == k].contact_delay.dropna()
    if len(s):
        ax[1].boxplot(s.values, positions=[i], widths=0.6, showfliers=False)
ax[1].set_xticks(x, [LABEL_SHORT.get(k, k) for k in keys], rotation=20)
ax[1].axhline(0, color="gray", lw=0.6)
ax[1].set_ylabel("first contact − expert's first contact [s]")
ax[1].set_title("Contact delay (episodes that touch)")
tables = sorted(d[d.ood].table.unique(), key=lambda t: (-("tilt28" in t), t))
w = 0.8 / len(keys)
for j, k in enumerate(keys):
    s = ood[ood.key == k].set_index("table").reindex(tables)
    xs = np.arange(len(tables)) + (j - len(keys) / 2 + 0.5) * w
    ax[2].bar(xs, s.rate, w, color=COLORS[k], label=LABEL_SHORT.get(k, k))
    ax[2].errorbar(xs, s.rate, yerr=[s.rate - s.lo, s.hi - s.rate], fmt="none", ecolor="k", lw=0.6, capsize=1.5)
ax[2].set_xticks(np.arange(len(tables)), [t.replace("_", "\n") for t in tables], fontsize=8)
ax[2].set_ylim(0, 1.05); ax[2].set_ylabel("success (Wilson 95% CI)")
ax[2].set_title("OOD tables (steeper 28° tilts first)")
ax[2].legend(fontsize=7, ncol=3)
fig.tight_layout()
fig.savefig("results/figures/p4_contact_ood.png", dpi=150)
print("saved results/figures/p4_contact_ood.png")
