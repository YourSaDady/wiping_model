"""Stretch goals (Phase 6).

1. Pose+force label `pf` (trained at K_n = 500, 3 seeds), its targets rebuilt at deployment with the
   controller's own K′ (x_d = x̂ + K′⁻¹(D′ x̂̇ − f̂)), deployed at K′ = 250, 500, 1000; compared with
   (b) and (c′) trained at 500 and deployed at the mismatched K′ (seed 0), and with (b)/(c′) trained
   natively at K′ (seed 0).
2. Label (c) with corrupted force inside the label (seed 0): noise σ = 1, 3 N; bias ±2 N along n.
   Prediction for a bias b: target shifts b/K along n → force ≈ f + b·K_eff/K.

Usage: python results/stretch_analysis.py
Outputs: results/tables/stretch_pf.csv, results/tables/stretch_noise.csv, results/figures/stretch.png
"""
import json
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

from common import COLORS, k_eff
from wipe.stats import wilson


def rows(name):
    p = f"results/raw/{name}_test.jsonl"
    if not os.path.exists(p):
        return None
    d = pd.DataFrame([json.loads(l) for l in open(p)])
    return d


def summarize(d, **meta):
    out = []
    for dist, s in (("ID", d[~d.ood]), ("OOD", d[d.ood])):
        k, n = int(s.success.sum()), len(s)
        lo, hi = wilson(k, n)
        rate = k / n if n else np.nan
        lo, hi = min(lo, rate), max(hi, rate)            # guard float round-off at 0 and 1
        out.append(dict(**meta, dist=dist, k=k, n=n, rate=rate, lo=lo, hi=hi,
                        fn=s.mean_fn_wipe.mean(), peak_fail=int((s.peak_fn > 20).sum()),
                        uncovered=s.uncovered.mean()))
    return out


res = []
for Kp in (250, 500, 1000):
    for s in (0, 1, 2):
        d = rows(f"K500_pf_s{s}" + ("" if Kp == 500 else f"_dK{Kp}"))
        if d is not None:
            res += summarize(d, policy="pf (rebuilt at K')", trained_K=500, deploy_K=Kp, seed=s)
    for l in ("b", "cp"):
        d = rows(f"K500_{l}_s0" + ("" if Kp == 500 else f"_dK{Kp}"))
        if d is not None:
            res += summarize(d, policy=f"{l} trained at 500", trained_K=500, deploy_K=Kp, seed=0)
        d = rows(f"K{Kp}_{l}_s0")
        if d is not None:
            res += summarize(d, policy=f"{l} trained at K'", trained_K=Kp, deploy_K=Kp, seed=0)
pf = pd.DataFrame(res)
if len(pf):
    pf.to_csv("results/tables/stretch_pf.csv", index=False)
    piv = pf.groupby(["policy", "deploy_K", "dist"]).agg(k=("k", "sum"), n=("n", "sum"), fn=("fn", "mean"),
                                                        peak_fail=("peak_fail", "sum")).reset_index()
    piv["rate"] = piv.k / piv.n
    print(piv.round(3).to_string(index=False))

res = []
clean = rows("K500_c_s0")
if clean is not None:
    res += summarize(clean, variant="clean", sigma=0.0, bias=0.0)
for tag, sg, b in (("_fn1", 1.0, 0.0), ("_fn3", 3.0, 0.0), ("_fbp2", 0.0, 2.0), ("_fbm2", 0.0, -2.0)):
    d = rows(f"K500_c_s0{tag}")
    if d is not None:
        res += summarize(d, variant=tag[1:], sigma=sg, bias=b)
        # force shift vs clean on the same scenes (paired, ID tables, both in contact)
        if clean is not None:
            m = d.merge(clean, on=["table", "index"], suffixes=("", "_clean"))
            m = m[(~m.ood) & (m.mean_fn_wipe > 2) & (m.mean_fn_wipe_clean > 2)]
            ke = m.k_e.values
            pred = b * k_eff(500, ke) / 500
            res[-2]["fn_shift"] = float((m.mean_fn_wipe - m.mean_fn_wipe_clean).mean()) if len(m) else np.nan
            res[-2]["fn_shift_pred"] = float(pred.mean()) if len(m) else np.nan
nz = pd.DataFrame(res)
if len(nz):
    nz.to_csv("results/tables/stretch_noise.csv", index=False)
    print(nz.round(3).to_string(index=False))

if len(pf):
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2), gridspec_kw={"width_ratios": [1, 1, 1.1]})
    for pol, c, ls in (("pf (rebuilt at K')", "#7a4fa3", "-"), ("b trained at 500", COLORS["b"], "--"),
                       ("b trained at K'", COLORS["b"], "-"), ("cp trained at 500", COLORS["cp"], "--"),
                       ("cp trained at K'", COLORS["cp"], "-")):
        s = piv[(piv.policy == pol) & (piv.dist == "ID")].sort_values("deploy_K")
        if len(s):
            ax[0].plot(s.deploy_K, s.rate, "o" + ls, color=c, label=pol)
            ax[1].plot(s.deploy_K, s.fn, "o" + ls, color=c, label=pol)
    for a in ax[:2]:
        a.set_xscale("log"); a.set_xticks([250, 500, 1000], [250, 500, 1000]); a.set_xlabel("deployment K'_n [N/m]")
        a.xaxis.set_minor_formatter(NullFormatter())
    ax[0].set_ylabel("ID success"); ax[0].set_ylim(-0.03, 1.03); ax[0].legend(fontsize=7)
    ax[1].set_ylabel("mean f_n while wiping [N]"); ax[1].axhline(10, color="gray", lw=0.6, ls=":")
    if len(nz):
        idn = nz[nz.dist == "ID"].set_index("variant").reindex(["clean", "fn1", "fn3", "fbp2", "fbm2"])
        names = ["clean", "noise\nσ=1 N", "noise\nσ=3 N", "bias\n+2 N", "bias\n−2 N"]
        xx = np.arange(len(idn))
        ax[2].bar(xx, idn.rate, 0.6, color=COLORS["c"], edgecolor="k", lw=0.5)
        ax[2].errorbar(xx, idn.rate, yerr=[idn.rate - idn.lo, idn.hi - idn.rate], fmt="none", ecolor="k", capsize=2)
        for xi, f in zip(xx, idn.fn):
            ax[2].text(xi, 0.01, f"{f:.1f} N", ha="center", va="bottom", fontsize=7, color="k")
        ax[2].set_xticks(xx, names, fontsize=8)
        ax[2].set_ylabel("ID success of label (c), seed 0 (text: mean f_n)")
        ax[2].set_title("Force noise / bias inside label (c)")
    fig.suptitle("Stretch goals (policies trained on K_n = 500 demos unless noted)")
    fig.tight_layout()
    fig.savefig("results/figures/stretch.png", dpi=150)
    print("saved results/figures/stretch.png")
