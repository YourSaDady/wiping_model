"""P2 offline check: how far labels (c) and (c') are from the commanded target (b).

P2 (proposal): label (c) = x - K^-1 f ignores damping and inertia, so it should match (b)
during steady wiping but deviate at contact onset, at lift-off and at high speed v; adding
K^-1 D xdot (label c') should close most of the gap. On the demonstrations we compute, per
step, the gap g = label - x_d (= label - b) and report

  |g|         norm [mm]
  g_n         component along the table normal [mm] (signed; < 0 = label deeper than x_d)
  |g_t|       in-plane norm [mm]
  K_n |g_n|   the normal force error [N] the gap would cause against a rigid table
              (against a table of stiffness k_e it is K_eff |g_n|, K_eff = K_n k_e / (K_n + k_e))

grouped by expert phase, by event window (0.5 s after contact onset, 0.5 s after lift-off
starts), by in-plane speed during the sweep, and by table. Also reported: label (a) (the P1
offset) and the proposal's same-index variant c(t) = x_t - K^-1 f_t.

Reads the CSV dataset (scripts/export_dataset.py), no simulation.
Usage: python scripts/p2_offline_check.py [--data data/hf/wipe-labels] [--K 500] [--splits train val test]
Writes results/p2/*.csv and results/figures/p2_offline_gap.png.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from diffusion_policy.dataset import LABEL_COLS, PHASES, load_episodes, load_steps, resolve_data
from wipe.config import GAINS_BY_KN
from wipe.data import DATA_ROOT

ONSET_STEPS = 10          # 0.5 s event windows at 20 Hz
VARIANTS = ("a", "c", "c(t)", "cp")


def gaps(df, K):
    """Adds gap columns for every variant to df (one row per step)."""
    g = GAINS_BY_KN[K]
    n = df[["n_x", "n_y", "n_z"]].to_numpy()
    x = df[["x_x", "x_y", "x_z"]].to_numpy()
    f = df[["f_x", "f_y", "f_z"]].to_numpy()
    b = df[LABEL_COLS["b"]].to_numpy()
    # K^-1 f in the table frame: f_n / K_n along n, the rest / K_t
    fn = (f * n).sum(1, keepdims=True)
    kinv_f = (f - fn * n) / g.K_t + fn * n / g.K_n
    lab = {"a": df[LABEL_COLS["a"]].to_numpy(), "c": df[LABEL_COLS["c"]].to_numpy(),
           "c(t)": x - kinv_f, "cp": df[LABEL_COLS["cp"]].to_numpy()}
    out = {}
    for v, l in lab.items():
        d = l - b
        dn = (d * n).sum(1)
        out[f"{v}|norm"] = 1e3 * np.linalg.norm(d, axis=1)
        out[f"{v}|n"] = 1e3 * dn
        out[f"{v}|t"] = 1e3 * np.linalg.norm(d - dn[:, None] * n, axis=1)
        out[f"{v}|force"] = g.K_n * np.abs(dn)
    xd = df[["xdot_x", "xdot_y", "xdot_z"]].to_numpy()
    out["speed_t"] = np.linalg.norm(xd - (xd * n).sum(1, keepdims=True) * n, axis=1)
    return df.assign(**out)


def event_masks(df):
    """Onset: the 0.5 s after the first step with f_n > 1 N; lift-off: the first 0.5 s of the lift phase."""
    onset = np.zeros(len(df), bool)
    lift = np.zeros(len(df), bool)
    for _, idx in df.groupby("episode_id", sort=False).indices.items():
        fn = df.fn.to_numpy()[idx]
        ph = df.phase.to_numpy()[idx]
        c = np.flatnonzero(fn > 1.0)
        if len(c):
            onset[idx[c[0]:c[0] + ONSET_STEPS]] = True
        l = np.flatnonzero(ph == 3)
        if len(l):
            lift[idx[l[0]:l[0] + ONSET_STEPS]] = True
    return onset, lift


def summarize(df, groups):
    rows = []
    for name, m in groups:
        d = df[m]
        if not len(d):
            continue
        r = dict(group=name, steps=int(len(d)))
        for v in VARIANTS:
            r[f"{v} |g| mean mm"] = d[f"{v}|norm"].mean()
            r[f"{v} |g| p95 mm"] = d[f"{v}|norm"].quantile(0.95)
            r[f"{v} g_n mean mm"] = d[f"{v}|n"].mean()
            r[f"{v} |g_t| mean mm"] = d[f"{v}|t"].mean()
            r[f"{v} K_n|g_n| mean N"] = d[f"{v}|force"].mean()
        rows.append(r)
    return pd.DataFrame(rows)


def md(df, variants=("c", "cp", "c(t)", "a")):
    head = "| group | steps | " + " | ".join(f"{v}: \\|g\\| mean / p95 [mm], g_n [mm], K_n\\|g_n\\| [N]"
                                            for v in variants) + " |"
    L = [head, "|---|---|" + "---|" * len(variants)]
    for _, r in df.iterrows():
        L.append(f"| {r.group} | {r.steps:,} | " + " | ".join(
            f"{r[f'{v} |g| mean mm']:.2f} / {r[f'{v} |g| p95 mm']:.2f}, {r[f'{v} g_n mean mm']:+.2f}, "
            f"{r[f'{v} K_n|g_n| mean N']:.2f}" for v in variants) + " |")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(DATA_ROOT, "hf", "wipe-labels"))
    ap.add_argument("--K", type=int, nargs="+", default=[500])
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--out", default="results/p2")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    report = []
    for K in args.K:
        root = resolve_data(args.data, K)
        eps = load_episodes(root, K).set_index("episode_id")
        df = pd.concat([load_steps(root, K, s) for s in args.splits], ignore_index=True)
        df = gaps(df, K)
        df["k_e"] = eps.k_e.reindex(df.episode_id).to_numpy()
        df["tilt"] = eps.tilt_deg.reindex(df.episode_id).to_numpy()
        df["table"] = eps.table.reindex(df.episode_id).to_numpy()
        onset, lift = event_masks(df)
        ph = df.phase.to_numpy()
        by_phase = summarize(df, [(PHASES[p], ph == p) for p in PHASES] +
                             [("contact onset (0.5 s)", onset), ("lift-off (0.5 s)", lift),
                              ("all", np.ones(len(df), bool))])
        edges = [0.0, 0.03, 0.06, 0.09, 0.12, 1.0]
        sp = df.speed_t.to_numpy()
        by_speed = summarize(df, [(f"sweep, |v_t| {lo:.2f}-{hi:.2f} m/s" if hi < 1 else f"sweep, |v_t| > {lo:.2f} m/s",
                                   (ph == 2) & (sp >= lo) & (sp < hi)) for lo, hi in zip(edges[:-1], edges[1:])])
        by_table = summarize(df, [(f"{t} (sweep)", (df.table == t).to_numpy() & (ph == 2))
                                  for t in df.table.unique()] +
                             [(f"{t} (onset)", (df.table == t).to_numpy() & onset) for t in df.table.unique()])
        for name, t in (("by_phase", by_phase), ("by_speed", by_speed), ("by_table", by_table)):
            t.to_csv(os.path.join(args.out, f"K{K}_{name}.csv"), index=False, float_format="%.4g")
        report.append(f"### K_n = {K} ({', '.join(args.splits)}: {df.episode_id.nunique()} episodes, {len(df):,} steps)\n")
        report += ["By phase / event:\n", md(by_phase), "\nSweep, by in-plane speed:\n", md(by_speed), ""]

        if K == args.K[0]:
            fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
            labels = list(by_phase.group)
            xx = np.arange(len(labels))
            for i, (v, c) in enumerate((("c", "#d81b60"), ("cp", "#1e88e5"), ("c(t)", "#9e9e9e"))):
                ax[0].bar(xx + (i - 1) * 0.27, by_phase[f"{v} |g| mean mm"], 0.27, color=c, label=v)
                ax[1].bar(xx + (i - 1) * 0.27, by_phase[f"{v} K_n|g_n| mean N"], 0.27, color=c, label=v)
            for a, yl in zip(ax[:2], ("mean |label − x_d| [mm]", "K_n · mean |g_n| [N]")):
                a.set_xticks(xx, labels, rotation=35, ha="right", fontsize=8)
                a.set_ylabel(yl); a.legend(); a.grid(axis="y", alpha=0.3)
            mids = [0.015, 0.045, 0.075, 0.105, 0.13][:len(by_speed)]
            for v, c in (("c", "#d81b60"), ("cp", "#1e88e5")):
                ax[2].plot(mids, by_speed[f"{v} |g| mean mm"], "o-", color=c, label=f"{v} |g|")
                ax[2].plot(mids, by_speed[f"{v} |g_t| mean mm"], "s--", color=c, alpha=0.6, label=f"{v} |g_t|")
            ax[2].set_xlabel("in-plane speed during the sweep [m/s]"); ax[2].set_ylabel("[mm]")
            ax[2].legend(); ax[2].grid(alpha=0.3)
            fig.suptitle(f"P2 offline check, K_n = {K}: gap of label (c), (c') and c(t) to the commanded target (b)")
            fig.tight_layout()
            os.makedirs("results/figures", exist_ok=True)
            fig.savefig("results/figures/p2_offline_gap.png", dpi=130)
    text = "\n".join(report)
    open(os.path.join(args.out, "p2_report.md"), "w").write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
