"""Figure: normal force vs time when each label of one demo is replayed open loop
(flat tables, K_n = 500, one scene per k_e). Data: results/phase1/replay_traces.npy
written by scripts/phase1_checks.py."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {"demo": "#222222", "a": "#d1495b", "b": "#2e86ab", "c": "#edae49", "cp": "#66a182"}
NAMES = {"demo": "expert demo", "a": "(a) x", "b": "(b) x_d", "c": "(c) x - K^-1 f", "cp": "(c') + K^-1 D xdot"}

tr = np.load("results/phase1/replay_traces.npy", allow_pickle=True)
sel = [(key, t) for key, t in tr if key[2] == 500 and key[0].startswith("tilt00") and key[1] == 0]
sel.sort(key=lambda kt: float(kt[0][0].split("ke")[1].replace("e", "e")))
fig, axes = plt.subplots(len(sel), 2, figsize=(11, 2.3 * len(sel)), sharex="col",
                         gridspec_kw={"width_ratios": [2.2, 1]})
for row, (key, t) in zip(axes, sel):
    ke = key[0].split("ke")[1]
    for lab in ["demo", "b", "c", "cp", "a"]:
        ts, fn = t[lab]
        ls = "--" if lab == "b" else "-"
        for ax in row:
            ax.plot(ts, fn, ls, color=COLORS[lab], lw=1.0 if lab != "demo" else 2.2, alpha=0.9,
                    label=NAMES[lab])
    t_on = t["demo"][0][np.argmax(t["demo"][1] > 0)]
    row[1].set_xlim(t_on - 0.5, t_on + 2.0)
    for ax in row:
        ax.axhline(10, color="gray", lw=0.5, ls=":"); ax.axhline(5, color="gray", lw=0.5, ls=":")
        ax.axhline(20, color="k", lw=0.5, ls=":")
    row[0].set_ylabel(f"k_e = {ke} N/m\nf_n [N]")
    row[1].set_title("contact onset (zoom)" if row is axes[0] else "", fontsize=9)
axes[0, 0].legend(ncol=5, fontsize=8, loc="upper left", bbox_to_anchor=(0, 1.35))
axes[-1, 0].set_xlabel("time [s]"); axes[-1, 1].set_xlabel("time [s]")
fig.suptitle("Open-loop replay of each label from the same demonstration (flat table, K_n = 500 N/m)", y=1.0)
fig.tight_layout()
fig.savefig("results/figures/phase1_replay_force.png", dpi=150, bbox_inches="tight")
print("saved results/figures/phase1_replay_force.png")
