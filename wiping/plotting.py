"""Matplotlib summaries of one logged episode."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from .dataset import make_labels
from .expert import PHASE_NAMES
from .lanes import plan_lanes

PHASE_COLORS = ["#d9d9d9", "#fdd49e", "#c7e9c0", "#c6dbef", "#ffffff"]


def _shade_phases(ax, t, phase):
    if len(phase) == 0 or phase.max() < 0:
        return
    k0 = 0
    for k in range(1, len(phase) + 1):
        if k == len(phase) or phase[k] != phase[k0]:
            p = int(phase[k0])
            ax.axvspan(t[k0], t[min(k, len(t) - 1)], color=PHASE_COLORS[p], alpha=0.6, lw=0,
                       label=PHASE_NAMES[p])
            k0 = k


def plot_episode(ep: dict, title: str | None = None, path=None):
    cfg = ep["cfg"]
    tab = cfg.table
    N = len(ep["x_d"])
    t = ep["t"]
    fig, axs = plt.subplots(2, 2, figsize=(12, 7.5))
    m = ep["metrics"]
    fig.suptitle(title or (f"{cfg.name}: k_e={tab.k_e:.0f} N/m, tilt={np.degrees(tab.tilt):.0f} deg, mu={tab.mu:.2f}"
                           f"  |  success={m['success']}, u_T={m['u_T']:.3f}, max f_n={m['fn_max']:.1f} N"))

    # (1) normal force
    ax = axs[0, 0]
    _shade_phases(ax, t[:N], ep["phase"])
    ax.plot(t, ep["f_n"], "k", lw=1.2, label="f_n")
    ax.axhline(cfg.expert.f_des, color="g", ls="--", lw=1, label="f_des")
    ax.axhline(cfg.f_min(), color="r", ls=":", lw=1, label="f_min")
    ax.axhline(cfg.f_max(), color="r", ls="--", lw=1, label="f_max")
    ax.set(xlabel="t (s)", ylabel="normal force (N)", title="normal force")
    h, lab = ax.get_legend_handles_labels()
    uniq = dict(zip(lab, h))
    ax.legend(uniq.values(), uniq.keys(), fontsize=7, ncol=3, loc="upper right")

    # (2) height along n: tool, commanded target, labels
    ax = axs[0, 1]
    w = tab.to_table(ep["x"])[:, 2]
    labels = make_labels(ep)
    ax.plot(t, 1e3 * w, "k", lw=1.2, label="tcp x (a)")
    ax.step(t[:N], 1e3 * tab.to_table(ep["x_d"])[:, 2], "b", where="post", lw=1, label="x_d (b)")
    ax.plot(t[1:], 1e3 * tab.to_table(labels["label_c_force_comp"])[:, 2], "m--", lw=1, label="x - K^-1 f (c)")
    ax.plot(t[1:], 1e3 * tab.to_table(labels["label_c_damp"])[:, 2], "c:", lw=1, label="+ K^-1 D xdot (c')")
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set(xlabel="t (s)", ylabel="height above surface along n (mm)", title="normal direction: tool vs. targets")
    ax.legend(fontsize=7)

    # (3) top view in table coordinates
    ax = axs[1, 0]
    reg = cfg.region
    c = reg.corner_uv()
    img = ep["clean_image"]
    dirt = ListedColormap(["#8c5a2b", "#ffffff"])
    ax.imshow(img.T.astype(float), origin="lower", extent=[c[0], c[0] + reg.L, c[1], c[1] + reg.L], cmap=dirt,
              vmin=0, vmax=1, alpha=0.9)
    wp = plan_lanes(reg, cfg.expert)
    ax.plot(wp[:, 0], wp[:, 1], "b-", lw=0.8, alpha=0.6, label="planned lanes")
    uv = tab.to_table(ep["x"])
    good = ep["f_n"] >= cfg.f_min()
    ax.plot(uv[:, 0], uv[:, 1], color="0.4", lw=0.6, label="tcp")
    ax.scatter(uv[good, 0], uv[good, 1], s=3, c="k", label="f_n >= f_min")
    ax.set_aspect("equal")
    ax.set(xlabel="u (m)", ylabel="v (m)", title="top view (table frame): white = cleaned")
    ax.legend(fontsize=7, loc="upper right")

    # (4) uncovered fraction + tangential speed
    ax = axs[1, 1]
    ax.plot(t, ep["u"], "k", label="uncovered fraction u_t")
    ax.axhline(cfg.success.eps_u, color="r", ls="--", lw=1, label="eps_u")
    vt = ep["xdot"] - np.outer(ep["xdot"] @ tab.normal(), tab.normal())
    ax2 = ax.twinx()
    ax2.plot(t, np.linalg.norm(vt, axis=1), color="tab:blue", lw=0.8, alpha=0.7)
    ax2.axhline(cfg.expert.speed, color="tab:blue", ls=":", lw=0.8)
    ax2.set_ylabel("tangential speed (m/s)", color="tab:blue")
    ax.set(xlabel="t (s)", ylabel="u_t", title="coverage", ylim=(-0.02, 1.02))
    ax.legend(fontsize=7, loc="center right")

    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=120)
    return fig
