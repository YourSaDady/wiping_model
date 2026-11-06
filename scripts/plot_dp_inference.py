"""Draws the Diffusion Policy inference diagram (README "Inference"): one policy call, then the
timeline of how its target chunks are executed by the impedance controller and the plant.
Usage: python scripts/plot_dp_inference.py  ->  results/figures/dp_inference.png
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

BLUE, BLUE_L = "#1565c0", "#e3f2fd"
PINK, PINK_L = "#c2185b", "#fce4ec"
GREEN, GREEN_L = "#2e7d32", "#e8f5e9"
ORANGE, ORANGE_L = "#e65100", "#fff3e0"
GREY, GREY_L = "#616161", "#f5f5f5"
INK = "#212121"
plt.rcParams.update({"font.size": 10, "mathtext.fontset": "dejavusans"})


def rbox(ax, x, y, w, h, ec, fc, lw=1.6, r=0.012):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", ec=ec, fc=fc, lw=lw))


def arrow(ax, p, q, color=INK, lw=1.6, rad=0.0, style="-|>"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=15, color=color, lw=lw,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))


def text(ax, x, y, s, **kw):
    kw.setdefault("ha", "center"); kw.setdefault("va", "center"); kw.setdefault("color", INK)
    ax.text(x, y, s, **kw)


def top_panel(ax):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    text(ax, 0.0, 0.97, "①  One policy call at step $t$   (repeated every 8 steps = 0.4 s)", ha="left",
         fontsize=13, weight="bold")
    y0, h = 0.04, 0.82
    # observation
    rbox(ax, 0.005, y0, 0.17, h, BLUE, BLUE_L)
    text(ax, 0.09, y0 + h - 0.07, "Observation", fontsize=11.5, weight="bold", color=BLUE)
    text(ax, 0.09, 0.47, "last $T_o$ = 2 steps:  $o_{t-1},\\ o_t$\n\n"
         "$o = (x,\\ \\dot{x},\\ f,\\ p_0,\\ L,\\ \\hat{n})$\n16 numbers each\n\n"
         "min-max normalize\n→  condition $s_t$ (32 numbers)", fontsize=10, linespacing=1.6)
    # denoising loop
    rbox(ax, 0.215, y0, 0.43, h, PINK, PINK_L)
    text(ax, 0.43, y0 + h - 0.07, "Denoising loop   (DDIM: 16 of the 100 training steps)", fontsize=11.5,
         weight="bold", color=PINK)
    text(ax, 0.285, 0.60, "start:\n$A_{90} \\sim \\mathcal{N}(0, I)$\n16 × 3", fontsize=10, linespacing=1.5)
    rbox(ax, 0.36, 0.43, 0.17, 0.30, PINK, "white", lw=1.4, r=0.01)
    text(ax, 0.445, 0.66, "1D temporal U-Net", fontsize=10.5, weight="bold", color=PINK)
    text(ax, 0.445, 0.555, "$\\hat{\\epsilon} = \\epsilon_\\theta(A_k,\\ k,\\ s_t)$", fontsize=11)
    text(ax, 0.445, 0.49, "65.7 M parameters", fontsize=9, color=GREY)
    arrow(ax, (0.325, 0.60), (0.36, 0.60), PINK)
    # loop: out of the U-Net right side, down, back in from below
    ax.add_patch(FancyArrowPatch((0.53, 0.47), (0.49, 0.43), arrowstyle="-|>", mutation_scale=15, color=PINK,
                                 lw=1.5, connectionstyle="arc3,rad=-1.6", shrinkA=0, shrinkB=0))
    text(ax, 0.59, 0.60, "repeat for\n$k$ = 90, 84,\n…, 6, 0", fontsize=9.5, color=PINK, linespacing=1.4)
    text(ax, 0.43, 0.25, "update:   $\\hat{a}_0 = (A_k - \\sqrt{1-\\bar\\alpha_k}\\,\\hat\\epsilon)\\,/\\,\\sqrt{\\bar\\alpha_k}$"
         "  (clip to [−1, 1]),     $A_{k-6} = \\sqrt{\\bar\\alpha_{k-6}}\\,\\hat{a}_0 + \\sqrt{1-\\bar\\alpha_{k-6}}\\,\\hat\\epsilon$",
         fontsize=9)
    text(ax, 0.43, 0.12, "FiLM conditioning in every residual block:   $(\\gamma, \\beta)$ = Linear([emb($k$), $s_t$]),"
         "   $h \\leftarrow \\gamma \\odot h + \\beta$", fontsize=9, color=GREY)
    arrow(ax, (0.175, 0.47), (0.215, 0.47), BLUE)
    text(ax, 0.195, 0.51, "$s_t$", fontsize=11, color=BLUE, weight="bold")
    # action chunk
    rbox(ax, 0.685, y0, 0.15, h, ORANGE, ORANGE_L)
    text(ax, 0.76, y0 + h - 0.07, "Action chunk", fontsize=11.5, weight="bold", color=ORANGE)
    text(ax, 0.76, 0.47, "$A_0$ → unnormalize\n\n$\\Delta_0 \\ldots \\Delta_{15}$  [m]\noffsets from $x_t$\n\n"
         "$x_{d,t+j} = x_t + \\Delta_j$\n$j$ = 0 … 15", fontsize=10, linespacing=1.6)
    arrow(ax, (0.645, 0.47), (0.685, 0.47), PINK)
    # label meaning
    rbox(ax, 0.86, y0, 0.135, h, GREY, GREY_L)
    text(ax, 0.9275, y0 + h - 0.07, "What $x_d$ means", fontsize=11.5, weight="bold", color=GREY)
    text(ax, 0.9275, 0.47, "trained on label:\n(a)  $x_{t+1}$\n(b)  $x_{d,t}$\n(c)  $x_{t+1} - K^{-1} f_{t+1}$\n"
         "(c′) (c) $+\\ K^{-1} D \\dot{x}_{t+1}$\n\nsame network and\ninputs; only the\ntraining targets differ",
         fontsize=9.5, linespacing=1.55)
    ax.plot([0.835, 0.86], [0.47, 0.47], ls=":", color=GREY, lw=1.5)


def bottom_panel(ax):
    ax.set_xlim(-3.0, 17.6); ax.set_ylim(-0.75, 5.6); ax.axis("off")
    text(ax, -3.0, 5.45, "②  Execution timeline   (policy step Δt = 50 ms; controller and simulation at 1 kHz)",
         ha="left", fontsize=13, weight="bold")
    Y = dict(call=4.55, chunk1=3.65, chunk2=3.0, ctrl=1.95, plant=0.85)
    labels = [("call", "policy call"), ("chunk1", "target chunks\n(zero-order hold)"), ("ctrl", "impedance\ncontroller"),
              ("plant", "Drake plant\n+ sensors")]
    for k, s in labels:
        yy = (Y["chunk1"] + Y["chunk2"]) / 2 + 0.2 if k == "chunk1" else Y[k]
        text(ax, -2.9, yy, s, ha="left", fontsize=10.5, weight="bold", linespacing=1.3)
    for yy in (4.05, 2.55, 1.4):
        ax.plot([-0.1, 16.9], [yy, yy], color="#e0e0e0", lw=0.9)
    # time axis
    ax.add_patch(FancyArrowPatch((-0.1, -0.15), (17.2, -0.15), arrowstyle="-|>", mutation_scale=15, color=INK, lw=1.2))
    for j in range(17):
        ax.plot([j, j], [-0.22, -0.08], color=INK, lw=1)
        if j in (0, 4, 7, 8, 12, 16):
            text(ax, j, -0.42, "$t$" if j == 0 else f"$t{{+}}{j}$", fontsize=10)
    text(ax, 17.3, -0.15, "time", ha="left", fontsize=10)
    text(ax, 0.5, -0.62, "← 50 ms →", fontsize=8.5, color=GREY)
    # policy calls
    for t0 in (0, 8, 16):
        rbox(ax, t0 - 0.42, Y["call"] - 0.32, 0.84, 0.64, PINK, PINK_L, lw=1.5, r=0.12)
        text(ax, t0, Y["call"], "DP", fontsize=10, weight="bold", color=PINK)
    text(ax, 4, Y["call"], "DDIM sampling of a 16-step chunk", fontsize=9, color=PINK)
    text(ax, 12, Y["call"], "next call from $o_{t+7},\\ o_{t+8}$", fontsize=9, color=PINK)
    # two chunks, staggered: 8 executed (solid), 8 discarded (hatched)
    for t0, yy in ((0, Y["chunk1"]), (8, Y["chunk2"])):
        for j in range(16):
            if t0 + j >= 16:
                break
            ex = j < 8
            ax.add_patch(Rectangle((t0 + j, yy - 0.25), 1, 0.5, ec=ORANGE if ex else "#9e9e9e",
                                   fc=ORANGE_L if ex else "white", lw=1.1, hatch=None if ex else "////"))
            if ex:
                text(ax, t0 + j + 0.5, yy, f"$x_{{d,t+{t0 + j}}}$" if t0 + j else "$x_{d,t}$", fontsize=8)
        arrow(ax, (t0 + 0.15, Y["call"] - 0.32), (t0 + 0.15, yy + 0.25), PINK, lw=1.4)
    text(ax, 12, Y["chunk1"], "$j$ = 8 … 15: predicted, never executed", fontsize=9, color=GREY,
         bbox=dict(fc="white", ec="none", pad=2))
    # controller
    for k in range(160):
        ax.plot([k / 10, k / 10], [Y["ctrl"] - 0.38, Y["ctrl"] - 0.12], color=GREEN, lw=0.6, alpha=0.75)
    text(ax, 0.05, Y["ctrl"] + 0.18, "$\\tau = J^T [\\,K(x_d - x) - D\\dot{x}\\ ;\\ K_{rot}\\, e_R - D_{rot}\\, \\omega\\,] - \\tau_g + N\\tau_{null}$",
         ha="left", fontsize=9.5, color=GREEN)
    text(ax, 8.4, Y["ctrl"] + 0.18, "50 torque updates per held target", ha="left", fontsize=9.5, color=GREEN)
    # plant
    ax.add_patch(Rectangle((0, Y["plant"] - 0.25), 16, 0.5, ec=GREY, fc=GREY_L, lw=1))
    text(ax, 8, Y["plant"], "iiwa + ball tool + table (contact stiffness $k_e$, friction μ), 1 ms steps "
         "→ measures $x, \\dot{x}, f$ every 50 ms; coverage sampled at 100 Hz", fontsize=9)
    # feedback of the observations into the next call
    ax.add_patch(FancyArrowPatch((7.9, Y["plant"] + 0.25), (7.9, Y["call"] - 0.32), arrowstyle="-|>",
                                 mutation_scale=15, color=BLUE, lw=1.6, connectionstyle="arc3,rad=0.0",
                                 shrinkA=0, shrinkB=0, zorder=5))
    text(ax, 6.75, 1.45, "observations\n$o_{t+7},\\ o_{t+8}$", fontsize=9, color=BLUE, weight="bold",
         bbox=dict(fc="white", ec="none", pad=2), zorder=6)


def main():
    fig = plt.figure(figsize=(15, 9.4), dpi=150)
    top_panel(fig.add_axes([0.015, 0.545, 0.97, 0.44]))
    bottom_panel(fig.add_axes([0.015, 0.02, 0.97, 0.50]))
    os.makedirs("results/figures", exist_ok=True)
    fig.savefig("results/figures/dp_inference.png", facecolor="white")
    print("wrote results/figures/dp_inference.png")


if __name__ == "__main__":
    main()
