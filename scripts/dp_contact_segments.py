"""Contact segments of one closed-loop policy episode: when the tool is on / off the table.

For each segment prints its time span, the maximum tool height above the table, the maximum 20 Hz
normal force, the running 1 kHz peak force and the uncovered fraction at its end. Shows lift-offs
during the sweep (and the impact when the tool lands again) and early lift-off before the region
is done.
Usage: python scripts/dp_contact_segments.py --run runs/K500_cp_s0 --tables tilt00_ke1e5 tilt00_ke2e3
"""
import argparse

import numpy as np

from diffusion_policy.policy import DiffusionPolicy
from wipe.config import DT_POLICY, GAINS_BY_KN
from wipe.data import obs_vector
from wipe.scene import make_scene
from wipe.sim import WipeEnv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--tables", nargs="+", default=["tilt00_ke1e5", "tilt00_ke2e3"])
    ap.add_argument("--split", default="val")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--t-max", type=float, default=50.0)
    args = ap.parse_args()
    pol = DiffusionPolicy(args.run, "best")
    for tab in args.tables:
        sc = make_scene(tab, args.split, args.index)
        env = WipeEnv(sc, GAINS_BY_KN[pol.cfg["K"]])
        n, p0 = sc.arr("n"), sc.arr("p0")
        o = env.observe()
        hist = [obs_vector(o["x"], o["xdot"], o["f"], sc)]
        calls, H = 0, []
        while env.k * DT_POLICY < args.t_max:
            O = np.stack([hist[max(0, len(hist) - pol.To + i)] for i in range(pol.To)])[None]
            A = pol.act(O, o["x"][None], [sc.seed], [calls])[0]
            calls += 1
            for xd in A[:8]:
                o = env.step(xd)
                hist.append(obs_vector(o["x"], o["xdot"], o["f"], sc))
                H.append((o["t"], (o["x"] - p0) @ n, n @ o["f"], env.peak_force, env.metrics()["uncovered"]))
        H = np.array(H)
        c = H[:, 2] > 1.0
        b = np.r_[0, np.flatnonzero(np.diff(c.astype(int))) + 1, len(c)]
        print(f"{tab} ({args.split} {args.index}):")
        for s, e in zip(b[:-1], b[1:]):
            seg = H[s:e]
            print(f"  {'CONTACT' if c[s] else 'free   '} {seg[0, 0]:5.2f}-{seg[-1, 0]:5.2f} s  max height "
                  f"{1e3 * seg[:, 1].max():5.1f} mm  f_n max {seg[:, 2].max():5.1f} N  peak (1 kHz) "
                  f"{seg[-1, 3]:5.1f} N  uncovered {seg[-1, 4]:.2f}")


if __name__ == "__main__":
    main()
