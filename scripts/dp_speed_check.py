"""Closed-loop speed of a trained policy vs the expert, on one val scene per table.

Diagnoses slow policies (e.g. label (c), whose free-space target lacks the K^-1 D xdot lead the
impedance controller needs to keep moving). Reports the median speed before first contact and
the median in-plane speed while in contact (f_n > 1 N), next to the expert's approach and sweep
speeds on the val demos of the same table.
Usage: python scripts/dp_speed_check.py --run runs/K500_c_s0 --tables tilt00_ke2e4 tilt20_ke500
"""
import argparse

import numpy as np
import pandas as pd

from diffusion_policy.policy import DiffusionPolicy
from wipe.config import DT_POLICY, GAINS_BY_KN
from wipe.data import obs_vector
from wipe.scene import make_scene
from wipe.sim import WipeEnv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data", default="data/hf/wipe-labels")
    ap.add_argument("--tables", nargs="+", default=["tilt00_ke2e4", "tilt20_ke500"])
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--t-max", type=float, default=50.0)
    args = ap.parse_args()
    pol = DiffusionPolicy(args.run, "best")
    K = pol.cfg["K"]
    S = pd.read_csv(f"{args.data}/K{K}/val.csv")
    E = pd.read_csv(f"{args.data}/K{K}/episodes.csv")
    for tab in args.tables:
        sc = make_scene(tab, "val", args.index)
        env = WipeEnv(sc, GAINS_BY_KN[K])
        n = sc.arr("n")
        o = env.observe()
        hist = [obs_vector(o["x"], o["xdot"], o["f"], sc)]
        calls, v_contact, v_free = 0, [], []
        while env.k * DT_POLICY < args.t_max:
            O = np.stack([hist[max(0, len(hist) - pol.To + i)] for i in range(pol.To)])[None]
            A = pol.act(O, o["x"][None], [sc.seed], [calls])[0]
            calls += 1
            for xd in A[:8]:
                o = env.step(xd)
                hist.append(obs_vector(o["x"], o["xdot"], o["f"], sc))
                v = o["xdot"]
                if n @ o["f"] > 1.0:
                    v_contact.append(np.linalg.norm(v - (v @ n) * n))
                elif not v_contact:
                    v_free.append(np.linalg.norm(v))
        d = S[S.episode_id.isin(E[(E.table == tab) & (E.dataset_split == "val")].episode_id)]
        dn = d[["n_x", "n_y", "n_z"]].to_numpy()
        dv = d[["xdot_x", "xdot_y", "xdot_z"]].to_numpy()
        dvt = np.linalg.norm(dv - (dv * dn).sum(1, keepdims=True) * dn, axis=1)
        print(f"{tab}: policy pre-contact |v| {np.median(v_free):.4f} m/s, in-contact |v_t| "
              f"{np.median(v_contact):.3f} m/s, uncovered {env.metrics()['uncovered']:.2f} | expert approach |v| "
              f"{np.median(np.linalg.norm(dv[d.phase == 0], axis=1)):.4f} m/s, sweep |v_t| "
              f"{np.median(dvt[d.phase == 2]):.3f} m/s")


if __name__ == "__main__":
    main()
