"""P2 diagnostic: is the remaining (c')-(b) gap the inertial term?

Near-static balance at the TCP is K(x_d - x) - D xdot + f = [Lambda a]_trans, where a is the
6-D spatial acceleration of the ball centre and Lambda = (J M^-1 J^T)^-1. Label (c') drops
the right-hand side, so x_d - (c') should equal K^-1 [Lambda a]_trans evaluated just before
each zero-order-hold switch (t_{k+1}^-). We replay demos (label (b) is exact), step the plant
1 ms at a time over the last millisecond of each interval to estimate a, and compare.

Usage: python results/p2_inertia_check.py [--n 3]
Output: results/tables/p2_inertia_check.csv
"""
import argparse

import numpy as np
import pandas as pd
from pydrake.all import JacobianWrtVariable

from wipe.config import DT_POLICY, DT_SIM, MAIN_GAINS
from wipe.data import list_demos, load_episode
from wipe.labels import make_labels
from wipe.sim import WipeEnv

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=3)
ap.add_argument("--tables", default="tilt00_ke2e4,tilt20_ke500,tilt10_ke1e5")
args = ap.parse_args()
rows = []
g = MAIN_GAINS
for table in args.tables.split(","):
    for p in list_demos(500, table)[:args.n]:
        ep = load_episode(p)
        sc = ep["scene"]
        n = sc.arr("n")
        K, D = g.K(n), g.D(n)
        cp = make_labels(ep["x"], ep["xdot"], ep["f"], ep["x_d"], K, D, "cp")
        env = WipeEnv(sc, g)
        plant, ctx, frame = env.plant, env.pctx, env.tool.body_frame()
        W = plant.world_frame()
        pred = []
        for k, xd in enumerate(ep["x_d"]):
            env.xd_port.FixValue(env.root, xd)
            t1 = (k + 1) * DT_POLICY
            env.simulator.AdvanceTo(t1 - DT_SIM)
            V0 = plant.EvalBodySpatialVelocityInWorld(ctx, env.tool).get_coeffs().copy()
            env.simulator.AdvanceTo(t1 - 1e-9)
            V1 = plant.EvalBodySpatialVelocityInWorld(ctx, env.tool).get_coeffs().copy()
            a = (V1 - V0) / DT_SIM
            M = plant.CalcMassMatrix(ctx)
            J = plant.CalcJacobianSpatialVelocity(ctx, JacobianWrtVariable.kV, frame, np.zeros(3), W, W)
            Lam = np.linalg.inv(J @ np.linalg.solve(M, J.T))
            pred.append(np.linalg.solve(K, (Lam @ a)[3:]))
            env.simulator.AdvanceTo(t1)
            env.k = k + 1
        pred = np.array(pred)
        resid = ep["x_d"] - cp
        ph = ep["phase"]
        for name, m in (("all", ph >= 0), ("sweep", ph == 2), ("approach", ph == 0)):
            r, q = resid[m], pred[m]
            rows.append(dict(table=table, demo=p.split("/")[-1], phase=name,
                             resid_mm=1e3 * np.linalg.norm(r, axis=1).mean(),
                             after_inertia_mm=1e3 * np.linalg.norm(r - q, axis=1).mean(),
                             corr=float(np.corrcoef(r.ravel(), q.ravel())[0, 1])))
        print(rows[-3:], flush=True)
df = pd.DataFrame(rows)
df.to_csv("results/tables/p2_inertia_check.csv", index=False)
print(df.groupby("phase")[["resid_mm", "after_inertia_mm", "corr"]].mean())
