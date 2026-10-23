"""Phase 1 acceptance checks.

1. Expert on N 'dev' scenes per table (all tables, main K; training tables at
   K_n = 250 and 1000): success >= 95 % and peak force <= f_max.
2. Open-loop replay of each label on flat tables: (b) must reproduce the demo
   exactly; the force kept by (a) during the sweep must match K/(K + k_e).
3. Figure: normal force vs time for each label replay.

Usage: python scripts/phase1_checks.py [--n 50] [--workers 28]
Outputs: results/phase1/{expert_check.csv, replay_check.csv, replay_force_*.png}
"""
import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from wipe.config import ALL_TABLES, GAINS_BY_KN, TASK, TRAIN_TABLES
from wipe.labels import LABELS, make_labels
from wipe.rollout import replay_targets, run_expert
from wipe.scene import make_scene

OUT = "results/phase1"


def expert_job(args):
    table, idx, K_n = args
    sc = make_scene(table, "dev", idx)
    ep = run_expert(sc, GAINS_BY_KN[K_n])
    m = ep["metrics"]
    m.update(table=table, index=idx, K_n=K_n, expert_done=ep["expert_done"], v=sc.v, L=sc.L)
    return m


def replay_job(args):
    table, idx, K_n = args
    g = GAINS_BY_KN[K_n]
    sc = make_scene(table, "dev", idx)
    demo = run_expert(sc, g)
    n = sc.arr("n")
    K, D = g.K(n), g.D(n)
    sweep = demo["phase"] == 2
    out = {"demo": demo}
    rows = []
    for lab in LABELS:
        tgt = make_labels(demo["x"], demo["xdot"], demo["f"], demo["x_d"], K, D, lab)
        rep = replay_targets(sc, tgt, g)
        out[lab] = rep
        fn_demo = demo["f"][1:] @ n
        fn_rep = rep["f"][1:] @ n
        rows.append(dict(
            table=table, index=idx, K_n=K_n, k_e=sc.k_e, tilt=sc.tilt_deg, label=lab, v=sc.v,
            max_dx=float(np.abs(rep["x"] - demo["x"]).max()),
            fn_demo_sweep=float(fn_demo[sweep].mean()), fn_rep_sweep=float(fn_rep[sweep].mean()),
            ratio=float(fn_rep[sweep].mean() / fn_demo[sweep].mean()),
            ratio_pred_a=K_n / (K_n + sc.k_e),
            peak_rep=rep["metrics"]["peak_fn"], success_rep=rep["metrics"]["success"],
            uncovered_rep=rep["metrics"]["uncovered"],
            lag_inplane_mm=float(1e3 * np.mean(np.linalg.norm(
                (rep["x"][1:] - demo["x"][1:])[sweep] - np.outer((rep["x"][1:] - demo["x"][1:])[sweep] @ n, n),
                axis=1))),
        ))
    traces = {lab: (out[lab]["t"][1:], out[lab]["f"][1:] @ n) for lab in LABELS}
    traces["demo"] = (demo["t"][1:], demo["f"][1:] @ n)
    return rows, traces, (table, idx, K_n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--workers", type=int, default=28)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    jobs = [(t.name, i, 500) for t in ALL_TABLES for i in range(args.n)]
    jobs += [(t.name, i, K) for K in (250, 1000) for t in TRAIN_TABLES for i in range(args.n)]
    with ProcessPoolExecutor(args.workers) as ex:
        res = list(ex.map(expert_job, jobs, chunksize=2))
    df = pd.DataFrame(res)
    df.to_csv(f"{OUT}/expert_check.csv", index=False)
    summ = df.groupby(["K_n", "table"]).agg(
        success=("success", "mean"), peak_max=("peak_fn", "max"), peak_mean=("peak_fn", "mean"),
        unc_max=("uncovered", "max"), fn_wipe=("mean_fn_wipe", "mean"), dur_max=("duration", "max"),
        done=("expert_done", "mean"), n=("success", "size")).reset_index()
    summ.to_csv(f"{OUT}/expert_check_summary.csv", index=False)
    print(summ.to_string())

    rjobs = [(f"tilt00_ke{k}", i, 500) for k in ("500", "2e3", "2e4", "1e5") for i in range(3)]
    rjobs += [(f"tilt20_ke{k}", i, 500) for k in ("500", "2e4") for i in range(3)]
    rjobs += [(f"tilt00_ke{k}", i, K) for K in (250, 1000) for k in ("500", "2e3", "2e4", "1e5") for i in range(2)]
    with ProcessPoolExecutor(args.workers) as ex:
        rres = list(ex.map(replay_job, rjobs))
    rows = [r for rr, _, _ in rres for r in rr]
    rdf = pd.DataFrame(rows)
    rdf.to_csv(f"{OUT}/replay_check.csv", index=False)
    print(rdf.groupby(["K_n", "tilt", "k_e", "label"]).agg(
        max_dx=("max_dx", "max"), ratio=("ratio", "mean"), ratio_pred_a=("ratio_pred_a", "first"),
        lag_mm=("lag_inplane_mm", "mean"), success=("success_rep", "mean")).to_string())
    np.save(f"{OUT}/replay_traces.npy", np.array([(key, tr) for _, tr, key in rres], dtype=object),
            allow_pickle=True)


if __name__ == "__main__":
    main()
