"""Select a checkpoint with closed-loop validation scenes (never test scenes).

Evaluates the EMA snapshots of a run on the validation split (10 scenes per
training table by default), then picks the snapshot with the most successes;
ties go to the lower validation denoising loss (log.csv). Writes
<run>/selected.json and <run>/val_eval.csv.

Usage: python scripts/select_ckpt.py --run runs/K500_b_s0 [--steps 30000,40000,50000,60000]
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from wipe.config import TRAIN_TABLES
from wipe.evaluate import evaluate_policy
from wipe.policy import DiffusionPolicy
from wipe.scene import make_scene


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--steps", default="")
    ap.add_argument("--last", type=int, default=4, help="evaluate the last N snapshots if --steps is empty")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()
    snaps = sorted(int(f[4:-3]) for f in os.listdir(args.run) if f.startswith("ema_") and f.endswith(".pt"))
    steps = [int(s) for s in args.steps.split(",")] if args.steps else snaps[-args.last:]
    log = pd.read_csv(os.path.join(args.run, "log.csv")).drop_duplicates("step", keep="last").set_index("step")
    out_csv = os.path.join(args.run, "val_eval.csv")
    prev = pd.read_csv(out_csv) if os.path.exists(out_csv) else pd.DataFrame()
    rows = [prev] if len(prev) else []
    for s in steps:
        if len(prev) and (prev.step == s).any():
            continue
        pol = DiffusionPolicy(args.run, os.path.join(args.run, f"ema_{s}.pt"))
        jobs = []
        for t in TRAIN_TABLES:
            for i in range(args.n):
                sc = make_scene(t.name, "val", i)
                jobs.append(dict(scene=sc, K_n=pol.cfg["K"], seed=sc.seed, table=t.name, index=i))
        res = evaluate_policy(pol, jobs, n_workers=args.workers)
        df = pd.DataFrame([dict(step=s, table=j["table"], index=j["index"], **o["metrics"]) for j, o in res])
        rows.append(df)
        print(f"{args.run} step {s}: val success {df.success.sum()}/{len(df)} "
              f"uncovered {df.uncovered.mean():.3f} peak>20 {(df.peak_fn > 20).sum()}", flush=True)
        pd.concat(rows).to_csv(out_csv, index=False)
    allv = pd.concat(rows)
    summ = allv.groupby("step").agg(success=("success", "sum"), n=("success", "size"),
                                    uncovered=("uncovered", "mean")).reset_index()
    summ["val_loss"] = [log.loc[s, "val_loss"] if s in log.index else np.nan for s in summ.step]
    summ = summ.sort_values(["success", "val_loss"], ascending=[False, True])
    best = int(summ.iloc[0].step)
    sel = dict(ckpt=os.path.join(args.run, f"ema_{best}.pt"), step=best,
               val_success=int(summ.iloc[0].success), val_n=int(summ.iloc[0].n),
               table=summ.to_dict(orient="records"))
    json.dump(sel, open(os.path.join(args.run, "selected.json"), "w"), indent=1)
    print(summ.to_string(index=False))
    print("selected", sel["ckpt"])


if __name__ == "__main__":
    main()
