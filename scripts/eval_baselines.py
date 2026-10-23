"""Evaluate the two baselines that use no demonstrations.

  scripted: the privileged scripted wiping program (the demo expert)
  hybrid:   the classical hybrid force/position controller (no table height)

Usage: python scripts/eval_baselines.py --method hybrid --split test --tables all --n 50 [--K 500]
Appends JSON lines to results/raw/<method>_K<K>_<split>.jsonl (resumable).
"""
import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from wipe.config import ALL_TABLES, GAINS_BY_KN, OOD_TABLES, TRAIN_TABLES
from wipe.rollout import run_expert, run_hybrid
from wipe.scene import make_scene


def job(args):
    method, table, split, idx, K_n, trace, record = args
    sc = make_scene(table, split, idx)
    g = GAINS_BY_KN[K_n]
    ep = run_expert(sc, g, record_q=record) if method == "scripted" else run_hybrid(sc, g, record_q=record)
    keep = None
    if trace or record:
        keep = {"x": ep["x"].astype(np.float32), "f": ep["f"].astype(np.float32), "samples": ep["samples"]}
        if "x_d" in ep:
            keep["xd"] = ep["x_d"].astype(np.float32)
        if record:
            keep["q_hi"] = ep["q_hi"]
    return (table, idx, sc, ep["metrics"], keep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["scripted", "hybrid"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--tables", default="all")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--K", type=int, default=500)
    ap.add_argument("--workers", type=int, default=28)
    ap.add_argument("--traces", type=int, default=0)
    ap.add_argument("--record-q", type=int, default=0)
    args = ap.parse_args()
    tables = {"train": TRAIN_TABLES, "ood": OOD_TABLES, "all": ALL_TABLES}.get(args.tables) or \
        [t for t in ALL_TABLES if t.name in args.tables.split(",")]
    name = f"{args.method}_K{args.K}"
    out = f"results/raw/{name}_{args.split}.jsonl"
    os.makedirs("results/raw", exist_ok=True)
    done = set()
    if os.path.exists(out):
        done = {(r["table"], r["index"]) for r in map(json.loads, open(out))}
    jobs = [(args.method, t.name, args.split, i, args.K, i < args.traces, i < args.record_q)
            for t in tables for i in range(args.n) if (t.name, i) not in done]
    print(f"{name}: {len(jobs)} episodes to run", flush=True)
    trace_dir = os.path.splitext(out)[0] + "_traces"
    with open(out, "a") as f, ProcessPoolExecutor(args.workers) as ex:
        for table, idx, sc, m, keep in ex.map(job, jobs, chunksize=1):
            row = dict(method=name, ckpt=None, K_n=args.K, split=args.split, table=table, index=idx,
                       ood=sc.spec.split == "ood", k_e=sc.k_e, tilt=sc.tilt_deg, v=sc.v, **m)
            f.write(json.dumps(row) + "\n"); f.flush()
            if keep is not None:
                os.makedirs(trace_dir, exist_ok=True)
                np.savez_compressed(os.path.join(trace_dir, f"{table}_{idx:03d}.npz"),
                                    scene=json.dumps(sc.to_dict()), **keep)
    rows = [json.loads(l) for l in open(out)]
    print(f"{name} {args.split}: success {sum(r['success'] for r in rows)}/{len(rows)}")


if __name__ == "__main__":
    main()
