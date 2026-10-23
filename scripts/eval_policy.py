"""Closed-loop evaluation of a Diffusion Policy (or the nearest-neighbour smoke test).

Usage:
  python scripts/eval_policy.py --run runs/K500_b_s0 --split val --tables train --n 10
  python scripts/eval_policy.py --run runs/K500_b_s0 --ckpt selected --split test --tables all --n 50
  python scripts/eval_policy.py --nn b --K 500 --split val --tables train --n 10
Appends one JSON line per episode to --out (default results/raw/<name>_<split>.jsonl)
and skips episodes already present, so it can resume. --traces N saves the
20 Hz traces and 100 Hz samples of the first N scenes per table.
"""
import argparse
import json
import os

import numpy as np

from wipe.config import ALL_TABLES, OOD_TABLES, T_MAX, TRAIN_TABLES
from wipe.evaluate import evaluate_policy
from wipe.scene import make_scene


def table_set(name):
    return {"train": TRAIN_TABLES, "ood": OOD_TABLES, "all": ALL_TABLES}.get(name) or \
        [t for t in ALL_TABLES if t.name in name.split(",")]


def resolve_ckpt(run, ckpt):
    if ckpt in (None, "latest"):
        return None
    if ckpt == "selected":
        return json.load(open(os.path.join(run, "selected.json")))["ckpt"]
    if ckpt.isdigit():
        return os.path.join(run, f"ema_{ckpt}.pt")
    return ckpt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run")
    ap.add_argument("--nn", choices=["a", "b", "c", "cp"])
    ap.add_argument("--K", type=int, default=None)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--split", default="val", choices=["val", "test", "dev"])
    ap.add_argument("--tables", default="train")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--n-exec", type=int, default=8)
    ap.add_argument("--ddim-steps", type=int, default=None)
    ap.add_argument("--t-max", type=float, default=T_MAX)
    ap.add_argument("--traces", type=int, default=0)
    ap.add_argument("--record-q", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    if args.nn:
        from wipe.data import split_paths
        from wipe.policy import NearestNeighborPolicy
        K = args.K or 500
        tr, _ = split_paths(K, [t.name for t in TRAIN_TABLES], 32, 4)
        policy = NearestNeighborPolicy(tr, args.nn, K)
        name = args.name or f"NN_K{K}_{args.nn}"
        ckpt = None
    else:
        from wipe.policy import DiffusionPolicy
        ckpt = resolve_ckpt(args.run, args.ckpt)
        policy = DiffusionPolicy(args.run, ckpt, ddim_steps=args.ddim_steps)
        K = policy.cfg["K"]
        tag = "" if args.ckpt in (None, "latest", "selected") else f"_ck{os.path.basename(policy.ckpt)[4:-3]}"
        name = args.name or os.path.basename(os.path.normpath(args.run)) + tag
        ckpt = policy.ckpt
    out = args.out or f"results/raw/{name}_{args.split}.jsonl"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    done = set()
    if os.path.exists(out):
        for line in open(out):
            r = json.loads(line)
            done.add((r["table"], r["index"]))
    jobs = []
    for t in table_set(args.tables):
        for i in range(args.start, args.start + args.n):
            if (t.name, i) in done:
                continue
            sc = make_scene(t.name, args.split, i)
            jobs.append(dict(scene=sc, K_n=K, seed=sc.seed, table=t.name, index=i,
                             trace=i - args.start < args.traces, record=i - args.start < args.record_q))
    print(f"{name}: {len(jobs)} episodes to run ({len(done)} already in {out})", flush=True)
    trace_dir = os.path.splitext(out)[0] + "_traces"
    f = open(out, "a")

    def on_result(job, res):
        row = dict(method=name, ckpt=ckpt, K_n=K, split=args.split, table=job["table"], index=job["index"],
                   ood=job["scene"].spec.split == "ood", k_e=job["scene"].k_e, tilt=job["scene"].tilt_deg,
                   v=job["scene"].v, **res["metrics"])
        f.write(json.dumps(row) + "\n"); f.flush()
        if job["trace"] or job["record"]:
            os.makedirs(trace_dir, exist_ok=True)
            arrs = {k: v for k, v in res["trace"].items()}
            arrs["samples"] = res["samples"]
            if "q_hi" in res:
                arrs["q_hi"] = res["q_hi"]
            np.savez_compressed(os.path.join(trace_dir, f"{job['table']}_{job['index']:03d}.npz"),
                                scene=json.dumps(job["scene"].to_dict()), **arrs)

    if jobs:
        evaluate_policy(policy, jobs, n_workers=args.workers, n_exec=args.n_exec, t_max=args.t_max,
                        on_result=on_result)
    f.close()
    rows = [json.loads(l) for l in open(out)]
    sel = [r for r in rows if r["table"] in {t.name for t in table_set(args.tables)}]
    print(f"{name} {args.split}: success {sum(r['success'] for r in sel)}/{len(sel)}  "
          f"mean uncovered {np.mean([r['uncovered'] for r in sel]):.3f}  "
          f"mean fn {np.mean([r['mean_fn_wipe'] for r in sel]):.2f}  peak>fmax "
          f"{sum(r['peak_fn'] > 20 for r in sel)}")


if __name__ == "__main__":
    main()
