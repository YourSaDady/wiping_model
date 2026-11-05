"""Generate scripted-expert episodes.

--split demo (training demonstrations): for each training table we run the expert on
demo scenes 0, 1, 2, ... and keep the first N successful episodes (failed expert
episodes are logged and skipped, never used for training).
--split test / val / dev (offline test set and expert reference): the expert runs on
scenes 0..N-1 of that split on every table in --tables (default: all 17), and every
episode is kept (its success flag is in the manifest).
Existing files are skipped, so the script resumes after an interruption.

Usage: python scripts/gen_demos.py --K 500 [--n 36] [--workers 16]
       python scripts/gen_demos.py --K 500 --split test --n 10
Output: data/demos/K{K}/{table}/{idx:03d}.npz + manifest.json (demo split),
        data/expert_{split}/K{K}/{table}/{idx:03d}.npz + manifest.json (other splits)
"""
import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor

from wipe.config import ALL_TABLES, GAINS_BY_KN, N_DEMOS_PER_TABLE, OOD_TABLES, TRAIN_TABLES
from wipe.data import demo_path, load_episode, save_episode, split_dir
from wipe.rollout import run_expert
from wipe.scene import make_scene


def job(args):
    K_n, table, idx, split = args
    path = demo_path(K_n, table, idx, split=split)
    if os.path.exists(path):
        m = load_episode(path)["metrics"]
        return table, idx, m["success"], m
    ep = run_expert(make_scene(table, split, idx), GAINS_BY_KN[K_n])
    save_episode(path, ep)
    return table, idx, ep["metrics"]["success"], ep["metrics"]


def table_names(spec, split):
    if not spec:
        spec = "train" if split == "demo" else "all"
    sets = {"train": TRAIN_TABLES, "ood": OOD_TABLES, "all": ALL_TABLES}
    return [t.name for t in sets[spec]] if spec in sets else spec.split(",")


def log(K, table, idx, ok, m):
    print(f"K{K} {table} {idx:03d} ok={ok} unc={m['uncovered']:.3f} peak={m['peak_fn']:.1f} "
          f"dur={m['duration']:.1f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=500)
    ap.add_argument("--split", default="demo", choices=["demo", "val", "test", "dev"])
    ap.add_argument("--n", type=int, default=N_DEMOS_PER_TABLE)
    ap.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    ap.add_argument("--tables", default="", help="'train', 'ood', 'all' or a comma list "
                                                 "(default: train for demo, all otherwise)")
    args = ap.parse_args()
    tables = table_names(args.tables, args.split)
    if args.split == "demo":
        assert all(t in {s.name for s in TRAIN_TABLES} for t in tables), "demos only on training tables"
    kept = {t: [] for t in tables}
    failed = {t: [] for t in tables}
    with ProcessPoolExecutor(args.workers) as ex:
        if args.split == "demo":
            pending = {t: 0 for t in tables}      # next index to try
            while any(len(kept[t]) < args.n for t in tables):
                jobs = []
                for t in tables:
                    need = args.n - len(kept[t])
                    jobs += [(args.K, t, pending[t] + i, "demo") for i in range(need)]
                    pending[t] += need
                for table, idx, ok, m in ex.map(job, jobs):
                    (kept if ok else failed)[table].append(idx)
                    log(args.K, table, idx, ok, m)
        else:
            jobs = [(args.K, t, i, args.split) for t in tables for i in range(args.n)]
            for table, idx, ok, m in ex.map(job, jobs):
                kept[table].append(idx)
                if not ok:
                    failed[table].append(idx)
                log(args.K, table, idx, ok, m)
    out = os.path.join(split_dir(args.K, args.split), "manifest.json")
    manifest = json.load(open(out)) if os.path.exists(out) else {}
    for t in tables:
        manifest[t] = {"kept": sorted(kept[t])[:args.n] if args.split == "demo" else sorted(kept[t]),
                       "failed": sorted(failed[t])}
    with open(out, "w") as f:
        json.dump(manifest, f, indent=1)
    print("failed:", {t: v["failed"] for t, v in manifest.items() if v["failed"]})


if __name__ == "__main__":
    main()
