"""Generate scripted-expert demonstrations.

For each (K_n, training table) we run the expert on demo scenes 0, 1, 2, ...
and keep the first N successful episodes (failed expert episodes are logged
and skipped, never used for training). Existing files are skipped, so the
script resumes after an interruption.

Usage: python scripts/gen_demos.py --K 500 [--n 36] [--workers 28]
Output: data/demos/K{K}/{table}/{idx:03d}.npz, data/demos/K{K}/manifest.json
"""
import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor

from wipe.config import GAINS_BY_KN, N_DEMOS_PER_TABLE, TRAIN_TABLES
from wipe.data import DATA_ROOT, demo_path, load_episode, save_episode
from wipe.rollout import run_expert
from wipe.scene import make_scene


def job(args):
    K_n, table, idx = args
    path = demo_path(K_n, table, idx)
    if os.path.exists(path):
        m = load_episode(path)["metrics"]
        return table, idx, m["success"], m
    ep = run_expert(make_scene(table, "demo", idx), GAINS_BY_KN[K_n])
    save_episode(path, ep)
    return table, idx, ep["metrics"]["success"], ep["metrics"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=500)
    ap.add_argument("--n", type=int, default=N_DEMOS_PER_TABLE)
    ap.add_argument("--workers", type=int, default=28)
    ap.add_argument("--tables", default="")
    args = ap.parse_args()
    tables = [t.name for t in TRAIN_TABLES if not args.tables or t.name in args.tables.split(",")]
    manifest = {}
    with ProcessPoolExecutor(args.workers) as ex:
        pending = {t: 0 for t in tables}      # next index to try
        kept = {t: [] for t in tables}
        failed = {t: [] for t in tables}
        while any(len(kept[t]) < args.n for t in tables):
            jobs = []
            for t in tables:
                need = args.n - len(kept[t])
                jobs += [(args.K, t, pending[t] + i) for i in range(need)]
                pending[t] += need
            for table, idx, ok, m in ex.map(job, jobs):
                (kept if ok else failed)[table].append(idx)
                print(f"K{args.K} {table} {idx:03d} ok={ok} unc={m['uncovered']:.3f} peak={m['peak_fn']:.1f} "
                      f"dur={m['duration']:.1f}", flush=True)
    for t in tables:
        manifest[t] = {"kept": sorted(kept[t])[:args.n], "failed": sorted(failed[t])}
    with open(os.path.join(DATA_ROOT, "demos", f"K{args.K}", "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print("failed:", {t: v["failed"] for t, v in manifest.items() if v["failed"]})


if __name__ == "__main__":
    main()
