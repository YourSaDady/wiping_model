"""Collect scripted demonstrations into one .npz (format: wiping/dataset.py).

Two ways to choose tables:
  --mode sample   draw every episode from the train / held-out ranges (wiping/presets.py)
  --mode presets  named table types, --per-preset episodes each, with the region,
                  friction and speed jittered per episode

Examples:
    python scripts/collect_demos.py --mode presets --presets flat_soft flat_medium flat_hard --per-preset 5
    python scripts/collect_demos.py --mode sample --split train --n 200 --workers 6 --out data/train.npz
    python scripts/collect_demos.py --mode sample --split train --n 50 --flat-only
    python scripts/collect_demos.py --backend pointmass --n 20      # pipeline check without Drake

It prints a per-table reliability summary; failed episodes are dropped unless
--keep-failed (their configs are still listed in <out>.summary.json).
"""

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wiping import TRAIN_PRESETS, DemoWriter, EpisodeConfig, make_config, run_episode, sample_config  # noqa: E402
from wiping.presets import PRESETS  # noqa: E402


def build_configs(args):
    rng = np.random.default_rng(args.seed)
    cfgs = []
    if args.mode == "sample":
        for i in range(args.n):
            cfgs.append(sample_config(rng, args.split, backend=args.backend, flat_only=args.flat_only,
                                      seed=args.seed * 100000 + i))
    else:
        for p in args.presets:
            for j in range(args.per_preset):
                over = {}
                if j > 0 or args.jitter_first:  # episode 0 of each preset is the nominal one
                    over = dict(table__mu=float(rng.uniform(0.3, 0.7)),
                                region__center_uv=(float(rng.uniform(-0.03, 0.03)), float(rng.uniform(-0.03, 0.03))),
                                region__L=float(rng.uniform(0.10, 0.14)),
                                expert__speed=float(rng.uniform(0.04, 0.08)))
                cfg = make_config(p, backend=args.backend, seed=args.seed * 100000 + len(cfgs), **over)
                cfgs.append(cfg.replace(name=p))
    return cfgs


def _run(cfg_dict):
    cfg = EpisodeConfig.from_dict(cfg_dict)
    try:
        ep = run_episode(cfg)
        return ep, None
    except Exception as e:  # IK failure etc. -- keep going, report it
        return None, f"{type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["sample", "presets"], default="sample")
    ap.add_argument("--split", choices=["train", "heldout"], default="train")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--flat-only", action="store_true")
    ap.add_argument("--presets", nargs="*", default=TRAIN_PRESETS, choices=sorted(PRESETS))
    ap.add_argument("--per-preset", type=int, default=5)
    ap.add_argument("--jitter-first", action="store_true")
    ap.add_argument("--backend", default="drake", choices=["drake", "pointmass"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--keep-failed", action="store_true")
    ap.add_argument("--out", default=None, help="default: data/demos_<mode>_<split|presets>_<backend>.npz")
    args = ap.parse_args()

    cfgs = build_configs(args)
    out = Path(args.out or ROOT / "data" / f"demos_{args.mode}_{args.split if args.mode == 'sample' else 'presets'}_{args.backend}.npz")
    print(f"{len(cfgs)} episodes -> {out}")

    writer, rows = DemoWriter(), []
    t0 = time.time()
    dicts = [c.to_dict() for c in cfgs]
    if args.workers > 1:
        with ProcessPoolExecutor(args.workers) as ex:
            results = ex.map(_run, dicts)
            results = list(results)
    else:
        results = map(_run, dicts)
    for i, (cfg, (ep, err)) in enumerate(zip(cfgs, results)):
        row = dict(i=i, name=cfg.name, k_e=cfg.table.k_e, tilt_deg=float(np.degrees(cfg.table.tilt)),
                   mu=cfg.table.mu, speed=cfg.expert.speed, L=cfg.region.L)
        if ep is None:
            row.update(success=False, error=err)
        else:
            row.update(ep["metrics"])
            if ep["metrics"]["success"] or args.keep_failed:
                writer.add(ep)
        rows.append(row)
        m = row
        print(f"  [{i:4d}] {cfg.name:22s} k_e={cfg.table.k_e:8.0f} tilt={row['tilt_deg']:5.1f} "
              + (f"ERROR {err}" if ep is None else
                 f"ok={m['success']!s:5s} u_T={m['u_T']:.3f} fmax={m['fn_max']:5.1f} "
                 f"f_wipe={m['fn_wipe_mean']:5.2f}+-{m['fn_wipe_std']:.2f}  wall={ep['wall_time']:.1f}s"),
              flush=True)

    # reliability summary
    print("\nreliability by table type:")
    names = sorted({r["name"] for r in rows}) if args.mode == "presets" else ["all"]
    for nm in names:
        rs = [r for r in rows if args.mode != "presets" or r["name"] == nm]
        s = np.array([r["success"] for r in rs])
        print(f"  {nm:22s} {s.sum():4d}/{len(s):<4d} = {s.mean():.2f}")
    print(f"total wall time {time.time() - t0:.0f}s")

    if len(writer):
        writer.save(out)
        print("saved", out, f"({len(writer)} episodes)")
    summ = out.with_suffix(".summary.json")
    summ.parent.mkdir(parents=True, exist_ok=True)
    summ.write_text(json.dumps(dict(args=vars(args), rows=rows), indent=1, default=float))
    print("summary", summ)


if __name__ == "__main__":
    main()
