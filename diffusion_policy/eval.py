"""Closed-loop inference of a trained Diffusion Policy in the Drake wiping environment.

Batch evaluation (receding horizon: predict H = 16 targets, execute n_exec = 8 at 20 Hz, repeat
until t_max; worker processes simulate, the main process batches inference):
  python -m diffusion_policy.eval --run runs/K500_c_s0 --split val --tables train --n 10
  python -m diffusion_policy.eval --run runs/K500_c_s0 --ckpt 40000 --split test --tables all --n 50
  python -m diffusion_policy.eval --nn c --data data/hf/wipe-labels --split val --n 2   (no training)
One episode with the live Meshcat view and top view (as scripts/watch_expert.py):
  python -m diffusion_policy.eval --run runs/K500_c_s0 --watch tilt00_ke2e4 0 [--split val]

Batch mode appends one JSON line per episode to --out (default results/raw/<name>_<split>.jsonl)
and skips episodes already there, so it resumes. Needs pydrake (wipe/).
"""
import argparse
import json
import os
import sys
import time

import numpy as np

from wipe.config import ALL_TABLES, DT_POLICY, GAINS_BY_KN, OOD_TABLES, T_MAX, TASK, TRAIN_TABLES
from wipe.scene import make_scene


def table_set(name):
    return {"train": TRAIN_TABLES, "ood": OOD_TABLES, "all": ALL_TABLES}.get(name) or \
        [t for t in ALL_TABLES if t.name in name.split(",")]


def load_policy(args):
    if args.nn:
        from diffusion_policy.policy import NearestNeighborPolicy
        pol = NearestNeighborPolicy(args.data, args.K, args.nn, device=args.device)
        return pol, args.name or f"NN_K{args.K}_{args.nn}", args.K
    from diffusion_policy.policy import DiffusionPolicy
    pol = DiffusionPolicy(args.run, args.ckpt, device=args.device, ddim_steps=args.ddim_steps)
    tag = "" if args.ckpt == "best" else f"_ck{args.ckpt}"
    return pol, args.name or os.path.basename(os.path.normpath(args.run)) + tag, pol.cfg["K"]


def watch(policy, K, args):
    """One episode in-process, with Meshcat and the live top view of scripts/watch_expert.py."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
    from watch_expert import TopView, draw_cleaned, draw_region, failure_reasons, param_text
    from wipe.data import obs_vector
    from wipe.paths import LanePath
    from wipe.sim import WipeEnv

    table, index = args.watch[0], int(args.watch[1])
    scene = make_scene(table, args.split, index)
    gains = GAINS_BY_KN[K]
    meshcat = None
    if not args.no_meshcat:
        from pydrake.all import StartMeshcat
        meshcat = StartMeshcat()
        print("Meshcat:", meshcat.web_url(), "(open it now; the episode starts in 3 s)")
        time.sleep(3.0)
    env = WipeEnv(scene, gains, meshcat=meshcat)
    env.simulator.set_target_realtime_rate(args.rate)
    lanes = LanePath(scene, tuple(scene.to_plane(scene.arr("start"))[:2]))
    if meshcat is not None:
        draw_region(meshcat, scene, lanes)
        meshcat.SetCameraPose(np.array([1.4, -1.1, 0.9]), scene.arr("c"))
        env.visualizer.StartRecording()
        env.diagram.ForcedPublish(env.root)
    params = param_text(scene, gains, 0.3, TASK.f_des, TASK)
    print(params)
    view = None if args.no_topview else TopView(scene, env.coverage, lanes, TASK, params)
    n = scene.arr("n")
    o = env.observe()
    hist = [obs_vector(o["x"], o["xdot"], o["f"], scene)]
    calls, fn_wipe = 0, []
    while env.k * DT_POLICY < args.t_max - 1e-9:
        O = np.stack([hist[max(0, len(hist) - policy.To + i)] for i in range(policy.To)])[None]
        A = policy.act(O, o["x"][None], [scene.seed], [calls])[0]
        calls += 1
        for xd in A[:args.n_exec]:
            o = env.step(xd)
            hist.append(obs_vector(o["x"], o["xdot"], o["f"], scene))
            fn = float(n @ o["f"])
            if fn > 1.0:
                fn_wipe.append(fn)
            if env.k % 2 == 0:
                if meshcat is not None:
                    draw_cleaned(meshcat, scene, env.coverage)
                if view is not None:
                    view.update(o["t"], "policy", o["x"], fn, env.samples[-1][5], o["u"], env.peak_force,
                                float(np.mean(fn_wipe)) if fn_wipe else 0.0)
            if env.k * DT_POLICY >= args.t_max - 1e-9:
                break
    m = env.metrics()
    reasons = failure_reasons(m, TASK, True)
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items()})
    print("SUCCESS" if m["success"] else "FAIL: " + "; ".join(reasons))
    if meshcat is not None:
        env.visualizer.StopRecording()
        env.visualizer.PublishRecording()
    if view is not None:
        view.set_verdict(m["success"], reasons)
        if args.save_topview:
            view.save(args.save_topview)
    if (meshcat is not None or view is not None) and sys.stdin.isatty():
        input("Done. Press Enter to quit.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", help="run directory of diffusion_policy/train.py")
    ap.add_argument("--ckpt", default="best", help="best | last | <step> | path")
    ap.add_argument("--nn", choices=["a", "b", "c", "cp"], help="nearest-neighbour policy on this label instead")
    ap.add_argument("--data", default="data/hf/wipe-labels", help="dataset for --nn")
    ap.add_argument("--K", type=int, default=500, help="controller K_n for --nn")
    ap.add_argument("--split", default="val", choices=["val", "test", "dev"])
    ap.add_argument("--tables", default="train", help="train | ood | all | comma list")
    ap.add_argument("--n", type=int, default=10, help="scenes per table")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    ap.add_argument("--n-exec", type=int, default=8)
    ap.add_argument("--ddim-steps", type=int, default=None)
    ap.add_argument("--t-max", type=float, default=T_MAX)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=None)
    ap.add_argument("--name", default=None)
    ap.add_argument("--watch", nargs=2, metavar=("TABLE", "INDEX"), help="one episode with Meshcat + top view")
    ap.add_argument("--rate", type=float, default=1.0)
    ap.add_argument("--no-meshcat", action="store_true")
    ap.add_argument("--no-topview", action="store_true")
    ap.add_argument("--save-topview", default=None)
    args = ap.parse_args()
    assert args.run or args.nn, "give --run or --nn"
    policy, name, K = load_policy(args)
    if args.watch:
        return watch(policy, K, args)

    from wipe.evaluate import evaluate_policy
    out = args.out or f"results/raw/{name}_{args.split}.jsonl"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    done = set()
    if os.path.exists(out):
        done = {(r["table"], r["index"]) for r in map(json.loads, open(out))}
    jobs = []
    for t in table_set(args.tables):
        for i in range(args.start, args.start + args.n):
            if (t.name, i) not in done:
                sc = make_scene(t.name, args.split, i)
                jobs.append(dict(scene=sc, K_n=K, seed=sc.seed, table=t.name, index=i))
    print(f"{name}: {len(jobs)} episodes to run ({len(done)} already in {out})", flush=True)
    f = open(out, "a")

    def on_result(job, res):
        sc = job["scene"]
        row = dict(method=name, ckpt=getattr(policy, "ckpt", None), K_n=K, split=args.split, table=job["table"],
                   index=job["index"], ood=sc.spec.split == "ood", k_e=sc.k_e, tilt=sc.tilt_deg, v=sc.v,
                   **res["metrics"])
        f.write(json.dumps(row) + "\n"); f.flush()

    if jobs:
        evaluate_policy(policy, jobs, n_workers=args.workers, n_exec=args.n_exec, t_max=args.t_max,
                        on_result=on_result)
    f.close()
    names = {t.name for t in table_set(args.tables)}
    rows = [r for r in map(json.loads, open(out)) if r["table"] in names]
    print(f"{name} {args.split}: success {sum(r['success'] for r in rows)}/{len(rows)}  "
          f"mean uncovered {np.mean([r['uncovered'] for r in rows]):.3f}  "
          f"mean f_n while wiping {np.mean([r['mean_fn_wipe'] for r in rows]):.2f} N  "
          f"peak > f_max {sum(r['peak_fn'] > TASK.f_max for r in rows)}")
    for t in sorted(names):
        rt = [r for r in rows if r["table"] == t]
        if rt:
            print(f"  {t:16s} {sum(r['success'] for r in rt)}/{len(rt)}  u {np.mean([r['uncovered'] for r in rt]):.3f}"
                  f"  fn {np.mean([r['mean_fn_wipe'] for r in rt]):.2f}  peak {max(r['peak_fn'] for r in rt):.1f}")


if __name__ == "__main__":
    main()
