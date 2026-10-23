"""P3 helper: a policy's target error along n on held-out validation demos.

For every validation window (held-out demo episodes of the run), sample the
policy's chunk with DDIM and compare the first target with the label:
delta = n . (a_hat_0 - a_0). Reports bias and spread (std) of delta overall,
in the sweep phase, and per table. P3 predicts a closed-loop force spread of
about K_eff * std(delta).

Usage: python scripts/action_spread.py --run runs/K500_b_s0 [--ckpt selected]
Output: <run>/action_spread.json
"""
import argparse
import json
import os

import numpy as np
import torch

from wipe.data import load_episode
from wipe.dp.dataset import episode_arrays, make_windows
from wipe.policy import DiffusionPolicy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default="selected")
    ap.add_argument("--batch", type=int, default=4096)
    args = ap.parse_args()
    ck = json.load(open(os.path.join(args.run, "selected.json")))["ckpt"] if args.ckpt == "selected" else args.ckpt
    pol = DiffusionPolicy(args.run, ck)
    cfg = pol.cfg
    out = {"ckpt": ck, "per_table": {}}
    all_d, all_sweep = [], []
    for p in cfg["val_paths"]:
        ep = load_episode(p)
        sc = ep["scene"]
        n = sc.arr("n")
        obs, lab = episode_arrays(ep, cfg["label"], cfg["K"], cfg.get("obs_version", 1))
        O, A = make_windows(obs, lab, ep["x"], cfg["To"], cfg["H"], cfg["relative"])
        x = ep["x"][:len(O)]
        preds = []
        for i in range(0, len(O), args.batch):
            seeds = np.full(len(O[i:i + args.batch]), 7)
            calls = np.arange(i, i + len(seeds))
            preds.append(pol.act(O[i:i + args.batch], x[i:i + args.batch], seeds, calls))
        P = np.concatenate(preds)
        a0_hat = P[:, 0]
        a0 = lab                                     # world-frame label at each step
        d = (a0_hat - a0) @ n
        sweep = ep["phase"] == 2
        all_d.append(d)
        all_sweep.append(d[sweep])
        out["per_table"].setdefault(sc.table, []).append(dict(bias=float(d[sweep].mean()), std=float(d[sweep].std())))
    D, Ds = np.concatenate(all_d), np.concatenate(all_sweep)
    out.update(bias_all_mm=1e3 * float(D.mean()), std_all_mm=1e3 * float(D.std()),
               bias_sweep_mm=1e3 * float(Ds.mean()), std_sweep_mm=1e3 * float(Ds.std()),
               mad_sweep_mm=1e3 * float(np.median(np.abs(Ds - np.median(Ds)))), n_windows=int(len(D)))
    out["per_table"] = {t: dict(bias_mm=1e3 * np.mean([r["bias"] for r in v]),
                                std_mm=1e3 * np.sqrt(np.mean([r["std"] ** 2 for r in v])))
                        for t, v in out["per_table"].items()}
    json.dump(out, open(os.path.join(args.run, "action_spread.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "per_table"}, indent=1))


if __name__ == "__main__":
    main()
