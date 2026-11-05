"""Export the expert episodes as the CSV dataset (one folder, ready for Hugging Face).

Reads data/demos/K{K}/ (gen_demos.py --split demo) and data/expert_test/K{K}/
(gen_demos.py --split test) and writes

  <out>/K{K}/train.csv     first n_train kept demos per training table   (policy training)
  <out>/K{K}/val.csv       next n_val kept demos per training table      (offline validation loss)
  <out>/K{K}/test.csv      expert on test scenes 0..n_test-1 of all 17 tables (offline test,
                           in-distribution + held-out tables; same scenes as the closed-loop test)
  <out>/K{K}/episodes.csv  one row per episode (scene, table, expert metrics, split)
  <out>/scenes/val.csv, test.csv   closed-loop evaluation scenes (10 val / 50 test per table)
  <out>/stats.json, <out>/stats.md, <out>/README.md (dataset card)

Each step row holds the observation at step t and all four labels for step t
(wipe/labels.py), so the four training sets differ only in the label columns.

Usage: python scripts/export_dataset.py --K 500 250 1000 [--out data/hf/wipe-labels]
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from diffusion_policy.dataset import LABEL_COLS, LABELS, OBS_COLS, PHASES, STEP_COLS
from wipe.config import (ALL_TABLES, GAINS_BY_KN, N_TEST_SCENES_PER_TABLE, N_VAL_SCENES_PER_TABLE, TASK,
                         TRAIN_TABLES)
from wipe.data import DATA_ROOT, demo_path, load_episode, split_dir
from wipe.labels import make_labels
from wipe.scene import make_scene

SCENE_COLS = ["table", "split", "index", "tilt_deg", "k_e", "L", "v"] + \
    [f"{k}_{c}" for k in ("p0", "n", "t1", "t2", "start", "c") for c in "xyz"]


def scene_row(sc):
    d = dict(table=sc.table, split=sc.split, index=sc.index, tilt_deg=sc.tilt_deg, k_e=sc.k_e, L=sc.L, v=sc.v)
    for k in ("p0", "n", "t1", "t2", "start", "c"):
        d.update({f"{k}_{c}": float(v) for c, v in zip("xyz", getattr(sc, k))})
    return d


def episode_rows(ep, eid, K):
    sc = ep["scene"]
    g = GAINS_BY_KN[K]
    n = sc.arr("n")
    T = len(ep["x_d"])
    lab = {l: make_labels(ep["x"], ep["xdot"], ep["f"], ep["x_d"], g.K(n), g.D(n), l) for l in LABELS}
    const = np.concatenate([sc.arr("p0"), [sc.L], n])
    obs = np.concatenate([ep["x"][:T], ep["xdot"][:T], ep["f"][:T], np.broadcast_to(const, (T, 7))], 1)
    d = {"episode_id": eid, "step": np.arange(T), "t": ep["t"][:T], "phase": ep["phase"][:T]}
    d.update({c: obs[:, i] for i, c in enumerate(OBS_COLS)})
    d["fn"] = ep["f"][:T] @ n
    d["u"] = ep["u"][:T]
    for l in LABELS:
        d.update({c: lab[l][:, i] for i, c in enumerate(LABEL_COLS[l])})
    return pd.DataFrame(d)[STEP_COLS]


def collect(K, n_train, n_val, n_test):
    man = json.load(open(os.path.join(split_dir(K, "demo"), "manifest.json")))
    plan = []
    for t in TRAIN_TABLES:
        kept = man[t.name]["kept"]
        assert len(kept) >= n_train + n_val, (t.name, len(kept))
        plan += [("train", t.name, i, "demo") for i in kept[:n_train]]
        plan += [("val", t.name, i, "demo") for i in kept[n_train:n_train + n_val]]
    for t in ALL_TABLES:
        for i in range(n_test):
            plan.append(("test", t.name, i, "test"))
    steps = {"train": [], "val": [], "test": []}
    eps = []
    for split, table, idx, src in plan:
        path = demo_path(K, table, idx, split=src)
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path}: run scripts/gen_demos.py --K {K} --split {src}")
        ep = load_episode(path)
        eid = f"K{K}_{table}_{src}{idx:03d}"
        steps[split].append(episode_rows(ep, eid, K))
        m = ep["metrics"]
        eps.append(dict(episode_id=eid, dataset_split=split, K_n=K, table_split=ep["scene"].spec.split,
                        n_steps=len(ep["x_d"]), expert_done=bool(ep.get("expert_done", True)),
                        **scene_row(ep["scene"]),
                        **{k: m[k] for k in ("success", "uncovered", "peak_fn", "mean_fn_wipe", "std_fn_wipe",
                                             "duration", "first_contact_t")}))
    return {k: pd.concat(v, ignore_index=True) for k, v in steps.items()}, pd.DataFrame(eps)


def stats(K, steps, eps):
    s = {"K_n": K, "splits": {}, "tables": {}, "labels": {}}
    for split, df in steps.items():
        e = eps[eps.dataset_split == split]
        sw = df[df.phase == 2]
        s["splits"][split] = dict(
            episodes=len(e), tables=int(e.table.nunique()), ood_episodes=int((e.table_split == "ood").sum()),
            steps=len(df), hours=len(df) * 0.05 / 3600, duration_mean=float(e.duration.mean()),
            duration_min=float(e.duration.min()), duration_max=float(e.duration.max()),
            expert_success=float(e.success.mean()), uncovered_mean=float(e.uncovered.mean()),
            peak_fn_max=float(e.peak_fn.max()), fn_sweep_mean=float(sw.fn.mean()), fn_sweep_std=float(sw.fn.std()),
            phase_frac={PHASES[p]: float((df.phase == p).mean()) for p in PHASES})
    tr = steps["train"]
    et = eps[eps.dataset_split == "train"]
    for table, e in et.groupby("table", sort=False):
        d = tr[tr.episode_id.isin(e.episode_id)]
        sw = d[d.phase == 2]
        s["tables"][table] = dict(episodes=len(e), steps=len(d), duration_mean=float(e.duration.mean()),
                                  fn_sweep_mean=float(sw.fn.mean()), fn_sweep_std=float(sw.fn.std()),
                                  peak_fn_max=float(e.peak_fn.max()))
    # where each label puts the target relative to the ball: offset along n (depth) and in-plane
    x = tr[OBS_COLS[0:3]].to_numpy()
    n = tr[["n_x", "n_y", "n_z"]].to_numpy()
    for l in LABELS:
        d = tr[LABEL_COLS[l]].to_numpy() - x
        dn = (d * n).sum(1)
        dt = np.linalg.norm(d - dn[:, None] * n, axis=1)
        s["labels"][l] = {}
        for p, name in PHASES.items():
            m = tr.phase.to_numpy() == p
            s["labels"][l][name] = dict(normal_mm_mean=1e3 * float(dn[m].mean()), normal_mm_std=1e3 * float(dn[m].std()),
                                        tangential_mm_mean=1e3 * float(dt[m].mean()))
    return s


def stats_md(all_stats):
    L = []
    L.append("| K_n | split | episodes (tables) | steps (20 Hz) | hours | duration mean [min, max] s | "
             "expert success | mean u_T | f_n sweep mean ± std [N] | max peak f_n [N] |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for s in all_stats:
        for split, d in s["splits"].items():
            ood = f", {d['ood_episodes']} OOD" if d["ood_episodes"] else ""
            L.append(f"| {s['K_n']} | {split} | {d['episodes']} ({d['tables']}{ood}) | {d['steps']:,} | "
                     f"{d['hours']:.2f} | {d['duration_mean']:.1f} [{d['duration_min']:.1f}, {d['duration_max']:.1f}] | "
                     f"{d['expert_success']:.3f} | {d['uncovered_mean']:.4f} | {d['fn_sweep_mean']:.2f} ± "
                     f"{d['fn_sweep_std']:.2f} | {d['peak_fn_max']:.1f} |")
    L.append("")
    s = all_stats[0]
    L.append(f"Phase share of the K_n = {s['K_n']} train steps: " + ", ".join(
        f"{k} {100 * v:.1f} %" for k, v in s["splits"]["train"]["phase_frac"].items()))
    L.append("")
    L.append(f"Per training table (K_n = {s['K_n']}, train split):")
    L.append("")
    L.append("| table | episodes | steps | mean duration [s] | f_n sweep mean ± std [N] | max peak f_n [N] |")
    L.append("|---|---|---|---|---|---|")
    for t, d in s["tables"].items():
        L.append(f"| {t} | {d['episodes']} | {d['steps']:,} | {d['duration_mean']:.1f} | "
                 f"{d['fn_sweep_mean']:.2f} ± {d['fn_sweep_std']:.2f} | {d['peak_fn_max']:.1f} |")
    L.append("")
    for s in all_stats:
        L.append(f"Label offset from the current ball position, label_t − x_t, K_n = {s['K_n']} train split "
                 "(mean ± std along n̂ [mm]; mean in-plane norm [mm]):")
        L.append("")
        L.append("| label | " + " | ".join(PHASES.values()) + " |")
        L.append("|---|" + "---|" * len(PHASES))
        for l, d in s["labels"].items():
            L.append(f"| {l} | " + " | ".join(
                f"{d[p]['normal_mm_mean']:+.1f} ± {d[p]['normal_mm_std']:.1f}; {d[p]['tangential_mm_mean']:.1f}"
                for p in PHASES.values()) + " |")
        L.append("")
    return "\n".join(L)


CARD = """---
license: mit
pretty_name: Force-aware wiping demonstrations with four action labels
tags: [robotics, imitation-learning, diffusion-policy, drake, contact]
configs:
{configs}
---

# Force-aware wiping: scripted-expert demonstrations with four action labels

MIT 6.4212 (Fall 2026) project of Yanchuan Tang, Linda Yinghui He, Yichuan Song. Code and full
documentation: the `wiping_model` repository (README sections "Data" and "Diffusion Policy").

A Cartesian-impedance KUKA iiwa (Drake) wipes an L x L region of a table with a ball tool. A scripted
expert presses along -n until it holds f_des = 10 N, sweeps 6 lanes and returns home. Every step row
holds the observation and **four action labels built from the same episode**:

| label | columns | formula |
|---|---|---|
| (a) measured pose | `a_x a_y a_z` | x_(t+1) |
| (b) commanded target | `b_x b_y b_z` | x_(d,t) |
| (c) force-compensated | `c_x c_y c_z` | x_(t+1) - K^-1 f_(t+1) |
| (c') + damping | `cp_x cp_y cp_z` | x_(t+1) + K^-1 (D xdot_(t+1) - f_(t+1)) |

Observation columns: `x_*` ball centre [m], `xdot_*` [m/s], `f_*` contact force on the tool [N]
(world frame), `p0_*` region corner, `L` region side, `n_*` table normal. Also `episode_id`, `step`,
`t` [s], `phase` (0 approach, 1 press, 2 sweep, 3 lift, 4 return), `fn` = n.f, `u` uncovered fraction
so far. K is the controller stiffness of the folder: K_t = 1000 N/m in-plane, K_n along n.

Folders: `K500` (main), `K250`, `K1000` (controller-gain sweep); each has `train.csv`, `val.csv`
(held-out demos, same tables), `test.csv` (expert on test scenes of all 17 tables, incl. 5 held-out
tables) and `episodes.csv` (scene parameters and expert metrics per episode). `scenes/val.csv` and
`scenes/test.csv` list the closed-loop evaluation scenes.

## Statistics

{stats}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, nargs="+", default=[500])
    ap.add_argument("--out", default=os.path.join(DATA_ROOT, "hf", "wipe-labels"))
    ap.add_argument("--n-train", type=int, default=32)
    ap.add_argument("--n-val", type=int, default=4)
    ap.add_argument("--n-test", type=int, default=10)
    args = ap.parse_args()
    all_stats = []
    for K in args.K:
        steps, eps = collect(K, args.n_train, args.n_val, args.n_test)
        d = os.path.join(args.out, f"K{K}")
        os.makedirs(d, exist_ok=True)
        for split, df in steps.items():
            df.to_csv(os.path.join(d, f"{split}.csv"), index=False, float_format="%.7g")
        eps.to_csv(os.path.join(d, "episodes.csv"), index=False, float_format="%.7g")
        all_stats.append(stats(K, steps, eps))
        print(f"K{K}: " + ", ".join(f"{k} {len(v)} rows" for k, v in steps.items()), flush=True)
    os.makedirs(os.path.join(args.out, "scenes"), exist_ok=True)
    for split, n, tables in (("val", N_VAL_SCENES_PER_TABLE, TRAIN_TABLES), ("test", N_TEST_SCENES_PER_TABLE, ALL_TABLES)):
        rows = [dict(scene_row(make_scene(t.name, split, i)), table_split=t.split) for t in tables for i in range(n)]
        pd.DataFrame(rows).to_csv(os.path.join(args.out, "scenes", f"{split}.csv"), index=False, float_format="%.7g")
    json.dump(dict(task=TASK.__dict__, stats=all_stats), open(os.path.join(args.out, "stats.json"), "w"), indent=1)
    md = stats_md(all_stats)
    open(os.path.join(args.out, "stats.md"), "w").write(md + "\n")
    configs = "\n".join(f"- config_name: K{K}\n  data_files:\n" + "\n".join(
        f"  - split: {hf}\n    path: K{K}/{s}.csv" for hf, s in (("train", "train"), ("validation", "val"),
                                                                 ("test", "test"))) for K in args.K)
    open(os.path.join(args.out, "README.md"), "w").write(CARD.format(configs=configs, stats=md))
    print(md)


if __name__ == "__main__":
    main()
