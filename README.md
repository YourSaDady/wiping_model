# Force-aware wiping with a Cartesian-impedance iiwa (MIT 6.4212, Fall 2026)

Team: Yanchuan Tang, Linda Yinghui He, Yichuan Song.

This branch is the **Oct 30 milestone** snapshot: the Drake environment and the scripted wiping
program. The full pipeline (demonstrations, Diffusion Policies for the four action labels,
baselines, evaluation, figures) is on branch `yanchuan/full-pipeline`.

## Oct 30 milestone checklist

| Milestone item | Where | Status |
|---|---|---|
| Impedance-controlled iiwa with the ball tool | `wipe/sim.py` (plant, tool, contact), `wipe/controllers.py` (`ImpedanceController`), gains in `wipe/config.py` | done |
| Tilted tables with adjustable k_e | `wipe/config.py` (`TableSpec`, `TRAIN_TABLES`, `OOD_TABLES`), `wipe/scene.py` (seeded scenes) | done |
| Logging of x_t, ẋ_t, f_t, x_{d,t}, u_t | `wipe/rollout.py` (episode dict, 20 Hz), example log `results/milestone/flat_episode.npz` | done |
| Scripted wiping program wipes a flat table | `wipe/expert.py`, `wipe/paths.py`, `wipe/coverage.py`; figure + videos below | done (also on tilted tables) |

### Environment

- KUKA iiwa14 (`drake_models`, no collision geometry) with a rigid ball tool: radius R = 2 cm,
  0.3 kg, centre 0.17 m along `iiwa_link_7` +z.
- Discrete `MultibodyPlant`, 1 ms step, point contact; ball–table stiffness k_e (each geometry gets
  2 k_e, which combine in series to k_e), Hunt–Crossley dissipation 20 s/m, Coulomb μ = 0.3.
- Tables: tilt about world x, any tilt and k_e. Training set tilt ∈ {0°, 10°, 20°} ×
  k_e ∈ {500, 2e3, 2e4, 1e5} N/m; held-out tables (15°, 5e4), (28°, 2e4), (28°, 500), (0°, 250), (10°, 3e5).
- Scenes (table height and position, region corner p0 and side L ∈ [13, 17] cm, wiping speed,
  start pose) are drawn from `SeedSequence([crc32(table), split, index])`; demo/val/test/dev
  splits never share a seed.

### Controller

τ = Jᵀ [K_rot e_R − D_rot ω ; K (x_d − x) − D ẋ] − τ_g + N τ_null, with translational K, D diagonal in
the table frame (t1, t2, n): K_t = 1000 N/m, K_n = 500 N/m (250 and 1000 also supported), D_t = 110,
D_n = 250 N·s/m, K_rot = 100, D_rot = 10, null-space K = 5, D = 1. Every choice and the reason for it
is in `DECISIONS.md`.

### Logged signals (20 Hz, zero-order hold of x_d)

`run_expert` / `replay_targets` / `run_hybrid` in `wipe/rollout.py` return

- `t`, `x` (ball centre), `xdot`, `f` (contact force on the tool, +n when pressing), `q` (joints),
  `u` (online uncovered fraction of the region), each with T+1 rows;
- `x_d` (commanded impedance target, T rows) and the expert `phase`;
- `samples` (100 Hz: t, region coordinates, gap, f_n) used by the coverage metric, and `metrics`.

### Scripted wiping program

Approach (2 cm/s, slowing to 6 mm/s within 2 cm of the surface) → press (integrator on the force
error) → sweep a 6-lane boustrophedon path over the L × L region (path speed drops to 0.35 v
within 2 cm of each corner) → lift. The target stays below the surface by f/K so the spring
produces the desired 10 N. Success: uncovered fraction ≤ 0.1 (2 mm grid, wiping patch r = 2 cm,
a sample counts only if f_n ≥ 5 N) and the 1 kHz normal force never above 20 N.

## Results at this milestone

- `pytest tests/`: 8 tests pass (label formulas and time alignment, table-frame gains, lane path
  coverage, coverage requires f ≥ f_min, force sign, springs in series f_n = K_eff δ within 1 %).
- Scripted program on 50 dev scenes per table (`scripts/phase1_checks.py --n 50`,
  `results/phase1/expert_check_summary.csv`): success 50/50 on every table and gain tested. Flat
  tables at K_n = 500:

  | table | success | max peak f_n [N] | mean f_n while wiping [N] |
  |---|---|---|---|
  | tilt00_ke500 | 50/50 | 11.47 | 8.68 |
  | tilt00_ke2e3 | 50/50 | 11.50 | 8.96 |
  | tilt00_ke2e4 | 50/50 | 12.37 | 9.02 |
  | tilt00_ke1e5 | 50/50 | 12.41 | 9.03 |
  | tilt00_ke250 (held out) | 50/50 | 11.20 | 8.35 |

- One logged flat-table episode (tilt00_ke2e4, dev scene 0): `results/figures/milestone_flat_log.png`
  — region covered completely (u = 0), peak 1 kHz f_n = 10.6 N, f_n ≈ 10 N during the sweep.
- Videos of the scripted program on flat tables (2× real time; Drake camera, normal force,
  coverage map): `results/videos/tilt00_ke500_000_scripted.mp4` (soft, k_e = 500 N/m) and
  `results/videos/tilt00_ke1e5_000_scripted.mp4` (hard, k_e = 1e5 N/m).
- Open-loop replay of the action labels on flat tables: `results/figures/phase1_replay_force.png`.

## Run

```bash
uv venv --python /usr/bin/python3.12 .venv && source env.sh
uv pip install drake==1.58.0 numpy scipy matplotlib pandas tqdm pytest
pytest tests/
python scripts/try_expert.py tilt00_ke2e4 0      # one scripted episode, prints metrics per phase
python results/fig_milestone_log.py             # logged-signal figure + results/milestone/flat_episode.npz
python scripts/phase1_checks.py --n 50          # expert on 50 dev scenes per table (~12 min, 28 workers)
python results/fig_phase1_replay_force.py
```

The snapshot also contains code that was already started at this point and is finished on
`yanchuan/full-pipeline`: the hybrid force/position baseline, action labels, demo generation and
Diffusion Policy training/evaluation scripts. `STATUS.md` and `DECISIONS.md` are as of this snapshot.

## Attribution

- Drake (Tedrake et al.) for simulation and IK; the iiwa14 model from `drake_models`.
- Controllers follow the course textbook (Tedrake, *Robotic Manipulation*, Ch. 8).
- The code was written with an AI coding agent (Claude, Anthropic) from the team's design brief;
  the use log is `AI_USE.md` on `yanchuan/full-pipeline`.
