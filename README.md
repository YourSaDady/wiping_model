# Force-aware wiping with a Cartesian-impedance iiwa (MIT 6.4212, Fall 2026)

Team: Yanchuan Tang, Linda Yinghui He, Yichuan Song.

This branch is the **Oct 30 milestone** snapshot: the Drake environment and the scripted wiping
program. 

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
- `samples` (100 Hz: t, region coordinates u1 u2, gap, f_n, friction power P_fric) used by the
  coverage metric, and `metrics`.

### Scripted wiping program

Approach (2 cm/s, slowing to 6 mm/s within 2 cm of the surface) → press (integrator on the force
error) → sweep a 6-lane boustrophedon path over the L × L region (path speed drops to 0.35 v
within 2 cm of each corner) → lift (until the ball is 2 cm above the surface) → return (the
target moves straight back to the start position at 5 cm/s, and the episode ends once the ball has
settled there). The target stays below the surface by f/K so the spring produces the desired 10 N.
The expert regulates the **normal force** (f_des = 10 N on every demo); the target depth is whatever
that force needs on the given table. Success is defined in the next section.

The return phase brings the arm back to its start pose: the ball ends within 4.3 mm of the start
position, and the joints within 2.5° of q_0 (median 1.3°), since the orientation and the weak
null-space posture are held. That was checked on 3 dev scenes × 17 tables, where all 51 episodes
succeeded and finished. It adds 2.8–3.9 s, and the longest episode is now 43 s, under T_MAX = 50 s.
`ScriptedExpert(..., return_home=False)` restores the old ending.

**Scenes.** A *table* (e.g. `tilt20_ke1e5`) fixes the surface: tilt and k_e. A *scene* is one
randomized episode setup on that table (`wipe/scene.py: make_scene(table, split, index)`). It fixes
the table's position and height, the wiping region (corner p0, side L), the wiping speed v, and the
start pose. The scene is drawn deterministically from the seed of (table, split, index), so the same
triple always gives the same scene. The splits `demo`, `val`, `test` and `dev` use disjoint seeds:
demonstrations, validation, the final test and debugging never share a scene.

### Success criterion

A trial **succeeds** if, at the end of the episode,

1. the **uncovered fraction u_T ≤ ε_u = 0.1**, and
2. the **normal force never exceeded f_max = 20 N** (checked at 1 kHz, `PeakForceMonitor`).

The uncovered fraction comes from a 2 mm grid over the L × L region (`wipe/coverage.py`). Each cell
collects a **wiping dose**, the friction work done while the cell was under the wiping patch:

$$W_\text{cell} = \sum_{k\,:\;\|c_k - \text{cell}\| \le r} P_k\,\Delta t, \qquad
P_k = \max\bigl(0,\; -f_{t,k}\cdot v_{t,k}\bigr)$$

- c_k is the contact point (ball centre projected onto the undeformed surface), sampled at 100 Hz,
  and r = r_wipe = 2 cm is the radius of the virtual wiping patch.
- f_t is the tangential part of the measured contact force on the tool, and v_t is the tangential
  velocity of the ball's contact point. So P_k is the power dissipated by friction, |f_t|·|v_t| while
  sliding.
- A cell counts as **wiped** once **W_cell ≥ w_min = 0.06 J**, and u_T is the fraction of cells that
  are not.

**Threshold.** w_min = μ · f_min · 2r = 0.3 × 5 N × 4 cm. In words, a cell is clean once the patch
has slid one full patch diameter over it at the minimum force f_min = 5 N. The value is calibrated
against the proposal's f_min: the expert was run with a lowered force set point f_des on 8 tables ×
3 scenes, and its success rate under each candidate w_min was:

| expert f_des | 3 N | 4 N | 5 N | 6 N | 7 N | 10 N |
|---|---|---|---|---|---|---|
| w_min = 0.03 J (μ f_min r) | 0.50 | 0.88 | 0.92 | 1.00 | 1.00 | 1.00 |
| **w_min = 0.06 J (μ f_min 2r)** | **0.00** | **0.00** | **0.00** | **0.58** | **0.88** | **1.00** |

With 0.03 J the lanes overlap enough that wiping at 3–4 N passes, which would hide the force loss
that P1 is about. With 0.06 J, wiping at f_n ≤ f_min fails, as in the proposal's rule.

Why friction work and not a force threshold. Removing dirt takes pressure **and** sliding: by
Archard's wear law, worn volume ∝ normal load × sliding distance, and friction work is μ times that.
So this metric

- gives no credit for pressing in place (the press phase) or for sliding without load;
- lets a light pass be made up by passing again, while a pass that is too fast or too light does not
  finish a cell;
- uses the measured contact force, so it needs no knowledge of μ or of the table.

The proposal's original rule is still available: set `coverage_mode="force"` in `TaskParams`
(`wipe/config.py`). Under that rule a cell is wiped at the first 100 Hz sample with f_n ≥ f_min that
puts it under the patch. The force bound (2) is the same in both modes.

Check with the work criterion (Oct 29, 5 dev scenes × 17 tables): the scripted expert succeeds on
85/85, with worst u_T = 0.008. The cells it reaches get 1.1–1.7 w_min at the 5th percentile; the
lowest margin is on the k_e = 3e5 OOD table. The 50-scene numbers under "Results at this milestone"
were computed with the original force rule.

## Scripted demo: live view and failure cases

`scripts/watch_expert.py` runs the scripted expert on one scene with a live Meshcat view (robot,
region, planned lanes, wiped cells in white; replayable afterwards) and a live top view of the
region. In the top view each cell is coloured by W_cell / w_min: **any pink shade means not wiped
yet** (deep pink = no dose, paler = cleaner), and **white means wiped**. The dirty region is drawn in
the same pink in Meshcat. The panel next to it
lists the scene parameters and the live success checks (u vs ε_u, 1 kHz peak f_n vs f_max, time vs
the 50 s limit). The final verdict and its reasons are shown there and printed.

```bash
python scripts/watch_expert.py [table] [index] [options]     # default: tilt00_ke2e4, dev scene 0
```

| option | meaning | default |
|---|---|---|
| `table index` | preset table (`wipe/config.py` TABLES) and scene index; fixes position, height, start pose | `tilt00_ke2e4 0` |
| `--split` | scene split (`demo`, `val`, `test`, `dev`) | `dev` |
| `--tilt` | table tilt about world x [deg] | from the preset |
| `--ke` | contact stiffness k_e [N/m] | from the preset |
| `--mu` | Coulomb friction of ball and table | 0.3 |
| `--v` | wiping speed along the lanes [m/s] | sampled per scene (0.05–0.12) |
| `--fdes` | normal force the expert holds [N] | 10 |
| `--L` | side of the square region [m] | sampled per scene (0.13–0.17) |
| `--K` | controller normal stiffness K_n [N/m] (250 / 500 / 1000) | 500 |
| `--rate` | real-time factor (0 = as fast as possible) | 1 |
| `--no-meshcat`, `--no-topview`, `--save-topview PNG` | display / output | |

The options change the scene or the expert, never the success test (f_min, f_max, ε_u, w_min,
r_wipe and the 50 s limit stay the task's). Ball "hardness" is not a separate option: in point
contact the ball and the table are two springs in series and only their combined stiffness k_e
enters the dynamics (`DECISIONS.md`), so `--ke` covers both.

**Failure cases.** Each command changes one thing from the nominal scene (`tilt00_ke2e4`, dev 0:
L = 15.6 cm, v = 0.066 m/s). Results are from Oct 29 runs with the default task parameters.

| case | command | u_T | peak f_n | result and cause |
|---|---|---|---|---|
| nominal | `python scripts/watch_expert.py` | 0.001 | 10.6 N | success |
| force too low | `python scripts/watch_expert.py --fdes 4` | 0.89 | 4.5 N | **fail** (coverage): friction work per pass below w_min; cells stay pink |
| force at f_min | `python scripts/watch_expert.py --fdes 5` | 0.58 | 5.5 N | **fail** (coverage): borderline cells between lanes are not finished |
| force barely enough | `python scripts/watch_expert.py --fdes 6` | 0.045 | 6.6 N | success (control case: just above the threshold) |
| force too high | `python scripts/watch_expert.py --fdes 25` | 0.00 | 25.6 N | **fail** (force): the region is wiped but f_n > f_max |
| table too hard | `python scripts/watch_expert.py --ke 1e6` | 0.00 | 24.7 N | **fail** (force): impact spike at contact, ≈ v·√(k_e m_eff) |
| table too soft | `python scripts/watch_expert.py --ke 100` | 1.00 | 8.5 N | **fail** (coverage + time): 10 N needs 10 cm of penetration, so the force never builds up |
| moving too fast | `python scripts/watch_expert.py --v 0.4` | 0.55 | 25.8 N | **fail** (both): the arm cuts corners, leaves gaps, and hits the surface when it reverses |
| moving too slow | `python scripts/watch_expert.py --v 0.01` | 0.72 | 10.1 N | **fail** (time): only part of the lanes is done in 50 s |
| slippery surface | `python scripts/watch_expert.py --mu 0.05` | 1.00 | 10.5 N | **fail** (coverage): full force but almost no friction work; would pass under the old force rule |
| region too large | `python scripts/watch_expert.py --L 0.30` | 0.40 | 10.9 N | **fail** (coverage): 6 lanes at s = 6 cm > 2r leave unwiped strips |
| very steep table | `python scripts/watch_expert.py --tilt 60` | 0.00 | 10.9 N | success: not a failure for the expert, since gains are in the table frame (IK works up to at least 70°) |

Add `--no-meshcat --rate 0 --save-topview out.png` to run a case without the windows.

## Data

### Per-episode file

`scripts/gen_demos.py` writes one file per successful expert episode,
`data/demos/K{K_n}/{table}/{index:03d}.npz` (`wipe/data.py: save_episode / load_episode`). An
episode has T policy steps at 20 Hz. States have T+1 rows (t_0..t_T); commands have T rows (the
target held over [t_k, t_{k+1})).

| key | shape | meaning |
|---|---|---|
| `t` | (T+1,) | time [s] |
| `x` | (T+1, 3) | ball-centre position (the controlled point), world frame [m] |
| `xdot` | (T+1, 3) | ball-centre velocity [m/s] |
| `f` | (T+1, 3) | contact force of the table on the tool, world frame [N] (+n while pressing) |
| `q` | (T+1, 7) | iiwa joint angles |
| `u` | (T+1,) | uncovered fraction of the region so far |
| `x_d` | (T, 3) | commanded impedance target (what the expert sent) |
| `phase` | (T,) int8 | expert phase: 0 approach, 1 press, 2 sweep, 3 lift, 4 return (not an observation) |
| `samples` | (N, 6) | 100 Hz coverage samples: t, u1, u2 (region coords), gap to surface, f_n, friction power |
| `metrics` | JSON | success, uncovered, peak_fn, mean_fn_wipe, ... (`WipeEnv.metrics`) |
| `scene` | JSON | the `Scene`: table, tilt, k_e, p0, L, t1, t2, n, v, start (reproduces the episode) |
| `expert_done` | bool | the expert finished its plan |

**Observation** (same for every label, `wipe/data.py: obs_vector`, 16 numbers):
o_t = (x_t, ẋ_t, f_t, p0, L, n̂).

### Action labels

All labels are built from the **same** logged episode, when the training windows are made
(`wipe/labels.py: make_labels(x, xdot, f, x_d, K, D, which)`); nothing label-specific is stored. K
and D are the controller's 3×3 world-frame stiffness and damping for that episode,
`Gains.K(n)` and `Gains.D(n)` of `GAINS_BY_KN[K_n]`.

| label | in the text | code name (`which`, `--label`) | formula | meaning |
|---|---|---|---|---|
| (a) | measured pose | `"a"` | x_{t+1} | where the ball actually went: what most datasets log |
| (b) | commanded target | `"b"` | x_{d,t} | what the controller was told; the target sits below the surface by about f/K |
| (c) | force-compensated target | `"c"` | x_{t+1} − K⁻¹ f_{t+1} | quasi-static reconstruction of (b) from pose and force sensing |
| (c′) | force + damping compensated | `"cp"` | x_{t+1} + K⁻¹ (D ẋ_{t+1} − f_{t+1}) | adds the damping term the static formula ignores (P2) |

Training (`wipe/dp/dataset.py`, `scripts/train_dp.py`): windows of To = 2 observations and H = 16
labels. By default the labels are relative to the current position (a_j = label_{t+j} − x_t); the
policy executes 8 of the 16. One model per (label, K_n, seed):
`python scripts/train_dp.py --label b --K 500 --seed 0 --out runs/K500_b_s0`.

### Week 2 (Nov 6) collection plan

Demonstrations are collected on the 12 **training tables**. Every combination of tilt × k_e is
one setting:

| variable | values | where |
|---|---|---|
| table tilt θ | 0°, 10°, 20° | `TRAIN_TILTS` |
| contact stiffness k_e | 500, 2e3, 2e4, 1e5 N/m | `TRAIN_KE` |
| controller K_n (separate datasets) | **500** (main); 250 and 1000 for the K sweep (P1, P3) | `GAINS_BY_KN` |

| dataset | settings | demos per setting | total | command |
|---|---|---|---|---|
| main | 12 tables at K_n = 500 | 36 (32 train + 4 val) | 432 | `python scripts/gen_demos.py --K 500 --n 36 --workers 16` |
| K sweep | 12 tables at K_n = 250 | 36 | 432 | `python scripts/gen_demos.py --K 250 --n 36 --workers 16` |
| K sweep | 12 tables at K_n = 1000 | 36 | 432 | `python scripts/gen_demos.py --K 1000 --n 36 --workers 16` |

Each run writes `data/demos/K{K_n}/manifest.json` with the kept and failed scene indices. Failed
expert episodes are skipped and never used for training. Each demo yields all four labels, so
432 demos give four training sets of 432 episodes that differ only in the label. Use
`--workers` ≤ the number of cores (the original setup used 28). One episode takes about 2–3 s of
CPU time, so a K_n setting takes a few minutes.

**Varied per demo** (drawn from the seed of (table, `demo`, index), `wipe/scene.py`):
region centre x ∈ U[0.52, 0.60] m, y ∈ U[−0.04, 0.04] m, height ∈ U[0.06, 0.16] m; region side
L ∈ U[0.13, 0.17] m; wiping speed v ∈ U[0.05, 0.12] m/s; start height 4–7 cm above the surface with
±1 cm in-plane jitter.

**Shared by every demo:** f_des = 10 N; μ = 0.3; Hunt–Crossley dissipation 20 s/m; ball R = 2 cm;
1 ms plant step and 20 Hz expert and logging with zero-order hold; K_t = 1000, D_t = 110, D_n = 250,
K_rot = 100; 6 lanes with the corner slowdown; time limit T_MAX = 50 s; success test as above
(ε_u = 0.1, f_max = 20 N, w_min = 0.06 J, r_wipe = 2 cm). Held-out (OOD) tables are never used for
demos; validation and test scenes use the `val` / `test` splits, so they never share a seed with
demos (10 val and 50 test scenes per table, `wipe/config.py`).

## Results at this milestone

- `pytest tests/`: 9 tests pass (label formulas and time alignment, table-frame gains, lane path
  coverage, force-mode coverage requires f ≥ f_min, work-mode coverage needs sliding friction work,
  force sign, springs in series f_n = K_eff δ within 1 %).
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
uv venv --python 3.12 .venv && source env.sh      # uv downloads 3.12 if needed; source from this directory
uv pip install drake==1.58.0 numpy scipy matplotlib pandas tqdm pytest
pytest tests/
python scripts/try_expert.py tilt00_ke2e4 0      # one scripted episode, prints metrics per phase
python scripts/watch_expert.py tilt00_ke2e4 0    # same, live in Meshcat + live top view (options and failure cases above)
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
