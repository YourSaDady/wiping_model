# wiping_model: scripted wiping program + demo collection

6.4212 Fall 2026, group 17 (Tang, He, Song). This covers the Oct 30 milestone: an impedance-controlled iiwa with a ball
tool, tables with adjustable tilt and contact stiffness `k_e`, logging of `x_t, ẋ_t, f_t, x_d,t, u_t`, and a scripted
program that wipes a flat table. The same program and the same configs are meant to be used for the Nov 6 goal
(reliable wiping on all training tables, collecting the demos, building the labels).

> **Status (Oct 28).** Tested on Drake 1.57 in the course venv (`64210/.venv`, Python 3.13). The scripted program
> wipes all three flat tables (`tests/test_pipeline.py --drake`) and 60/60 tables sampled from the training ranges.
> Results and what changed during debugging are under "Drake results" at the end.

## Quick start

```bash
cd wiping_model
source ../../64210/.venv/bin/activate                   # the course venv (drake 1.57) has everything for the scripts

python tests/test_pipeline.py --drake                    # unit tests + the three flat tables on Drake (~10 s)
python scripts/run_wipe.py --preset flat_medium          # one episode -> figures/flat_medium_drake.png
python scripts/run_wipe.py --preset flat_hard --meshcat  # watch it; replay with Meshcat's Animation controls
jupyter lab notebooks/wiping_demo.ipynb                  # demo notebook (needs ipykernel, see below)
```

The course venv has no `ipykernel`. To run the notebook, either `uv pip install ipykernel` into it and pick that
kernel in VS Code / Jupyter, or open it from any Jupyter that can use that venv as a kernel.

Every script also takes `--backend pointmass`, which runs in about 1 s per episode with no Drake. Use it to check
pipeline changes, not to get numbers.

## Layout

| path | what it is |
|---|---|
| `wiping/config.py` | `EpisodeConfig` = `TableConfig` (height, tilt, tilt_dir, `k_e`, dissipation, mu) + `RegionConfig` (L×L region) + `ControllerConfig` (K, D) + `ExpertConfig` + `SimConfig` + `SuccessConfig`. JSON round trip; nested overrides `cfg.replace(table__k_e=3e4)` |
| `wiping/presets.py` | Named tables (`flat_soft/medium/hard`, `tilt10_*`, `tilt15_soft`, held-out `heldout_*`) and the train/held-out sampling ranges |
| `wiping/drake_env.py` | Drake scene and the Cartesian impedance controller (a `LeafSystem`) |
| `wiping/pointmass_env.py` | Test double with the same interface: a point mass with the same impedance law against a Hunt-Crossley plane |
| `wiping/lanes.py` | Back-and-forth lane plan over the region, plus a constant-speed path follower |
| `wiping/expert.py` | `ScriptedWiper`, the scripted wiping program |
| `wiping/coverage.py` | Dirt grid and the uncovered fraction `u_t` |
| `wiping/rollout.py` | `run_episode(cfg, policy=None)` and the success metrics. It also runs closed-loop policies and open-loop label replays |
| `wiping/dataset.py` | Observations, labels (a)/(b)/(c)/(c′)/(d), `DemoWriter` / `load_demos` |
| `wiping/plotting.py`, `wiping/viz.py` | Episode figure (matplotlib); region, lanes and wiped-trail overlays (Meshcat) |
| `scripts/run_wipe.py` | Run one episode: plot, Meshcat, or save |
| `scripts/collect_demos.py` | Batch collection with a per-table reliability summary (multi-process) |
| `notebooks/wiping_demo.ipynb` | Demo, generated from `notebooks/make_notebook.py` |
| `tests/test_pipeline.py` | Tests |

## Scene and controller

- **Robot.** iiwa14 (`iiwa14_no_collision.sdf`) with a ball tool of radius 2 cm (`sim.tool_radius`). The tcp is the
  bottom of the ball. The controller holds the tool axis along `-n`, so the tcp is the contact point, on tilted tables
  too.
- **Table.** A box whose top surface passes through `c = (0.55, 0, height)` with normal
  `n = (sin θ cos φ, sin θ sin φ, cos θ)`. It uses Drake point contact. Each geometry gets `point_stiffness = 2 k_e`, so
  the series stiffness seen by the tool is exactly `k_e`. Hunt-Crossley dissipation is 20 s/m.
- **Table frame.** `t1` is world x projected onto the plane, `t2 = n × t1`. For a flat table this is just (x, y, z).
- **Controller.** Operational-space impedance in the stiffness form (Chapter 8):
  `τ = Jᵀ[K_rot e_R − D_rot ω ; K(x_d − x) − D ẋ] − τ_g + C v + null-space posture`.
  There is no inertia shaping, so in static contact `K(x_d − x) + f = 0` holds with the K we chose.
  - K and D are 3×3 world matrices built from table-frame gains: `K = R diag(K_t, K_t, K_n) Rᵀ`, with defaults
    `K_t = 1000`, `K_n = 500` N/m.
  - `D_t = 2ζ_t √(K_t m_eff)` with ζ_t = 1, sized for free space (316 N s/m).
  - `D_n = 2ζ_n √((K_n + k_e) m_eff)` with ζ_n = 0.7, capped at `D_n_max = 2000`. It is sized for the contact mode,
    because damping chosen for free space makes the tool bounce on stiff tables.
  - `m_eff = 25 kg`. This is what the iiwa's operational-space inertia `Λ = (J M⁻¹ Jᵀ)⁻¹` at the tcp actually is
    with the orientation held: about 15–30 kg along n and 20–90 kg tangentially over the region.
  - Both K and D are logged per episode.

## The scripted program (`wiping/expert.py`)

It runs at 20 Hz with a zero-order hold, the same rate a policy will run at. It only moves the impedance target, so the
logged `x_d` is exactly the command a policy would have to produce.

1. **Approach.** The episode starts with the tcp hovering 3 cm above the first lane start, and the program descends
   along `-n`. The target never goes deeper than the static depth for `0.5 f_des`, `−0.5 f_des (1/K_n + 1/k_e)`, and
   at contact (`f_n > 1 N`) it is re-anchored to `x − K⁻¹f`. Both remove the free-space tracking lag `D_n ẋ / K_n`
   (several cm with the contact-mode `D_n`), which would otherwise turn into a force spike at contact.
2. **Press.** This is the force loop the proposal's footnote asked for. It is an integral loop on the commanded depth
   along n, `h ← h − (f_des − f_n)/(K_eff τ) · dt`, with `K_eff = K_n k_e/(K_n + k_e)` (the controller and table springs
   in series) and τ = 0.25 s. `f_n` is read from MultibodyPlant's contact-results port. The phase ends once
   `|f_n − f_des| ≤ 10%` for 0.3 s.
3. **Wipe.** The program follows the lanes at speed `v` while the same loop holds the force. The lanes use spacing
   `s < 2r` and an edge margin `min(s/2, 0.6 r)`, so every point of the square is within `r` of the path. The lane
   path goes through a critically damped 2nd-order filter (`expert.ref_omega = 8 rad/s`), which rounds the corners
   and ramps up from rest. The target leads that reference by `(μ f_n dir(v) + D_t v) / K_t`, the tool's steady lag
   while sliding (friction plus damping on absolute velocity). The phase ends once the reference has reached the last
   waypoint.
4. **Lift.** The program raises the target to 4 cm, holds for 0.3 s, and then the episode is done.

**Coverage.** A 5 mm dirt grid. A cell is cleaned when the contact point passes within `r = 1.5 cm` with
`f_n ≥ f_min`. Coverage is computed along segments between consecutive samples, so 20 Hz sampling leaves no gaps.

**Success** (from the proposal): `u_T ≤ ε_u = 0.1` and `max_t f_n ≤ f_max = 2 f_des`, with `f_min = 0.5 f_des`. For a
scripted run the plan must also have finished.

## Data format (`DemoWriter` → `.npz`)

Per-step arrays are concatenated over episodes. `episode_ends[i]` is the exclusive end row of episode i, as in
Diffusion Policy, and `ep_index` maps each row to its episode. Row t of an episode:

| key | shape | meaning |
|---|---|---|
| `obs` | N×16 | `o_t = (x_t, ẋ_t, f_t, p0, L, n)`; `p0` = region corner (min u, min v) on the surface |
| `t, x, xdot, f, f_n, q, v` | per step | raw state at step t (`f` = force of the table on the tool, world frame) |
| `x_d` | N×3 | commanded target held over [t, t+1) |
| `u` | N | uncovered fraction after sample t |
| `x_next, xdot_next, f_next, u_next` | per step | sample t+1, the one the command produced |
| `phase` | N | 0 approach, 1 press, 2 wipe, 3 lift, 4 done. Expert-internal: keep it out of the observation |
| `label_a_measured` | N×3 | `x_{t+1}` |
| `label_b_commanded` | N×3 | `x_{d,t}` |
| `label_c_force_comp` | N×3 | `x_{t+1} − K⁻¹ f_{t+1}` |
| `label_c_damp` | N×3 | `x_{t+1} + K⁻¹(D ẋ_{t+1} − f_{t+1})` |
| `label_d_pose_force` | N×6 | `[x_{t+1}, f_{t+1}]` (stretch goal: target `x − K′⁻¹ f`) |
| `K, D` | E×3×3 | per-episode world stiffness and damping |
| `p0, L, n, k_e, tilt, mu, f_des, speed, success, u_T` | E | per-episode parameters |
| `meta` | JSON | `[{config, metrics}]` per episode; `EpisodeConfig.from_dict(meta[i]["config"])` reproduces the episode |

Signs: f is the force on the tool (+n while pressing), so the robot pushes the table with −f.

## Collecting

```bash
# reliability check on the named tables (5 jittered episodes each)
python scripts/collect_demos.py --mode presets --per-preset 5 --workers 6

# training demos from the sampled ranges (failures are dropped and listed in the summary json)
python scripts/collect_demos.py --mode sample --split train --n 200 --workers 6 --out data/train.npz
python scripts/collect_demos.py --mode sample --split heldout --n 40 --out data/heldout_scenes.npz
```

- **Training ranges** (`presets.TRAIN_RANGES`): tilt 0–15°, `k_e` log-uniform 2e3–5e4, μ 0.3–0.7, L 0.10–0.14 m,
  v 0.04–0.08 m/s, region center ±3 cm, height 0.08–0.12 m.
- **Held-out ranges:** tilt 18–25°, `k_e` 5e4–1e5.

## Using it from other code

```python
from wiping import make_config, run_episode, make_labels
cfg = make_config("tilt10_hard", table__mu=0.3, expert__speed=0.05)
ep = run_episode(cfg)                         # scripted program
ep = run_episode(cfg, policy=my_policy)       # closed loop: my_policy(obs) -> x_d, with obs = dict(x, xdot, f, p0, L, n, t)
ep = run_episode(cfg, policy=labels["label_a_measured"])   # open-loop replay of a target sequence
```

## Tuning / debugging on Drake

If something misbehaves on a new table, in order:

1. **Look at the episode figure** (`scripts/run_wipe.py --preset ... --set ...`). It shows the force with phases, the
   tool vs. targets along n, the top view with cleaned cells, and the coverage/speed.
2. **Force ringing in contact.** Raise `controller__zeta_n` or `controller__D_n_max`. The controller is explicit at
   1 kHz; D up to 3000 ran stably.
3. **Force loop.** If `f_n` oscillates around `f_des`, increase `expert__force_tau` (0.25 → 0.5). If it converges too
   slowly, decrease it.
4. **Force dips at lane corners.** Lower `expert__ref_omega` (8 → 5) or the speed. Lower ω cuts corners a bit more.
5. **Coverage.** If lane ends are missed, check the top view; try a smaller `expert__lane_spacing`.
6. **IK at tilts.** `solve_ik` raises if the hover pose is unreachable. A steep tilt facing the base (tilt_dir ≈ 150°,
   24°) is really out of reach (300 random IK restarts fail). Move `table__center_xy` or limit `tilt_dir`.
7. **Wall time.** About 2 s of wall time per 20 s episode (the Python controller runs at 1 kHz). `--workers 8` collects
   60 episodes in about 17 s.

## Drake results (Oct 28, drake 1.57, defaults)

| test | result |
|---|---|
| flat_soft / flat_medium / flat_hard (k_e 2e3 / 1e4 / 5e4) | all succeed, u_T = 0, wipe f_n = 10.02 ± 0.4–0.5 N, max f_n ≤ 11.1 N |
| all 8 presets, incl. held-out 25° tilt and k_e = 1e5 | 8/8 |
| hardest corner cases (μ = 0.7, v = 0.08 m/s, k_e = 5e4) | succeed, min wipe f_n 7.1 N (f_min 5), max 12.2 N (f_max 20) |
| 30 sampled flat tables (train ranges) | 30/30 |
| 60 sampled training tables (tilt 0–15°) | 60/60 |
| 30 sampled held-out tables (tilt 18–25°, k_e 5e4–1e5) | 29/30; the failure is IK (unreachable start), not wiping |

**Label checks on flat_medium** (notebook §3–4). Median |label − x_d| in wipe: (a) 29.9 mm, (c) 16.9 mm, (c′) 5.1 mm.
Most of the (c) gap is tangential: the controller damps absolute velocity, so `D_t v / K_t ≈ 19 mm` at 6 cm/s, and
(c′) recovers it (P2). Open-loop replay: (a) drops the wipe force to 0.47 N, against the predicted `K/(K+k_e)` × 10 N =
0.48 N, and cleans nothing. (b) gives 10.02 N, (c) 9.94 N, (c′) 9.98 N (P1 preview).

**What changed during the first Drake runs.** The original defaults also succeeded on every preset, but the force
dipped to about 1.6 N at lane corners for μ = 0.7, v = 0.08 m/s on hard tables, which would fail on the sampled
range. Causes and fixes:

1. `m_eff = 3 kg` was 10× too small, so the real damping ratios were about 0.2 and the tool rang on every corner.
   It is now 25 kg (measured Λ), with ζ_n = 0.7 and `D_n_max` 450 → 2000.
2. The tangential reference turned 90° and started from rest in one step, and each friction flip kicked the normal
   force through the arm's coupled inertia. It is now smoothed by a 2nd-order filter (`ref_omega`), and the lead also
   includes `D_t v / K_t`.
3. The contact-onset spike came from the free-space lag `D_n v / K_n`. The approach target is now floored at the
   static depth for 0.5 f_des.

The cost is a slower approach: with D_n sized for contact, the free-space normal motion is overdamped, and contact
takes about 3–5.5 s instead of about 2.5 s.
