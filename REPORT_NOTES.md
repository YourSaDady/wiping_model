# Report notes (facts, numbers, figures — not report text)

**Read first.** The course communication policy (GAI-Comm.pdf) forbids GAI-generated report
text and oral presentations. These notes are therefore bullet points, numbers, tables and
pointers to figures, produced by the AI agent from scripts in this repo. The team writes the
report and narration in its own words and decides what to include. Every number has its
source script next to it. Report requirements (project page): IEEE conference template,
≈ 4500 words for 3 people (figures/references excluded); abstract, introduction, related work,
approach, results with a comparison of approaches, discussion/next steps, per-member
contributions, overlap disclosure, AI-use description (see `AI_USE.md`).

## Suggested outline (headings only)

1. Abstract
2. Introduction — the missing-force problem under impedance control; the question; contributions
3. Related work — DexForce, Adaptive Compliance Policy, Tune to Learn, Beyond State-as-Action, Diffusion Policy, ACT (proposal refs)
4. Approach
   4.1 Simulation and tables  4.2 Cartesian impedance controller  4.3 Scripted expert
   4.4 Labels (a)–(c′) and time alignment  4.5 Diffusion Policy  4.6 Baselines  4.7 Metric and statistics
5. Predictions P1–P4 (1-D two-springs model)
6. Results
   6.1 Expert and label replays (Phase 1)  6.2 Main comparison (ID / OOD)  6.3 P1  6.4 P2  6.5 P3  6.6 P4
   6.7 Stretch results (if included)
7. Discussion, limitations, next steps
8. Contributions, overlap statement, AI use, acknowledgments (GAI disclosure)

## Setup facts

- Drake 1.58 (pip), iiwa14 (no collision), ball tool R = 2 cm at 0.17 m from link 7, mass 0.3 kg.
- Discrete plant 1 ms; point contact, lagged approximation; pair stiffness k_e (2k_e per geometry);
  Hunt–Crossley 20 s/m; μ = 0.3.
- Controller: Cartesian stiffness (operational-space form, no inertia shaping), table-frame gains
  K_t = 1000, K_n ∈ {250, 500, 1000} N/m; D_t = 110, D_n = 250 N·s/m; K_rot = 100 N·m/rad, D_rot = 10;
  null-space joint centring K = 5, D = 1. Effective mass along n ≈ 16–19 kg.
- Task: f_des = 10 N, f_min = 5 N, f_max = 20 N (1 kHz, impacts included), ε_u = 0.10; r = 2 cm;
  L ∈ [0.13, 0.17] m, 6 lanes, s = L/5; v ∈ [5, 12] cm/s with corner slowdown to 0.35 v.
- Tables: train tilt {0, 10, 20}° × k_e {500, 2e3, 2e4, 1e5}; OOD (15°, 5e4), (28°, 2e4), (28°, 500), (0°, 250), (10°, 3e5).
- Data: 36 successful expert demos per training table per K (32 train / 4 val), 20 Hz.
- Policy: DP (Chi et al.) 1D U-Net 65.7M params, T_o = 2, H = 16, execute 8, DDPM 100 / DDIM 16,
  relative targets, 40k steps, checkpoint chosen on 120 closed-loop validation scenes.
- Evaluation: 50 test scenes × 17 tables per method, T_MAX = 50 s, identical DDIM noise per scene
  across methods (paired trials).

## Numbers so far

- Phase 1 (`scripts/phase1_checks.py`): expert 50/50 on every table and K; peak ≤ 15.6 N (train), 16.8 N (OOD 3e5).
  Replay: (b) exact; (a) force ratio 0.494 / 0.199 / 0.042 / 0.027 vs K/(K+k_e) = 0.5 / 0.2 / 0.024 / 0.005
  (k_e = 500 / 2e3 / 2e4 / 1e5, K_n = 500); (c) 99–100 % force in sweep but 27–30 N onset peaks on k_e = 1e5;
  (c′) succeeds in all replays.
- P2 offline (`results/p2_label_gap.py`): (c) in-plane gap 6.3 → 13.7 mm for 4.6 → 12.5 cm/s ≈ D_t v/K_t;
  (c′) 3.4–5.2 mm. Inertia check (`results/p2_inertia_check.py`, 9 demos on 3 tables): subtracting K⁻¹[Λa]_trans
  (a = 6-D TCP acceleration at t_{k+1}⁻ from 1 kHz stepping) reduces the (c′) residual in the sweep
  6.66 → 2.74 mm (corr. 0.83) and in the approach 0.52 → 0.22 mm.
- NN smoke test (val, K_n = 500): (a) 0, (b) 115, (c) 71, (c′) 90 of 120.
- Baselines on val: scripted 170/170, hybrid 166/170 (OOD k_e = 3e5 impacts).

## Final results (test split, 50 scenes × 17 tables per method; `results/table_main.py`)

K_n = 500, DP rows pool seeds 0–2 (1800 ID / 750 OOD episodes); baselines 600 / 250.
Full table with coverage, force and peaks: `results/tables/main_500.md`.

| method | ID success [95 % CI] | OOD success [95 % CI] | ID mean f_n while wiping |
|---|---|---|---|
| scripted (privileged) | 100 % [99, 100] | 100 % [98, 100] | 8.92 N |
| hybrid force/position | 100 % [99, 100] | 92 % [88, 95] (all 19 fails on k_e = 3e5) | 9.36 N |
| DP (a) x | 0 % [0, 0] (0/1800) | 0 % [0, 1] | 0.47 N |
| DP (b) x_d | 99.6 % [99, 100] (1792/1800) | 52 % [48, 56] | 9.03 N |
| DP (c) x − K⁻¹f | 7 % [6, 8] (129/1800) | 4 % [3, 6] | 9.01 N |
| DP (c′) x + K⁻¹(Dẋ − f) | 24 % [23, 26] (440/1800) | 28 % [25, 31] | 7.76 N |

- Per seed (ID / OOD): (b) 0.997/0.548, 0.995/0.552, 0.995/0.460; (c) 0.065/0.036, 0.043/0.028, 0.107/0.064;
  (c′) 0.197/0.300, 0.277/0.328, 0.260/0.204; (a) 0 everywhere.
- All 6 label pairs differ on ID tables (Holm-corrected Fisher p ≤ 3e-40, McNemar seed 0 p ≤ 1e-11); on OOD tables all pairs p ≤ 0.004 — `results/tables/label_tests_500.csv`.
- Controller-stiffness sweep, seed 0, ID success (`results/tables/ksweep_success.csv`):

| label | K_n = 250 | 500 | 1000 |
|---|---|---|---|
| (a) | 0/600 | 0/600 | 0/600 |
| (b) | 598/600 | 598/600 | 599/600 |
| (c) | 1/600 | 39/600 | 460/600 |
| (c′) | 301/600 | 118/600 | 195/600 |

  OOD (seed 0): (b) 96 / 137 / 202 of 250; (c) 0 / 9 / 51; (c′) 43 / 75 / 73.

### P1 (`results/fig_p1_force_ratio.py`, `results/tables/p1_force_ratio.csv`)
- Open loop (Phase 1 replay): (a) keeps K/(K+k_e) of the force on soft tables (0.494 vs 0.500; 0.199 vs 0.200; K_n = 1000: 0.662 vs 0.667).
- Closed loop, K_n = 500, ID: policy/expert force ratio of (a) = 0.078, 0.068, 0.033, 0.031 at k_e = 500, 2e3, 2e4, 1e5 vs predicted 0.50, 0.20, 0.024, 0.005.
  (a) never succeeds on any table at any K_n, including where K/(K+k_e)·f_des ≥ f_min (k_e = 500 at K_n = 500/1000).
  Reason: (a) hardly reaches the table (P4), so the static two-spring bound is not what limits it in closed loop.
- (b) keeps 101 % of the expert's force at every K_n; (c) 101–103 % at K_n = 500/1000 once in contact
  (5.5 % at K_n = 250, where it barely touches); (c′) 85–89 %.

### P2 (`results/p2_label_gap.py`, `results/p2_inertia_check.py`, `results/p2_velocity_gain.py`, `results/fig_p2_closed_loop.py`)
- Offline gap (c) − (b) in straight sweeps: 6.3 → 13.7 mm for 4.6 → 12.5 cm/s ≈ D_t v/K_t (dropped damping).
  (c′) 3.4–5.2 mm; the remainder is mostly inertia K⁻¹[Λa] under the 20 Hz hold (6.66 → 2.74 mm after subtracting it, 9 demos).
- Closed loop: (c) and (c′) fail for *different* reasons than the offline gap suggests:
  - (c): free-space problem (P4) — first contact 24.9 s after the expert (median), runs out of time.
  - (c′): reaches the table on time, but the policy feeds its observed velocity into the target:
    G_n = 0.325 s vs D_n/K_n = 0.5 s ((b): 0.149 s), leaving 87 N·s/m of the controller's 250 N·s/m normal damping
    ((b): 175). Same at every K_n (G_n/(D_n/K_n) ≈ 0.65–0.75). Effects: force chatter on stiff tables
    (peaks 19–50 N; 351/850 episodes > 20 N for seed 0) and overshoot / lost lanes in the plane.
  - Velocity gains are reproducible across seeds (G_n (c′) = 0.325 / 0.326 / 0.327 s).

### P3 (`results/p3_steady_spread.py` main; `results/fig_p3_force_spread.py` supplementary)
- The whole-episode force std (≈ 2.7 N for every method, incl. the expert) is dominated by onset/lift ramps → not informative.
- Steady-wiping std of f_n grows with table stiffness within every K_n for (b), (c), (c′) and the expert
  (Spearman with K_eff within K_n: (b) +0.56/+0.64/+0.64, (c) +0.75/+0.84, (c′) +0.84/+0.80/+0.76);
  the hybrid controller (force feedback) is flat (−0.03).
  K_n = 500, (b): 0.22–0.32 N (k_e ≤ 2e3) → 0.39–0.54 N (k_e ≥ 2e4); (c′): 1.1–1.4 → 2.2–2.6 N.
- But the spread keeps rising from k_e = 2e4 to 3e5 where K_eff is already saturated at K → contact dynamics on stiff
  surfaces matter beyond the quasi-static K_eff·δ model. Offline one-step target errors (σ_δ ≈ 0.1–0.5 mm,
  `scripts/action_spread.py`) predict only ≈ 0.1–0.2 N. Verdict: direction supported, magnitude/mechanism not.

### P4 (`results/fig_p4_contact_ood.py`, `results/fig_p4_approach.py`, `results/fig_p4_tilt28_paths.py`)
- Does (c) reach the table? Yes in 99.7 % of ID episodes, but late: median 24.9 s after the expert; approach speed
  1.4 mm/s (30 → 5 mm above surface) vs 13.3 mm/s for the expert. (a): 1.9 mm/s, 94 % ever touch, 5 % ever press ≥ f_min.
  (b) and (c′) approach like the expert (13.4 and 15.8 mm/s). The (c) deficit shrinks with K_n (lead ∝ D/K):
  ID success 0.2 % / 6.5 % / 77 % at K_n = 250 / 500 / 1000.
- Steeper unseen tilt (28°): (b) 0/150 on both 28° tables (3 seeds); (c′) 24–31 %; (a), (c) 0 %; scripted and hybrid 100 %.
  (b) presses correctly along the true normal (offset −20.3 mm, same as on 20°) but its commanded lead up the slope
  drops from +1.3 mm (20°, = expert) to −0.4…+0.5 mm, so it never steps to the next lane and wipes 1–8 cm *below* the
  region (`results/tables/p4_tilt28_u2.csv`).
- Other OOD tables for (b): softer k_e = 250: 99 %; 15°/5e4: 86 %; harder 3e5: 75 % (scripted 100 %, hybrid 62 %).

## Figures (script → file)

| figure | script | file |
|---|---|---|
| Label replays, force vs time | `results/fig_phase1_replay_force.py` | `results/figures/phase1_replay_force.png` |
| P2 label gap by phase and speed | `results/p2_label_gap.py` | `results/figures/p2_label_gap.png` |
| P1 force ratio, success vs k_e | `results/fig_p1_force_ratio.py` | `results/figures/p1_force_ratio.png` |
| P2 closed loop, velocity gain, K sweep | `results/fig_p2_closed_loop.py` | `results/figures/p2_closed_loop.png` |
| P3 steady force spread vs k_e (main) | `results/p3_steady_spread.py` | `results/figures/p3_steady_spread.png` |
| P3 whole-episode spread (supplementary) | `results/fig_p3_force_spread.py` | `results/figures/p3_force_spread.png` |
| P4 approach trajectories | `results/fig_p4_approach.py` | `results/figures/p4_approach.png` |
| P4 paths on 28° tables | `results/fig_p4_tilt28_paths.py` | `results/figures/p4_tilt28_paths.png` |
| P4 contact and OOD | `results/fig_p4_contact_ood.py` | `results/figures/p4_contact_ood.png` |
| Main tables (K_n = 500, 250, 1000) | `results/table_main.py --K {500,250,1000}` | `results/tables/main_{K}.md` |
| Rollout videos | `results/make_videos.sh` | `results/videos/*.mp4` |

## Limitations to discuss (facts)

- Point contact: no contact patch; r is a virtual wiping radius. k_e is a pure spring + HC damping.
- Simulation only; force is measured exactly (no sensor noise) except in the stretch study.
- Single robot, single tool, one region shape; lane direction fixed in the table frame.
- Effective mass ≈ 18 kg (reflected rotor inertia) makes contact underdamped on hard tables;
  approach and corner speeds were reduced to keep impacts below f_max.
- Hybrid baseline does not know the table height, so its guarded approach must be slow; it is
  borderline on the OOD k_e = 3e5 table.
