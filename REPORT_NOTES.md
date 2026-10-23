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

## Figures (script → file)

| figure | script | file |
|---|---|---|
| Label replays, force vs time | `results/fig_phase1_replay_force.py` | `results/figures/phase1_replay_force.png` |
| P2 label gap by phase and speed | `results/p2_label_gap.py` | `results/figures/p2_label_gap.png` |
| P1 force ratio, success vs k_e | `results/fig_p1_force_ratio.py` | `results/figures/p1_force_ratio.png` |
| P3 force spread vs K_eff | `results/fig_p3_force_spread.py` | `results/figures/p3_force_spread.png` |
| P4 contact and OOD | `results/fig_p4_contact_ood.py` | `results/figures/p4_contact_ood.png` |
| Main table | `results/table_main.py` | `results/tables/main_500.md` |

## Limitations to discuss (facts)

- Point contact: no contact patch; r is a virtual wiping radius. k_e is a pure spring + HC damping.
- Simulation only; force is measured exactly (no sensor noise) except in the stretch study.
- Single robot, single tool, one region shape; lane direction fixed in the table frame.
- Effective mass ≈ 18 kg (reflected rotor inertia) makes contact underdamped on hard tables;
  approach and corner speeds were reduced to keep impacts below f_max.
- Hybrid baseline does not know the table height, so its guarded approach must be slow; it is
  borderline on the OOD k_e = 3e5 table.
