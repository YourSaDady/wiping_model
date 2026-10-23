# STATUS

Project: does the action label matter for a learned wiping policy under Cartesian impedance
control, and which label recovers the missing force? (MIT 6.4212, Fall 2026; team Yanchuan
Tang, Linda Yinghui He, Yichuan Song). Due Dec 2, 2026; all of Phase 5 by Nov 24.

**Resume from here.** `source env.sh` from the project root; design choices are in
`DECISIONS.md`; commands to rerun everything are in `README.md` (section "Pipeline").

## Phase overview

| Phase | State | Key numbers |
|---|---|---|
| 0 Setup | done 2026-10-23 | venv (Py 3.12, drake 1.58.0, torch 2.11 cu128), git repo, 32 CPUs, A6000 48 GB |
| 1 Environment + expert | done 2026-10-23 | all checks pass, see below |
| 2 Data + NN smoke test | in progress | |
| 3 Diffusion Policy | not started | |
| 4 Baselines | code written (scripted, hybrid); final eval pending | |
| 5 Evaluation + P1–P4 | not started | |
| 6 Stretch | not started | |
| 7 Write-up material | not started | |

## Phase 1 results (2026-10-23)

Command: `python scripts/phase1_checks.py --n 50` (≈ 12 min on 28 workers), then
`python results/fig_phase1_replay_force.py`. Outputs in `results/phase1/`.

- Unit tests `pytest tests/` (8 pass): label formulas and k+1 alignment, table-frame gains,
  lane path covers the region, coverage needs f ≥ f_min, force sign (+n when pressing) and
  springs in series (static f_n = K_eff·δ within 1 % for 4 (K_n, k_e) pairs).
- **Expert**: 50 dev scenes per table. Success 50/50 on all 12 training tables at K_n = 500,
  250 and 1000, and on all 5 OOD tables at K_n = 500. Peak 1 kHz normal force ≤ 15.6 N on
  training tables (≤ 12.6 N at K_n = 500), 16.8 N on the OOD k_e = 3e5 table → f_max = 20 N kept.
  Slowest episodes ≈ 40 s → T_MAX raised to 50 s.
- **Replay of (b)** reproduces the demo exactly (max |Δx| = 0.0 m in every replay).
- **Replay of (a)**, mean f_n during the sweep relative to the demo, vs K/(K + k_e):
  K_n = 500: 0.494 vs 0.500 (k_e = 500), 0.199 vs 0.200 (2e3), 0.042 vs 0.024 (2e4), 0.027 vs 0.005 (1e5);
  K_n = 1000: 0.662 vs 0.667 (500), 0.331 vs 0.333 (2e3); K_n = 250: 0.319 vs 0.333 (500), 0.108 vs 0.111 (2e3).
  Matches on soft surfaces. On hard surfaces (a) keeps ≈ 0.2–0.4 N instead of 0.05–0.24 N: the
  replayed tool touches only intermittently (force bumps on some lanes, see figure), and a
  unilateral contact cannot average to the small linear prediction. Both are far below f_min.
- **Replay of (c)** keeps 99–100 % of the force during the sweep but overshoots at contact
  onset: peak 27–30 N on k_e = 1e5 (K_n = 500), 48–53 N at K_n = 250 → fails f_max there.
  **(c′)** succeeds in every replay. (a) succeeds only where K/(K+k_e)·f_des ≳ f_min
  (K_n = 1000 on k_e = 500: 2/2; K_n = 500, 20° soft: 1/3).
- Figure: `results/figures/phase1_replay_force.png`.
- Expert/controller changes made in Phase 1 (details in DECISIONS.md): slow zone 2 cm,
  lane-corner slowdown to 0.35 v, hybrid approach 8 mm/s.

## Running jobs

(none)

## Next

1. Generate demos K_n = 500 (12 tables × 36), then K_n = 250 and 1000.
2. Nearest-neighbour smoke test through the closed-loop evaluator on validation scenes.
3. Pilot DP on label (b): relative vs absolute targets, training time, closed-loop check.
