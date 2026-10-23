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
| 2 Data + NN smoke test | done 2026-10-23 | 3 × 432 demos, all expert episodes succeeded; NN smoke test (b) 115, (c′) 90, (c) 71, (a) 0 of 120 |
| 3 Diffusion Policy | pilot done, sweep running | pilot (b) 120/120 val; 20-run sweep in tmux `sweep` |
| 4 Baselines | working on val; final eval pending | val: scripted 170/170, hybrid 166/170 (4 fails on OOD k_e = 3e5, impact ≈ 21 N) |
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

## Phase 2 results (2026-10-23)

- Demos: `python scripts/gen_demos.py --K {500,250,1000}` → 432 per K (36 × 12 tables), every
  expert episode succeeded (manifests in `data/demos/K*/manifest.json`, no failures). 51 MB per K.
- Held-out scenes are defined by seed splits (`wipe/scene.py`): `val` (10 per table, checkpoint
  selection), `test` (50 per table, final evaluation only), `dev` (tuning).
- Loader: `wipe/dp/dataset.py` (DP windows T_o = 2, H = 16, padding as in DP).
- Nearest-neighbour smoke test through the closed-loop evaluator (K_n = 500, 120 val scenes):
  (a) 0/120 (mean f_n 0.66 N), (b) 115/120, (c) 71/120 (19 peaks > 20 N), (c′) 90/120 (15 peaks > 20 N).
  Same ordering as the prototype.
- Offline P2 (`python results/p2_label_gap.py`, K_n = 500 demos): in straight sweeps the (c)
  gap to (b) grows from 6.3 mm (4.6 cm/s) to 13.7 mm (12.5 cm/s), tracking D_t v/K_t; (c′)
  stays at 3.4–5.2 mm (remaining term: inertia under the 20 Hz zero-order hold — hypothesis,
  not yet verified). Along n: approach gap (c) 6.0 mm, (c′) 0.47 mm; contact onset (c) 2.2 mm,
  (c′) 1.5 mm. Tables in `results/tables/p2_gap_by_*.csv`, figure `results/figures/p2_label_gap.png`.

## Phase 3 pilot (2026-10-23)

- Label (b), K_n = 500, seed 0, 40k steps, relative vs absolute targets
  (`runs/pilot_K500_b_{rel,abs}`). Closed-loop val success at 10k/20k/30k/40k steps:
  relative 115/119/120/120, absolute 115/120/120/120 (of 120). Early failures were lane
  skipping (progress ambiguity), gone after 20k steps. Choices recorded in DECISIONS.md:
  relative targets, original 16-D observation, 40k steps, select among last 4 snapshots.
- Training speed: 38 ms/step alone (65.7M params, batch 256), ≈ 75–110 ms/step with 2–3 runs
  sharing the GPU (no throughput gain from sharing).

## Running jobs

- tmux `sweep`: `xargs -P 3 -L 1 scripts/run_one.sh < scripts/sweep_list.txt` (train 40k steps →
  select checkpoint on val), logs `logs/sweep.log`, `logs/<run>_{train,select}.log`.
  Started 2026-10-23 09:52 UTC; expected ≈ 9–10 h. Resumable: rerun the same command.

## Next

1. When the sweep finishes: check every run's `selected.json`; run `scripts/action_spread.py` per run.
2. Phase 5 final evaluation on test scenes (all methods, 50 scenes × 17 tables), then P1–P4.
