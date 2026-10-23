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
| 3 Diffusion Policy | done 2026-10-23 21:36 | 20 policies trained + selected on val (`runs/*/selected.json`) |
| 4 Baselines | done 2026-10-23 | test: scripted 850/850, hybrid ID 600/600, OOD 231/250 |
| 5 Evaluation + P1–P4 | done 2026-10-23 (time-limit sensitivity running) | see "Phase 5 results" and REPORT_NOTES.md |
| 6 Stretch | training (tmux `stretch`) | |
| 7 Write-up material | in progress | videos rendered; notes, guide, AI log drafted |

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
- tmux `evalq`: `scripts/eval_queue.sh` — final test evaluation (50 scenes × 17 tables): scripted
  at K_n = 500/250/1000 and hybrid first, then each sweep run once its `selected.json` exists.
  Results `results/raw/<method>_test.jsonl` (+ `_traces/` for the first 3 scenes per table, q
  recorded for scene 0). Started 2026-10-23 09:59 UTC; restarted 11:29 with the pipelined evaluator
  (`EVAL_WORKERS=48`, ≈ 20 min per policy while training shares the GPU). Resumable.
- tmux `extra`: `scripts/extra_queue.sh` — time-limit sensitivity (T_MAX = 100 s, seed-0 K_n = 500
  policies, training-table test scenes); starts after `evalq`.
- tmux `stretch`: `scripts/stretch_queue.sh` — starts after the main sweep finishes training.
- Dev worktree `../6.4212-wiping-dev` (branch `dev`): stretch-goal code (pose+force label `pf`,
  `--deploy-K`, label force noise/bias), P2 inertia diagnostic, P1/P3/P4 figure scripts. Merge
  into master after the sweep (kept separate so running jobs never load half-edited code).

## First results (validation, 2026-10-23 11:25)

- K500 seed 0 selection (120 val scenes): (b) 120/120 (30k), (a) 0/120 (mean f_n ≈ 0.5 N),
  (c) 6/120: it reaches the table only after median 29.5 s (expert ≈ 4 s), then wipes at
  ≈ 9.1 N but runs out of time (38 % uncovered); 19/120 also exceed 20 N at onset on hard tables.
- Baselines on the 850 test scenes (final): scripted K500 850/850; hybrid ID 600/600, OOD 231/250
  (19 fails on k_e = 3e5, impact peaks ≤ 22.1 N); scripted K250 fails 46/50 on k_e = 3e5 (only used
  as the P1 reference on training tables), scripted K1000 850/850.

## Findings so far (seed 0, validation unless noted; 2026-10-23 13:30)

- Val selection: K500 (a) 0, (b) 120, (c) 6, (c′) 24 of 120; K250 (a) 0 (never touches the table
  in 50 s), (b) 120.
- Test (final, seed 0, K500): (a) 0/850; (b) ID 598/600, OOD 137/250 (0/50 on both 28° tables,
  40/50 on k_e = 3e5, 47/50 on 15°/5e4); (c) 48/850 (142 peaks > 20 N).
- **(c′) mechanism** (`results/p2_velocity_gain.py`, `results/tables/p2_velocity_gain.csv`): the
  policy's commanded lead depends on the *observed* velocity with gain G_n = 0.325 s along n
  ((b): 0.149 s; D_n/K_n = 0.5 s), so it cancels ≈ 2/3 of the controller's normal damping (net
  88 vs 175 N·s/m for (b)) → force chatter on stiff tables (≈ 28–35 N peaks on k_e = 1e5) and
  lane-end overshoot in the plane (G_t = 0.164 vs 0.145 s, D_t/K_t = 0.11 s). The (c′) label
  contains D ẋ/K explicitly; the policy reproduces that correlation with its own velocity.
  (c′) does fix the free-space problem: first contact ≈ 4.0 s like the expert.

- Val selection (all seed 0): K250 (c) 0, (c′) 63; K1000 (a) 0, (b) 120, (c) 91, (c′) 36 of 120.
  (c) improves sharply with controller stiffness (0 / 6 / 91 at K_n = 250 / 500 / 1000), consistent
  with its free-space lead deficit scaling with D/K (1 / 0.5 / 0.25 s).
- P3 action spread (`scripts/action_spread.py`, val demos, sweep, std of n·(â₀ − a₀)): (b) 0.48 /
  0.24 / 0.12 mm at K_n = 250 / 500 / 1000 (≈ constant 0.12 N in force units); (c), (c′) 0.10–0.37 mm;
  (a) 0.01 mm. K_eff·σ_δ ≈ 0.1–0.2 N, far below the measured closed-loop spread (≈ 2.7 N, same as
  the expert): one-step action error is not what sets the force spread.

## Phase 5 results (final test split, 2026-10-23 22:45)

All numbers and the analysis are in `REPORT_NOTES.md` ("Final results"); tables in
`results/tables/`, figures in `results/figures/`. Headline (K_n = 500, 3 seeds pooled, ID / OOD):
scripted 100 / 100 %, hybrid 100 / 92 %, DP (a) 0 / 0 %, (b) 99.6 / 52 %, (c) 7 / 4 %, (c′) 24 / 28 %.
- P1: open-loop replay matches K/(K+k_e); closed-loop (a) presses 3–8 % of the expert force on every table
  and never succeeds — it barely reaches the table (P4), so the springs-in-series bound is not the limit.
- P2: (c) and (c′) both lose to (b), for different reasons: (c) free-space lead missing (contact 25 s late);
  (c′) feeds observed velocity into the target (G_n 0.33 s of D_n/K_n = 0.5 s) → loses 2/3 of the normal
  damping → force chatter on stiff tables. (c) improves to 77 % at K_n = 1000 (lead ∝ D/K).
- P3: steady-wiping force spread grows with table stiffness within every K_n (Spearman +0.56…+0.84) but keeps
  growing where K_eff has saturated → contact dynamics matter beyond K_eff·δ; hybrid flat.
- P4: (c) reaches the table in 99.7 % of episodes but at 1.4 mm/s; on 28° tables (b) fails 0/150 (drifts below
  the region; in-plane lead up the slope collapses), (c′) 24–31 %, baselines 100 %.
- Commands: `python results/table_main.py --K {500,250,1000}`, `results/fig_p1_force_ratio.py`,
  `results/fig_p2_closed_loop.py`, `results/p3_steady_spread.py`, `results/fig_p3_force_spread.py`,
  `results/fig_p4_contact_ood.py`, `results/fig_p4_approach.py`, `results/fig_p4_tilt28_paths.py`,
  `python results/p2_velocity_gain.py --runs <all 20 runs>`, `scripts/action_spread.py --run <run>`,
  `bash results/make_videos.sh`.

## Next

1. Time-limit sensitivity (tmux `extra`) → add to REPORT_NOTES.
2. Stretch results (tmux `stretch`): pf label at K′, force noise/bias → analysis script + notes.
3. Phase 7: storyboard, TEAM_GUIDE Q&A with final numbers, AI_USE, TA check-in notes.
4. Pending team decision: optional inertia-corrected (c″) diagnostic.
