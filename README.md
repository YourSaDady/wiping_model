# Action labels for force-aware wiping (MIT 6.4212 final project, Fall 2026)

Team: Yanchuan Tang, Linda Yinghui He, Yichuan Song.

A KUKA iiwa under Cartesian impedance control wipes an L×L region on tables of different
tilt and contact stiffness k_e. From the same scripted demonstrations we build four action
labels and train identical state-based Diffusion Policies on each:

| label | target the policy predicts |
|---|---|
| (a) | measured ball position x_{k+1} |
| (b) | commanded target x_{d,k} |
| (c) | force-compensated x_{k+1} − K⁻¹ f_{k+1} |
| (c′) | force + damping compensated x_{k+1} + K⁻¹(D ẋ_{k+1} − f_{k+1}) |

Baselines without demonstrations: the privileged scripted program and a classical hybrid
force/position controller. See `STATUS.md` (progress and numbers), `DECISIONS.md`
(every design choice), `REPORT_NOTES.md`, `TEAM_GUIDE.md`, `AI_USE.md`.

## Layout

- `wipe/` — library: `config.py` (all parameters, tables), `scene.py` (seeded scenes),
  `sim.py` (Drake plant, `WipeEnv`), `controllers.py` (impedance + hybrid controllers,
  force sensor, peak monitor), `expert.py`, `paths.py` (lane path), `coverage.py`,
  `labels.py`, `rollout.py`, `data.py`, `dp/` (U-Net, DDPM/DDIM, dataset), `policy.py`,
  `evaluate.py` (lockstep closed-loop evaluator), `stats.py`.
- `scripts/` — entry points (below).
- `results/` — one script per figure/table (`fig_*.py`, `p*_*.py`, `table_*.py`), outputs in
  `results/figures/`, `results/tables/`; raw per-episode results in `results/raw/` (not committed).
- `tests/` — unit tests (`pytest tests/`).

## Pipeline (run from the project root after `source env.sh`)

```bash
# 0. environment (once)
uv venv --python /usr/bin/python3.12 .venv && source env.sh
uv pip install drake==1.58.0 numpy scipy matplotlib pandas tqdm pytest
uv pip install torch --index-url https://download.pytorch.org/whl/cu128
pytest tests/

# 1. environment + expert checks (Phase 1)
python scripts/phase1_checks.py --n 50
python results/fig_phase1_replay_force.py

# 2. demonstrations (Phase 2)
python scripts/gen_demos.py --K 500
python scripts/gen_demos.py --K 250
python scripts/gen_demos.py --K 1000
python results/p2_label_gap.py                      # offline P2 check
for l in a b c cp; do python scripts/eval_policy.py --nn $l --K 500 --split val --tables train --n 10; done

# 3. Diffusion Policy (Phase 3): train, then select a checkpoint on validation scenes
python scripts/train_dp.py --label b --K 500 --seed 0 --out runs/K500_b_s0
python scripts/select_ckpt.py --run runs/K500_b_s0

# 4-5. final evaluation on the 50 test scenes per table
python scripts/eval_policy.py --run runs/K500_b_s0 --ckpt selected --split test --tables all --n 50
python scripts/eval_baselines.py --method scripted --split test --tables all --n 50
python scripts/eval_baselines.py --method hybrid --split test --tables all --n 50
```

The full sweep and the analysis scripts are listed in `STATUS.md` as they are added.

## Attribution

- Drake (Tedrake et al.) for simulation, IK and rendering; the iiwa14 model from
  `drake_models`.
- The 1D temporal U-Net with FiLM conditioning and the training recipe follow Chi et al.,
  "Diffusion Policy", RSS 2023, and their public state-based Push-T notebook
  (https://github.com/real-stanford/diffusion_policy). Our `wipe/dp/` is a re-implementation;
  no code was copied. DDPM: Ho et al. 2020; DDIM: Song et al. 2021; cosine schedule: Nichol &
  Dhariwal 2021.
- Controllers follow the textbook (Tedrake, *Robotic Manipulation*, Ch. 8).
- Statistics follow the course behavior-cloning notes, Sec. 5.2.
- The code was written with an AI coding agent (Claude); see `AI_USE.md`.
