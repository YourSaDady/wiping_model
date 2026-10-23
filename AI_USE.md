# AI use log

Running log of what the AI coding agent did, for the report's required "how we used AI"
section and the Acknowledgments disclosure. The course policy (GAI-Tech.pdf) allows any GenAI
use on non-communication project work; the communication policy (GAI-Comm.pdf) does **not**
allow GAI-generated report text or oral presentations and requires disclosure of GAI
assistance with text, figures and code in the Acknowledgments (AI system, sections, level of
use, sources). Consequently the agent produces code, experiments, figures, tables and
bullet-point notes only; it writes no report prose and no video narration.

**System:** Claude (Anthropic), model Opus 5.5, running as a coding agent (Claude Code) on the
lab server, started by Yanchuan Tang on 2026-10-23 with a written task brief and the team's
final proposal.

## What the team provided

- The research question, the proposal (labels (a)–(c), predictions P1–P4, baselines,
  milestones, division of work) and a task brief with lessons from the team's earlier laptop
  prototype (controller tuning, damping sizing, label time alignment, the K/(K+k_e) finding,
  the (b) ≠ (c) nearest-neighbour result, default parameter values).
- Decisions reserved for the team: scope changes and anything that changes the story.

## Log

| Date | What the agent did | Human decisions / inputs it relied on |
|---|---|---|
| 2026-10-23 | Read the course project page, AI policy PDFs, schedule, BC notes §4/§5.2, textbook Ch. 8 summary. Set up the venv, git repo. | Brief, proposal. |
| 2026-10-23 | Wrote the Drake environment (`wipe/sim.py`), Cartesian impedance controller and hybrid force/position controller (`wipe/controllers.py`), scripted expert, lane path, coverage metric, labels, rollouts, unit tests. | Controller form, gains, contact model and expert design from the brief; the agent chose table-frame gains (DECISIONS.md) and tuned the approach (2 cm slow zone), lane-corner slowdown and hybrid approach speed after measuring impact peaks. |
| 2026-10-23 | Ran Phase 1 checks (expert on 50 scenes × 17 tables × 3 K; open-loop label replays); made the replay figure. | — |
| 2026-10-23 | Generated 3 × 432 demonstrations; wrote the Diffusion Policy re-implementation (U-Net, DDPM/DDIM, EMA), dataset windows, training, checkpoint selection, lockstep closed-loop evaluator, nearest-neighbour smoke test, baselines evaluation, statistics helpers, P2 offline analysis, video renderer. | DP hyperparameters from the brief (paper defaults). |
| 2026-10-23 | Ran the label-(b) pilot (relative vs absolute targets, obs with/without progress feature), chose the representation and training budget from validation scenes, launched the 20-run sweep, the final test-evaluation queue and the stretch queue (pose+force label at a new K′; force noise/bias in label (c)). | Choice rules from the brief (pilot decides rel/abs; progress feature only if ambiguity hurts; select on validation only). |
| 2026-10-23 | Verified the P2 hypothesis that (c′)'s remaining gap is the inertial term (1 kHz diagnostic); wrote result-table/figure scripts for P1–P4, the team guide, report notes (bullets only), README pipeline. | — |
