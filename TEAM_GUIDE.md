# Team guide: how every part works and how to defend it

Study notes so each of us can answer questions about any part of the project in the final
in-class session. This is a learning aid, not report text: write the report in your own
words. Numbers quoted here come from scripts in the repo (file named next to each number).

---------------------------------------------------------------------------------------------

## 1. The question in one paragraph (for yourself)

Under Cartesian impedance control the hand behaves like a spring K pulled toward a commanded
target x_d. To press with force f the target must sit *below* the surface: K(x_d − x) + f = 0.
A behaviour-cloning dataset that only records where the hand went (label (a), x) loses that
offset, so a policy trained on it barely presses. We compare four labels built from the *same*
demonstrations — (a) x, (b) the commanded x_d, (c) x − K⁻¹f, (c′) x + K⁻¹(Dẋ − f) — with an
identical Diffusion Policy, and two baselines that use no demonstrations.

## 2. Simulation (`wipe/sim.py`, `wipe/config.py`)

- **Robot**: Drake's iiwa14 (no collision geometry), base at the origin, plus a welded tool:
  rod + ball of radius R = 2 cm whose centre is 0.17 m along link 7's z axis. The ball centre
  is the controlled point ("TCP"); all positions x are the ball centre.
- **Time stepping**: discrete MultibodyPlant, 1 ms step. Controller is a Python LeafSystem
  evaluated every step (≈ 6× faster than real time on one core).
- **Contact**: point contact, `kLagged` discrete approximation, Hunt–Crossley dissipation
  20 s/m, Coulomb friction μ = 0.3.
  - *Why point contact?* Its normal force is k·(penetration)·(1 + d·penetration rate), so the
    table stiffness k_e is one number we control exactly. Hydroelastic would make the
    stiffness depend on the geometry (no single k_e).
  - *Why 2k_e on both bodies?* Drake combines the two point stiffnesses in series:
    k = k₁k₂/(k₁+k₂). With k₁ = k₂ = 2k_e the pair has k_e.
    Checked by `tests/test_core.py::test_force_sign_and_springs_in_series`.
- **Contact force**: `ContactResults.point_pair_contact_info(i).contact_force()` is the force
  on body B in world; we flip the sign if the tool is body A. f is the force the table
  applies to the tool; it points along +n when pressing.
- **Tables**: a 0.5 × 0.5 × 0.1 m box tilted about world x by θ, normal
  n = (0, −sin θ, cos θ). Training: θ ∈ {0, 10, 20}° × k_e ∈ {500, 2e3, 2e4, 1e5} N/m.
  OOD: (15°, 5e4), (28°, 2e4), (28°, 500), (0°, 250), (10°, 3e5).
- **Effective mass** along n at the start poses is ≈ 16–19 kg, mostly the motors' reflected
  rotor inertia (rotor inertia × gear ratio², ≈ 3.4 kg·m² on joints 1–4). This matters for
  impacts: the peak impact force is roughly v·sqrt(k_e·m_eff).

## 3. Cartesian impedance controller (`wipe/controllers.py::ImpedanceController`)

`tau = Jᵀ [K_rot e_R − D_rot ω ; K(x_d − x) − D ẋ] − tau_g + N (K_null(q0 − q) − D_null q̇)`

- J: 6×7 spatial-velocity Jacobian of the ball centre (`CalcJacobianSpatialVelocity`), rows
  [ω; v]. Jᵀ maps a desired wrench at the TCP to joint torques (virtual work).
- −tau_g: gravity compensation (`CalcGravityGeneralizedForces` returns the generalized
  gravity force, so we subtract it). With perfect compensation the closed loop near rest is
  Λẍ + Dẋ + K(x − x_d) = f_ext: a spring–damper at the TCP (textbook Ch. 8, "Cartesian
  stiffness / operational-space control" without inertia shaping).
- Orientation: held at the IK start orientation with a rotational spring, e_R from the skew
  part of R_d Rᵀ.
- Null space: the 7-DoF arm has one redundant DoF; N = I − Jᵀ(JJᵀ)⁻¹J projects a weak joint
  spring toward the start posture so the elbow does not drift, without affecting the TCP.
- **Gains in the table frame**: K = K_t I + (K_n − K_t) n nᵀ (K_t = 1000, K_n = 500 N/m),
  D likewise (D_t = 110, D_n = 250 N·s/m). The stiffness along n is exactly K_n on every tilt.
  World-frame diagonal gains on a tilted table would couple in-plane motion into the normal
  force (because K_z ≠ K_x), which would confound the P1/P3 tests.
- **Why D_n = 250, not 2·sqrt(K_n m)?** Damping sized for free space (≈ 2·sqrt(500·18) ≈ 190)
  is far too little for the contact mode, whose stiffness is K_n + k_e: on k_e = 1e5 the
  contact damping ratio is D/(2·sqrt((K+k_e)m)) ≈ 0.09 even with D = 250. The tool bounces
  if it hits fast. We therefore slow the approach (below) instead of raising D further,
  which would make free-space motion sluggish (lag D/K·v).
- **K sweep**: K_n ∈ {250, 500, 1000} (D fixed) to change K/(K + k_e) and K_eff = K k_e/(K + k_e).

## 4. Scenes, region and lane path (`wipe/scene.py`, `wipe/paths.py`)

- Region: L×L square on the table, corner p0, axes t1 (lanes, world x) and t2 = n × t1.
  L ∈ [0.13, 0.17] m, 6 lanes, spacing s = L/5 < 2r.
- Lane path: start → p0 → lane 0 along +t1 → step s along t2 → lane 1 along −t1 → … The
  path speed is v ∈ [5, 12] cm/s, dipping to 0.35 v within 2 cm of each corner (without the
  dip, fast reversals made the tool overshoot, leave the surface and land again at up to 83 N
  on the hardest table).
- Seeds: scene = f(table, split, index) via `SeedSequence([crc32(table), split, index])`;
  demo, validation, test and dev splits never share a seed.

## 5. Scripted expert (`wipe/expert.py`)

Privileged (knows the plane and region), 20 Hz, zero-order hold (the target is constant
between policy steps, exactly like a learned policy):

1. approach along −n at 2 cm/s, 6 mm/s once the ball is within 2 cm of the surface, until
   f_n ≥ 1 N;
2. press: h_d ← h_d − (f_des − f_n)/(K_n·0.3 s)·Δt (integral force control through the
   target; time constant ≈ 0.3 s·K_n/K_eff);
3. sweep the lanes with the force integrator still on;
4. lift at 3 cm/s until 2 cm above the surface.

*Why must the expert run at exactly 20 Hz?* Then the logged x_d is exactly what the
controller saw, and replaying label (b) open loop reproduces the demonstration bit for bit
(checked: max |Δx| = 0). If the expert ran faster and we downsampled, (b) would not be exact.

## 6. Labels and time alignment (`wipe/labels.py`)

Label k is the target held over [t_k, t_{k+1}), so the label must use the state that
command produced, at k+1:

| label | formula | what it gets right / wrong |
|---|---|---|
| (a) | x_{k+1} | where the hand went; loses the spring offset → keeps only K/(K+k_e) of the force (two springs in series) |
| (b) | x_{d,k} | exactly the command; needs the robot to log x_d |
| (c) | x_{k+1} − K⁻¹f_{k+1} | static force balance; drops the damping term D ẋ/K and is equal to (a) before contact (f = 0) |
| (c′) | x_{k+1} + K⁻¹(Dẋ_{k+1} − f_{k+1}) | adds damping; still drops inertia Λẍ |

Derivation: near-static balance at the TCP, K(x_d − x) − Dẋ + f = Λẍ ≈ 0 ⇒
x_d = x + K⁻¹(Dẋ − f) (= (c′)); dropping Dẋ gives (c).

The observation is identical for every label (no x_d, no previous action, no expert phase):
o_t = (x, ẋ, f, p0, L, n), 2 frames.

## 7. Success metric (`wipe/coverage.py`, `WipeEnv.metrics`)

- Coverage on a 2 mm grid; a cell is wiped when it is within r = 2 cm of the contact point at
  a 100 Hz sample with f_n ≥ f_min = 5 N. r is a *task parameter* (a sponge pad radius): a
  rigid ball in point contact has essentially no contact patch.
- Success: uncovered fraction ≤ 0.10 and the 1 kHz normal force never exceeds f_max = 20 N
  (impacts included). f_des = 10 N.
- Mean / spread of f_n are computed over "wiping samples": contact point inside the region
  (grown by r) and the ball within 5 mm of the surface — so a policy that hovers just above
  the table shows up with f ≈ 0 rather than being excluded.

## 8. Hybrid force/position baseline (`HybridForcePositionController`)

Classical (Ch. 8, "hybrid position/force control"), at 1 kHz, torque level:
- in the plane: the same spring–damper as the impedance controller, tracking the lane path;
- along n: approach with a velocity servo at 8 mm/s (it does **not** know the table height),
  then force control F_n = −(f_ref + k_p e + k_i∫e) − D_n ẋ_n with the measured force,
  f_ref ramping to f_des over 0.5 s.
- 8 mm/s was chosen on the training tables: impact force ≈ v·sqrt(k_e m_eff) ≈ 11 N on
  k_e = 1e5. It is borderline on the OOD k_e = 3e5 table (≈ 20–22 N): the price of not
  knowing where the table is.

## 9. Diffusion Policy (`wipe/dp/`, `scripts/train_dp.py`)

- **What it learns**: the distribution of an action chunk A_t = (a_t, …, a_{t+15}) given the
  last 2 observations. Training: take a demo chunk A⁰, pick a diffusion step k ∈ {0..99}, add
  noise A^k = sqrt(ᾱ_k) A⁰ + sqrt(1 − ᾱ_k) ε, and train the network ε_θ(A^k, k, O_t) to predict ε
  (mean-squared error). Sampling: start from Gaussian noise and denoise with DDIM (16
  deterministic steps), each step estimating A⁰ = (A^k − sqrt(1−ᾱ_k) ε_θ)/sqrt(ᾱ_k), clipping to
  [−1, 1], and moving to the previous noise level.
- **Why diffusion and not plain regression?** Demonstrations are multimodal (e.g. the speed v
  varies between demos and is not observed); a regression head would average modes. BC notes §2.3.
- **Network**: 1D temporal U-Net over the 16-step chunk (channels 256 → 512 → 1024, kernel 5,
  GroupNorm + Mish), conditioned by FiLM: every residual block gets a per-channel scale and
  bias computed from [diffusion-step embedding, flattened 2-frame observation].
- **Noise schedule**: squared cosine (Nichol & Dhariwal), 100 steps.
- **Parameter count** (all 20 main policies are the same network, 65,670,403 parameters;
  the 9-D `pf` stretch policy has 65,681,161): step-embedding MLP 0.53M; encoder level 0
  (3→256) 1.48M, level 1 (256→512) 6.10M, level 2 (512→1024) 20.07M; bottleneck 22.17M;
  decoder (2048→512) 11.87M, (1024→256) 3.12M; output head 0.33M. FiLM layers are 4.14M (6.3 %).
  Most weights sit at the 1024-channel bottleneck, where the chunk has only 4 time steps:
  the paper's low-dim default is far larger than a 16×3 action chunk needs, but we kept it.
- **Shapes**: noisy chunk (B, 16, 3) → channels-first (B, 3, 16) → 256@16 → 512@8 → 1024@4 →
  bottleneck 1024@4 → 512@8 → 256@16 → 3@16. Conditioning vector per block: 256 (step) + 32
  (2 frames × 16 observation numbers) = 288.
- **Action representation**: targets relative to the current ball position, a_j = target_{t+j} − x_t
  (all labels identically). Execution: add x_t back, execute the first 8 targets (0.4 s), observe,
  re-plan (receding horizon, BC notes §4.2–4.3).
- **Normalization**: min-max to [−1, 1] using training windows only.
- **EMA**: evaluation uses an exponential moving average of the weights (decay → 0.9999).
- **Training**: AdamW 1e-4, batch 256, 40k steps, cosine LR with 500 warm-up; checkpoint chosen
  by closed-loop success on 120 validation scenes (never test scenes).
- **Pilot**: label (b) reached 120/120 validation scenes from 20k steps; relative vs absolute
  targets tied in closed loop; at 10k steps a few rollouts skipped lanes (progress ambiguity:
  with 2 frames the lane index must be inferred from position).

## 10. Closed-loop evaluation (`wipe/evaluate.py`)

- Lockstep: the policy runs on the GPU in the main process, batched over ~24 simulator worker
  processes; each worker executes 8 targets (0.4 s of simulated time) per round.
- Every episode runs the full T_MAX = 50 s (no early termination), so a policy that lifts early
  or comes back down and slams the table is caught.
- DDIM noise is seeded by (scene, inference call): every method sees the same noise on the same
  scene → paired trials, reproducible regardless of batching.
- Splits: `val` scenes (10/table) for checkpoint selection, `test` scenes (50/table) only for the
  final numbers.

## 11. Statistics (`wipe/stats.py`, BC notes §5.2)

- Wilson 95 % interval for each success rate (better than the normal interval near 0 or 1).
- Fisher's exact test for two success counts (conditional on the total successes, every
  assignment of successes to trials is equally likely). Two-sided unless the direction was
  fixed in advance. Checked against the notes' example: 7/10 vs 9/10 → p = 0.58.
- McNemar's exact test when the same scenes are run by both methods (paired): only discordant
  scenes (one succeeds, the other fails) carry information.
- Holm–Bonferroni correction across the 6 label pairs.
- Report effect sizes (success-rate differences with intervals), not only p-values.

## 12. Likely TA questions

- *Why does (a) lose force?* Springs in series: target at depth d_demo, tool settles at
  d_a with K(d_demo − d_a) = k_e d_a ⇒ f_a/f_demo = K/(K+k_e). Verified in Drake to within
  1 % statically and in open-loop replay on soft tables (`results/phase1/replay_check.csv`).
- *Why isn't (c) equal to (b)?* (c) uses static balance; it misses D ẋ/K (≈ D_t v/K_t =
  0.11 s × v in the plane: 5–14 mm, measured in `results/p2_label_gap.py`) and is identical
  to (a) before contact, so it never encodes "press down" in free space.
- *Why table-frame gains?* See §3.
- *Why Fisher / Wilson?* See the BC notes §5.2; we also report McNemar's exact test because
  all methods run the same test scenes (paired outcomes).
- *Why does (c′) still differ from (b)?* It still drops the inertial term [Λẍ]: under the 20 Hz
  zero-order hold the target jumps every 50 ms and the ≈ 9–18 kg effective mass accelerates
  within each interval. On 9 demos, subtracting K⁻¹[Λa] measured at 1 kHz reduces the (c′)
  sweep residual from 6.7 to 2.7 mm (`results/p2_inertia_check.py`).
- *Why 2 observation frames?* DP default; velocity is observed directly, so 2 frames mostly
  add the force trend.
- *Why relative actions?* Pilot tie in success, lower action error, and invariance to region
  position.
- *How do you know it is not overfit to the test scenes?* Test scenes have their own seeds and
  were run once, after all choices were frozen; checkpoints were chosen on validation scenes.

## 13. Results you must be able to explain (final test numbers; details in REPORT_NOTES.md)

Headline (K_n = 500, 50 test scenes × 12 training tables × 3 seeds = 1800 ID episodes per label;
OOD: 5 tables × 50 × 3 = 750): scripted 100 / 100 %, hybrid 100 / 92 %, (a) 0 / 0 %,
(b) 99.6 / 52 %, (c) 7 / 4 %, (c′) 24 / 28 % (ID / OOD success).

- *Why does (a) fail even on the soft table where K/(K+k_e) = 0.5 predicts 5 N?* Because in closed
  loop it never gets there: before contact (a)'s targets are just "where the hand will be", with no
  lead, so with D_n/K_n = 0.5 s of lag the arm creeps down at ≈ 1.9 mm/s and touches the table ≈ 23 s
  after the expert; once touching it presses ≈ 0.3–0.7 N. The two-spring formula holds open loop
  (replay: 0.494 vs 0.5) but is not the binding limit in closed loop.
- *Why does (c) fail if it reproduces the force in steady wiping?* Same free-space problem: (c) = (a)
  until f > 0. Median first contact 24.9 s late, approach 1.4 mm/s → runs out of the 50 s. Once in
  contact its force is right (101 % of the expert's). With a stiffer controller the missing lead D v/K
  shrinks and (c) recovers: 0.2 % → 6.5 % → 77 % ID success at K_n = 250 / 500 / 1000.
- *(c′) adds D ẋ: why is it still bad?* It does fix the approach (contact on time). But the policy
  learns that the target lead scales with its *observed* velocity, G_n = 0.33 s along n (D_n/K_n =
  0.5 s). Plugged into F = K(x_d − x) − D ẋ this cancels ≈ 2/3 of the controller's damping (87 vs
  250 N·s/m left; (b) leaves 175). On stiff tables the lightly damped contact chatters (peaks up to
  50 N), and in the plane it overshoots lane ends. Offline label accuracy ≠ closed-loop quality.
- *What explains the remaining (c′)–(b) gap offline?* Inertia: under the 20 Hz zero-order hold the
  target jumps every 50 ms and the ≈ 9–18 kg effective mass accelerates within each step; subtracting
  K⁻¹[Λa] cuts the gap from 6.7 to 2.7 mm.
- *P3?* With steady-wiping force std (onset and lift removed), spread grows with table stiffness for
  every impedance-based method (rank correlation +0.56 to +0.84 within each K_n) and is flat for the
  hybrid controller, which closes the loop on force. But it keeps growing where K_eff ≈ K has
  saturated (k_e 2e4 → 3e5), so contact vibration matters beyond the quasi-static K_eff·δ; and the
  policies' one-step target errors (0.1–0.5 mm) predict only 0.1–0.2 N. Direction supported,
  mechanism only partly.
- *P4, steeper tilt?* All learned policies degrade on 28° (unseen; training ≤ 20°): (b) 0 %, (c′)
  24–31 %, baselines 100 %. (b) presses correctly but its commanded lead up the slope collapses, so
  it never steps to the next lane and wipes below the region. Only 3 tilt values in training.
- *Why are the baselines so good?* The scripted program is privileged (knows the plane) and the
  hybrid controller regulates the measured force directly; neither depends on a label. They show
  the task is solvable; the comparison is about what a learned policy can recover from each label.
- *How many trials, how sure?* 600 ID scenes per policy per seed, Wilson 95 % CIs (e.g. (b)
  [99, 100] %), every label pair different with Holm-corrected Fisher p ≤ 3e-40 and paired McNemar
  p ≤ 1e-11 on ID tables.
- *Isn't (c)'s failure just your 50 s time limit?* Partly — we checked (pre-registered): with 100 s, (c) goes
  from 6.5 % to 82 % (seed 0, 600 ID scenes), the rest failing on onset force peaks > 20 N. So (c) has the
  right contact force but approaches ≈ 10× too slowly. (a) stays at 0 % (never presses), (c′) stays at 20 %
  (its failures are force chatter and lost lanes), (b) stays at 99.5 %.
