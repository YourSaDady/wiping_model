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

## 9. Likely TA questions (answers to be completed with the final numbers)

- *Why does (a) lose force?* Springs in series: target at depth d_demo, tool settles at
  d_a with K(d_demo − d_a) = k_e d_a ⇒ f_a/f_demo = K/(K+k_e). Verified in Drake to within
  1 % statically and in open-loop replay on soft tables (`results/phase1/replay_check.csv`).
- *Why isn't (c) equal to (b)?* (c) uses static balance; it misses D ẋ/K (≈ D_t v/K_t =
  0.11 s × v in the plane: 5–14 mm, measured in `results/p2_label_gap.py`) and is identical
  to (a) before contact, so it never encodes "press down" in free space.
- *Why table-frame gains?* See §3.
- *Why Fisher / Wilson?* See the BC notes §5.2; we also report McNemar's exact test because
  all methods run the same test scenes (paired outcomes).
