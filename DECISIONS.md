# Design decisions

Every choice the proposal or the task brief left open, with the reason. Numbers that
come from experiments point to the script that produced them. Newest decisions are
appended at the end of each section with a date.

## Software and simulation

| Item | Choice | Reason |
|---|---|---|
| Python / Drake | Python 3.12 venv (`uv`), `drake==1.58.0` (pip latest on 2026-10-23) | Brief suggested 1.57; 1.58 was current. All APIs used exist in both. |
| Torch | 2.11.0+cu128 on the A6000 | Matches the driver; own DDPM/DDIM code, no `diffusers` dependency. |
| Robot | `iiwa14_no_collision.sdf`, base welded at the origin | Brief. Only the ball has collision geometry. |
| Tool | Ball, radius R = 2 cm, centre 0.17 m along `iiwa_link_7` +z; 0.3 kg rod | Ball centre is the controlled point x (TCP). |
| Plant | Discrete, 1 ms step, point contact, `kLagged` approximation | Brief: point contact makes k_e one explicit number. |
| Contact stiffness | Both geometries get `point_stiffness = 2 k_e` | Drake combines point stiffnesses in series; pair stiffness = k_e. Verified by `tests/test_core.py::test_force_sign_and_springs_in_series` (f_n = K_eff·δ within 1 %). |
| Dissipation | Hunt–Crossley 20 s/m on both | Brief; equal values combine to 20 s/m. |
| Friction | μ = 0.3 (static = dynamic) | Wiping has friction; it adds a tangential force that label (c) compensates through K⁻¹f. |
| Contact force | Sum of `point_pair_contact_info(i).contact_force()` (force on body B, world), sign flipped when the tool is body A | Brief; sign checked in the unit test (table pushes along +n). |
| Effective mass | m_eff ≈ 16–19 kg along n at the start poses (reflected rotor inertia dominates) | Measured from Λ = (J M⁻¹ Jᵀ)⁻¹. Contact mode on hard tables is therefore underdamped (ζ ≈ 0.09 at k_e = 1e5 with D_n = 250), so impacts must be slow. |

## Controller

Cartesian stiffness control in operational-space form (Ch. 8), our own `LeafSystem`
(`wipe/controllers.py`):
`tau = Jᵀ[K_rot e_R − D_rot ω ; K(x_d − x) − D ẋ] − tau_g + N (K_null(q0 − q) − D_null q̇)`.

| Item | Choice | Reason |
|---|---|---|
| Gain frame (open decision 5) | **Table frame**: K = K_t I + (K_n − K_t) n nᵀ, same for D | P1 and P3 are 1-D predictions along n. In the table frame the stiffness along n is exactly K_n on every tilt, so K/(K+k_e) and K_eff are exact; world-frame diagonal gains on a tilted table couple in-plane motion into the normal force and would confound tilt with gain anisotropy. Cost: the controller needs n, which the policy also observes. |
| Main gains | K_t = 1000, K_n = 500 N/m; D_t = 110, D_n = 250 N·s/m; K_rot = 100 N·m/rad, D_rot = 10; K_null = 5, D_null = 1 | Prototype values. D_n is sized above free-space critical damping so the tool does not chatter on contact. |
| Orientation error | e_R = vee(skew part of R_d Rᵀ) (= sin θ · axis) | Exact direction, ≈ angle for the small errors here. |
| Null space | N = I − Jᵀ(J Jᵀ)⁻¹J, centring on the IK start configuration | Simple; keeps the elbow from drifting. |
| Orientation target | Held at the IK start orientation | Brief. |
| Start pose | IK: ball centre at the start point, tool axis along −n via `AddAngleBetweenVectorsConstraint`, yaw free, cost ‖q − q_nom‖², 8 deterministic restarts seeded by the scene | Brief. |
| K sweep | K_n ∈ {250, 1000} besides 500, same D and K_t | Brief §2 design issue: with K = 500 and k_e ≥ 2e3 the predictions are degenerate; the sweep changes K/(K+k_e) and K_eff. D kept fixed so only K changes. |

## Task and metric (open decisions 1, 2)

| Item | Choice | Reason |
|---|---|---|
| f_des, f_min, f_max, ε_u | 10 N, 5 N, 20 N, 0.10 | Defaults. f_max = 20 N kept because the expert stays under it on every training table (see Phase 1 in STATUS.md). |
| Peak force | Max of the raw 1 kHz normal force n·f over the whole episode | Strictest; includes impacts. |
| Coverage | 2 mm grid on the L×L square; a cell is wiped when its centre is within r of the contact point (ball centre projected onto the undeformed surface) at a 100 Hz sample with f_n ≥ f_min | 100 Hz keeps consecutive samples ≤ 1.2 mm apart at 12 cm/s. |
| Wiping radius r | 2 cm | A rigid ball in point contact has essentially no contact patch; r is a task parameter that stands for a sponge pad. |
| Region | L ~ U[0.13, 0.17] m, 6 lanes, spacing s = L/5 ∈ [2.6, 3.4] cm < 2r | Lanes run to the region edges, so the swept band covers the square. |
| Speed v | U[0.05, 0.12] m/s on straight segments; path speed dips linearly to 0.35 v within 2 cm (arc length) of every corner | Without the dip, fast lane reversals made the tool overshoot, lose contact and land again: 35.7 N on a training table (k_e = 1e5) and 83 N on k_e = 3e5. With it all 84 test episodes stayed ≤ 16.8 N. Speed never reaches zero, so the velocity still tells the direction at corners. |
| Lane direction t1 (open decision 3) | Fixed: t1 = world x (the tilt axis), t2 = n × t1 | Not randomized, so not observed. |
| Time limit | T_MAX = 50 s for learned policies and baselines, no early termination | The slowest expert episodes (v = 5 cm/s, L ≈ 0.17 m, soft table) take ≈ 40 s; Phase 1 found 40 s truncated the lift in 1 of 50 scenes on some tables. |
| Wiping samples for force statistics | 100 Hz samples whose contact point is within the region grown by r and whose ball bottom is ≤ 5 mm above the undeformed surface | Includes a tool that hovers just above the surface (f = 0), so a label that never presses shows a low mean force instead of no data. |

## Scenes and tables (open decisions 6, 7)

| Item | Choice | Reason |
|---|---|---|
| Training tables | tilt {0°, 10°, 20°} × k_e {500, 2e3, 2e4, 1e5} N/m (12 tables) | Default grid plus a soft surface near K (k_e = 500 = K_n) at every tilt, so (a) keeps 50 % of the force there and P1 can show a trend. |
| OOD tables | (15°, 5e4), (28°, 2e4), (28°, 500), (0°, 250), (10°, 3e5) | Unseen tilt/k_e inside the range; steeper; steeper + soft; softer; harder. |
| Tilt axis | World x; n = (0, −sin θ, cos θ) | Region stays reachable at 28° (IK and the expert succeed there). |
| Scene randomization | Region-centre surface point x ∈ [0.52, 0.60], y ∈ [−0.04, 0.04], z ∈ [0.06, 0.16] m; L; v; start 4–7 cm above the surface with ±1 cm in-plane jitter around p0 | |
| Seeds | `SeedSequence([crc32(table), split, index])`, splits demo / val / test / dev | Demo, validation and test scenes never share a seed. `dev` is for tuning only. |
| Demos per table | 36 kept successful expert episodes: 32 train + 4 validation (split by episode) | Default 30–40. Failed expert episodes are skipped and logged in the manifest. |
| Validation scenes | 10 new scenes per training table (split `val`) for closed-loop checkpoint selection | Test scenes stay untouched until the final evaluation. |
| Test scenes | 50 per table (split `test`) for every method | Brief. |

## Expert (privileged, 20 Hz, zero-order hold)

| Item | Choice | Reason |
|---|---|---|
| Rate | 20 Hz, same as the policy | The logged x_d is exactly what the controller saw, so replaying (b) reproduces the demo exactly. |
| Approach | Target moves along −n at 2 cm/s; 6 mm/s once the ball is within **2 cm** of the surface; until f_n ≥ 1 N | With the prototype's 1 cm zone the tool, lagging the target by D_n v/K_n = 1 cm, still hit at 9 mm/s (21.9 N on k_e = 3e5). |
| Force loop | h_d ← h_d − (f_des − f_n)/(K_n · 0.3 s) · Δt during press and sweep | Brief. |
| Press → sweep | \|f_n − f_des\| < 1 N for 4 steps and ≥ 0.3 s in press (cap 3 s) | |
| Lift | +n at 3 cm/s until the ball is 2 cm above the surface; episode ends | |

## Hybrid force/position baseline (1 kHz)

| Item | Choice | Reason |
|---|---|---|
| Information | p0 (in-plane only), L, t1, t2, n, scene speed v; **not** the table height, **no** demonstrations | Brief. p0 lies on the surface, but the baseline only uses its in-plane coordinates. |
| Normal direction | Approach: velocity servo F_n = D_n(−v_app − ẋ_n) with v_app = 8 mm/s until f_n ≥ 1 N. Then F_n = −(f_ref + k_p(f_ref − f_n) + k_i ∫(f_ref − f_n)) − D_n ẋ_n with k_p = 0.3, k_i = 6 s⁻¹, anti-windup ±10 N, f_ref ramping to f_des over 0.5 s | Blind approach: impact force ≈ v·sqrt(k_e m_eff). At 1.5 cm/s it reached 20.9 N on k_e = 1e5; 8 mm/s gives ≈ 11 N on the stiffest training table (tuned on training tables only). On the OOD k_e = 3e5 table it stays borderline (5/6, max 22 N in a 6-scene dev check). |
| Tangential | Same stiffness/damping as the impedance controller, tracking the same lane path and corner speed profile | Same lane pattern as the expert. |
| Lift | Velocity servo +3 cm/s for 1.5 s | No height knowledge. |
