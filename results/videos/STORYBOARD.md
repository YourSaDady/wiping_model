# Video storyboard (≈ 5 min, 3 speakers)

Shot list only. Per the course communication policy, narration must be written and spoken
by the team; nothing here is a script. "Show" lists the material; "Point to make" is a
reminder of the content, in note form. Suggested speakers follow the proposal's division of
work (Linda: environment, controller, metric, hybrid baseline; Yichuan: expert, data, labels,
P1–P4 analysis; Yanchuan: policy training, closed-loop evaluation). Time is split evenly.

All clips are 2× real time; each panel shows the Drake camera view, the normal force
(dotted f_min = 5 N, solid f_des = 10 N, dashed f_max = 20 N) and the coverage map.

| # | time | speaker | show | point to make (notes) |
|---|---|---|---|---|
| 1 | 0:00–0:25 | Linda | title card; `tilt10_ke2e4_000_scripted.mp4` (first 8 s) | wiping needs force; impedance control = spring K toward a target; target must be below the surface |
| 2 | 0:25–0:55 | Linda | proposal Fig. 1 side view (redraw); 1-D sketch: K(x_d − x) + f = 0, d_t = 2 cm at 10 N | the four labels; what a robot dataset usually logs |
| 3 | 0:55–1:40 | Linda | Drake scene: tables of different tilt/stiffness (colour = stiffness), region outline; controller equation; `tilt00_ke1e5_000_hybrid.mp4` | table-frame gains; effective mass ≈ 18 kg; damping sized for contact; hybrid baseline does not know the height (8 mm/s approach) |
| 4 | 1:40–2:15 | Yichuan | `results/figures/phase1_replay_force.png` | open-loop replay: (b) exact, (a) keeps K/(K+k_e), (c) overshoots at onset, (c′) fine |
| 5 | 2:15–2:40 | Yichuan | `results/figures/p2_label_gap.png` | (c) − (b) gap ≈ D v/K grows with speed; (c′) leaves the inertial part (≈ 4 mm) |
| 6 | 2:40–3:30 | Yanchuan | `tilt00_ke1e5_000_labels_grid.mp4` (full 25 s) | same scene, same network, only the label differs; (a)/(c) hover for 20+ s, (b) wipes, (c′) chatters and gives up |
| 7 | 3:30–3:55 | Yanchuan | `results/tables/main_500.md` as a slide (ID/OOD success with CIs, 3 seeds) | headline numbers; Fisher/McNemar; DP architecture box (2 obs frames → 16-step chunk, execute 8) |
| 8 | 3:55–4:20 | Yanchuan | `results/figures/p2_closed_loop.png` (middle + right) | why (c′) fails: velocity fed into the target cancels 2/3 of the damping; K sweep: (c) works at K = 1000 |
| 9 | 4:20–4:40 | Yichuan | `results/figures/p4_approach.png`, `results/figures/p3_steady_spread.png` | P4: (c) creeps at 1.4 mm/s; P3: spread grows with stiffness, not only via K_eff |
| 10 | 4:40–5:00 | Yichuan / all | `tilt28_ke2e4_000_ood_grid.mp4` (8 s) + takeaway slide | steeper unseen tilt: (b) drifts below the region; baselines fine; takeaway and limitations |

Other clips available in `results/videos/`: every label and both baselines on three scenes
(`tilt10_ke2e4_000`, `tilt00_ke1e5_000` hard, `tilt00_ke500_000` soft sponge), plus label grids.
Re-render with `bash results/make_videos.sh` (needs the test traces in `results/raw/`).
