#!/usr/bin/env bash
# Rollout videos (2x speed) on the same test scene for every method, from traces recorded
# by the final evaluation (--record-q 1 records q for scene 000 of every table).
# Usage: bash results/make_videos.sh
set -euo pipefail
cd "$(dirname "$0")/.."
source env.sh
R=results/raw; V=results/videos
mkdir -p $V
render() { python scripts/render_rollout.py "$@" 2>&1 | grep -v "libEGL\|pci id\|^$" | tail -n 1; }
for scene in tilt10_ke2e4_000 tilt00_ke1e5_000 tilt00_ke500_000; do
  for l in a b c cp; do
    render --trace $R/K500_${l}_s0_test_traces/$scene.npz --title "DP label ($l), K_n=500" --out $V/${scene}_${l}.mp4 --speed 2
  done
  render --trace $R/scripted_K500_test_traces/$scene.npz --title "scripted expert" --out $V/${scene}_scripted.mp4 --speed 2
  render --trace $R/hybrid_K500_test_traces/$scene.npz --title "hybrid force/position" --out $V/${scene}_hybrid.mp4 --speed 2
  render --grid $R/K500_a_s0_test_traces/$scene.npz,$R/K500_b_s0_test_traces/$scene.npz,$R/K500_c_s0_test_traces/$scene.npz,$R/K500_cp_s0_test_traces/$scene.npz \
         --titles "(a) x,(b) x_d,(c) x - K^-1 f,(c') x + K^-1(D xdot - f)" --out $V/${scene}_labels_grid.mp4 --speed 2
done
# steeper unseen tilt (OOD)
render --grid $R/scripted_K500_test_traces/tilt28_ke2e4_000.npz,$R/K500_b_s0_test_traces/tilt28_ke2e4_000.npz,$R/K500_cp_s0_test_traces/tilt28_ke2e4_000.npz \
       --titles "scripted expert,(b) x_d,(c') x + K^-1(D xdot - f)" --out $V/tilt28_ke2e4_000_ood_grid.mp4 --speed 2
