#!/usr/bin/env bash
# Phase 6 stretch runs. Waits for the 20 main sweep runs to finish training, then
# trains + selects the runs in scripts/stretch_list.txt (3 at a time; first field is
# the run-name TAG, "-" for none), then evaluates on the test split:
#   pose+force (pf) policies deployed at K' = 500, 250, 1000;
#   (b) and (c') K_n = 500 seed-0 policies deployed at the mismatched K' = 250, 1000;
#   noisy-force (c) policies at K_n = 500.
set -euo pipefail
cd "$(dirname "$0")/.."
source env.sh
: "${EVAL_WORKERS:=14}"
until [ "$(while read -r l K s; do [ -f runs/K${K}_${l}_s${s}/DONE ] && echo x; done < scripts/sweep_list.txt | wc -l)" -ge 20 ]; do
  sleep 120
done
echo "main sweep trained $(date)"
xargs -P 3 -L 1 sh -c 't=$1; shift; [ "$t" = "-" ] && t=""; TAG=$t exec scripts/run_one.sh "$@"' _ < scripts/stretch_list.txt
echo "stretch trained $(date)"
ev() { python scripts/eval_policy.py --ckpt selected --split test --tables all --n 50 --workers $EVAL_WORKERS "$@"; }
for s in 0 1 2; do
  for Kp in 500 250 1000; do ev --run runs/K500_pf_s$s --deploy-K $Kp >> logs/eval_test_K500_pf_s${s}_dK$Kp.log 2>&1; done
done
for l in b cp; do
  for Kp in 250 1000; do ev --run runs/K500_${l}_s0 --deploy-K $Kp >> logs/eval_test_K500_${l}_s0_dK$Kp.log 2>&1; done
done
for t in _fn1 _fn3 _fbp2 _fbm2; do ev --run runs/K500_c_s0$t >> logs/eval_test_K500_c_s0$t.log 2>&1; done
echo "stretch evals done $(date)"
