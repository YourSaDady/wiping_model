#!/usr/bin/env bash
# Secondary analysis (pre-registered 2026-10-23 after seeing that label (c) reaches the
# table ~25 s late): time-limit sensitivity. Seed-0 K_n = 500 policies of every label on
# the training-table test scenes with T_MAX = 100 s instead of 50 s. Starts after the
# main test-evaluation queue has finished.
set -euo pipefail
cd "$(dirname "$0")/.."
source env.sh
: "${EVAL_WORKERS:=14}"
until grep -q "K1000_cp_s0 test eval done\|runs/K500_cp_s2 test eval done" logs/eval_queue.log 2>/dev/null && \
      [ "$(grep -c 'test eval done' logs/eval_queue.log)" -ge 20 ]; do sleep 120; done
for l in a b c cp; do
  python scripts/eval_policy.py --run runs/K500_${l}_s0 --ckpt selected --split test --tables train --n 50 \
    --t-max 100 --name K500_${l}_s0_T100 --workers $EVAL_WORKERS >> logs/eval_test_K500_${l}_s0_T100.log 2>&1
done
echo "extra done $(date)"
