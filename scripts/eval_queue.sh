#!/usr/bin/env bash
# Final test-scene evaluation queue (Phase 5). Baselines first, then every run in
# scripts/sweep_list.txt as soon as its checkpoint has been selected on validation
# scenes. Each evaluation is resumable (skips episodes already in its JSONL file).
# Usage: scripts/eval_queue.sh   (env: EVAL_WORKERS, default 14)
set -euo pipefail
cd "$(dirname "$0")/.."
source env.sh
: "${EVAL_WORKERS:=14}"
for K in 500 250 1000; do
  python scripts/eval_baselines.py --method scripted --K $K --split test --tables all --n 50 \
    --workers $EVAL_WORKERS --traces 3 --record-q 1 >> logs/eval_test_scripted_K$K.log 2>&1
done
python scripts/eval_baselines.py --method hybrid --K 500 --split test --tables all --n 50 \
  --workers $EVAL_WORKERS --traces 3 --record-q 1 >> logs/eval_test_hybrid_K500.log 2>&1
echo "baselines done $(date)"
while read -r label K seed; do
  run=runs/K${K}_${label}_s${seed}
  until [ -f "$run/selected.json" ]; do sleep 60; done
  python scripts/eval_policy.py --run "$run" --ckpt selected --split test --tables all --n 50 \
    --workers $EVAL_WORKERS --traces 3 --record-q 1 >> "logs/eval_test_$(basename $run).log" 2>&1
  echo "$run test eval done $(date): $(tail -n 1 logs/eval_test_$(basename $run).log)"
done < scripts/sweep_list.txt
