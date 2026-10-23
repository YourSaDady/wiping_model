#!/usr/bin/env bash
# Train one policy and select its checkpoint on validation scenes. Idempotent:
# finished stages are skipped, training resumes from ckpt_last.pt.
# Usage: scripts/run_one.sh <label> <K_n> <seed> [extra train_dp.py args...]
set -euo pipefail
cd "$(dirname "$0")/.."
source env.sh
label=$1; K=$2; seed=$3; shift 3
: "${STEPS:=40000}"
: "${SEL_WORKERS:=12}"
: "${TAG:=}"
run=runs/K${K}_${label}_s${seed}${TAG}
mkdir -p logs
if [ ! -f "$run/DONE" ]; then
  python scripts/train_dp.py --label "$label" --K "$K" --seed "$seed" --steps "$STEPS" --out "$run" "$@" \
    >> "logs/$(basename $run)_train.log" 2>&1
fi
if [ ! -f "$run/selected.json" ]; then
  python scripts/select_ckpt.py --run "$run" --workers "$SEL_WORKERS" >> "logs/$(basename $run)_select.log" 2>&1
fi
echo "$run done: $(python -c "import json;d=json.load(open('$run/selected.json'));print(d['step'],d['val_success'],'/',d['val_n'])")"
