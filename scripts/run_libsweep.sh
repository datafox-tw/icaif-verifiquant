#!/bin/bash
# Wave 1-A: library-size sweep. Runs all 250Q against each nested subset library
# (10/25/50/75/100%), no CoT baseline (measured separately). ~4-6 h total.
# Usage: GEMINI_API_KEY=... bash scripts/run_libsweep.sh
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
LOG_DIR=results/libsweep
mkdir -p "$LOG_DIR"

for frac in 10 25 50 75 100; do
  echo "=== [$(date '+%F %T')] sweep point lib_${frac} start ==="
  PYTHONPATH=. $PY scripts/run_ood_family_eval.py \
    --lib-db "results/libsweep/lib_${frac}/cards.db" \
    --limit-ood 999 --limit-id 999 --no-baseline \
    --out-dir "results/libsweep/results_${frac}" \
    > "$LOG_DIR/sweep_${frac}.log" 2>&1
  echo "=== [$(date '+%F %T')] lib_${frac} done (exit $?) ==="
done
echo "=== SWEEP COMPLETE $(date '+%F %T') ==="
