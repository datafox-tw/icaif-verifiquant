#!/bin/bash
# Wave 2-A: gate ablation ladder on 250Q (K=1, Flash, full 250-card library lib_100).
# Cumulative rungs isolate each gate's contribution. Each rung is a fresh process so the
# module-level VQ_DISABLE_GATES is read correctly. ~2 h total.
# Usage: GEMINI_API_KEY=... bash scripts/run_gate_ladder.sh
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
Q=results/paper_v2_250/questions_250.jsonl
LIB=results/libsweep/lib_100/cards.db
OUT=results/gate_ladder
LOG=$OUT/logs
mkdir -p "$LOG"

rung() {  # rung <tag> <disable> [hint]
  local tag="$1" disable="$2" hint="${3:-}"
  echo "=== [$(date '+%F %T')] RUNG $tag (disable='$disable' hint='$hint') start ==="
  VQ_DISABLE_GATES="$disable" VQ_HINT_TRIGGERS="$hint" PYTHONPATH=. \
    $PY scripts/run_vq50_pro.py --tag "$tag" --model gemini-2.5-flash \
      --questions "$Q" --lib-db "$LIB" --out-dir "$OUT" > "$LOG/$tag.log" 2>&1
  echo "=== [$(date '+%F %T')] RUNG $tag done (exit $?) ==="
}

rung r1_compute_only "MN,F,E,I" ""
rung r2_fe          "MN,I"      ""
rung r3_fei         "MN"        ""
rung r4_full        ""          ""
rung r5_full_hint   ""          "1"
echo "=== GATE LADDER COMPLETE $(date '+%F %T') ==="
