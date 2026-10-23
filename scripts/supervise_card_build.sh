#!/bin/zsh
# Supervisor for the 250Q FIC card build. Auto-restarts on crash AND on stall
# (process alive but no new card written for STALL_SECS — happens when a network
# blip leaves an API call hung with no timeout). Resumes via --skip-existing-core,
# so no completed card is ever recomputed. Safe to run under nohup; survives
# disconnects (but NOT a full machine shutdown).
set -u
cd .
set -a; source .env; set +a

CORE=results/paper_v2_250/fic/core.jsonl
LOG=results/paper_v2_250/fic/build_fic.log
SUPLOG=results/paper_v2_250/fic/supervisor.log
TARGET=250
STALL_SECS=90

count() { wc -l < "$CORE" 2>/dev/null | tr -d ' ' || echo 0; }
log() { echo "[$(date '+%T')] $1" >> "$SUPLOG"; }

log "supervisor start; cards=$(count)/$TARGET"
while :; do
  n=$(count); [ "${n:-0}" -ge "$TARGET" ] && { log "DONE $n/$TARGET"; break; }

  .venv/bin/python -m verifiquant.preprocessing.dataset_case_to_fic \
    --input results/paper_v2_250/questions_250.jsonl \
    --functions-catalog-path verifiquant/data/functions-article-all.json \
    --financial-docs-path verifiquant/data/financial_documents.json \
    --core-output "$CORE" \
    --retrieval-output results/paper_v2_250/fic/retrieval.jsonl \
    --repair-output results/paper_v2_250/fic/repair.jsonl \
    --duplicate-fic-policy suffix --on-validation-error save \
    --validation-report results/paper_v2_250/fic/validation_report.json \
    --seed-report-output results/paper_v2_250/fic/seed_report.json \
    --skip-existing-core --checkpoint-every-record \
    >> "$LOG" 2>&1 &
  BPID=$!
  log "launched build pid=$BPID at $(count)/$TARGET"

  # watchdog: kill the build if it stalls (no new card within STALL_SECS)
  last=$(count); last_t=$(date +%s)
  while kill -0 "$BPID" 2>/dev/null; do
    sleep 15
    cur=$(count)
    [ "${cur:-0}" -ge "$TARGET" ] && break
    if [ "$cur" -gt "$last" ]; then last=$cur; last_t=$(date +%s); fi
    if [ $(( $(date +%s) - last_t )) -gt "$STALL_SECS" ]; then
      log "STALL at $cur/$TARGET (no card in ${STALL_SECS}s) -> kill pid=$BPID"
      kill -9 "$BPID" 2>/dev/null
      break
    fi
  done
  wait "$BPID" 2>/dev/null
  log "build pid=$BPID exited; cards=$(count)/$TARGET; restarting if <$TARGET"
  sleep 2
done
log "supervisor exit; cards=$(count)/$TARGET"
