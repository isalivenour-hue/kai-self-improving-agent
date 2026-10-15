#!/usr/bin/env bash
set -u
root="$HOME/ml-sandbox"
log="$root/state/continuous.log"
while true; do
  date -Is >> "$log"
  "$root/run-isolated.sh" experiments/persistent_neural_network.py >> "$log" 2>&1 || echo "supervised run failed: $?" >> "$log"
  "$root/run-isolated.sh" experiments/reinforcement_learning.py >> "$log" 2>&1 || echo "reinforcement run failed: $?" >> "$log"
  "$root/run-isolated.sh" experiments/web_text_learner.py >> "$log" 2>&1 || echo "web text learner failed: $?" >> "$log"
  "$root/run-isolated.sh" experiments/web_corpus_ingest.py >> "$log" 2>&1 || echo "web corpus ingest failed: $?" >> "$log"
  "$root/run-isolated.sh" experiments/web_retriever.py >> "$log" 2>&1 || echo "web retriever failed: $?" >> "$log"
  "$root/run-isolated.sh" experiments/deep_neural_network.py >> "$log" 2>&1 || echo "deep learner failed: $?" >> "$log"
  "$root/run-isolated.sh" experiments/meta_controller.py >> "$log" 2>&1 || echo "meta controller failed: $?" >> "$log"
  "$root/run-isolated.sh" experiments/memory_consolidator.py >> "$log" 2>&1 || echo "memory consolidation failed: $?" >> "$log"
  "$root/run-isolated.sh" experiments/backup_state.py >> "$log" 2>&1 || echo "backup failed: $?" >> "$log"
  "$root/run-isolated.sh" experiments/self_improvement.py >> "$log" 2>&1 || echo "self-improvement run failed: $?" >> "$log"
  sleep 60
done
