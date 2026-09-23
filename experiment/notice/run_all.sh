#!/usr/bin/env bash
# Run both advance-notice experiments on the six benchmark kitchens.
#   bash experiment/notice/run_all.sh probe rnn fcp     # Section 6.2
#   bash experiment/notice/run_all.sh behavior rnn      # Section 6.3
# A cell whose output file already exists is skipped.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY="${PY:-/home/cilab/anaconda3/envs/overcooked-replan/bin/python}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
what="$1"; shift
mkdir -p outputs/notice/logs
for arm in "$@"; do
  for layout in split_0 split_1 outage_0 outage_1 distance_0 distance_1; do
    log="outputs/notice/logs/${what}_${layout}_${arm}.log"
    case "$what" in
      probe)    done_file="outputs/notice/probe_${layout}_${arm}_seat0.jsonl" ;;
      behavior) done_file="outputs/notice/behavior_${layout}_${arm}.jsonl" ;;
    esac
    if [ -f "$done_file" ]; then echo "== $what $layout $arm already done"; continue; fi
    echo "== $what $layout $arm -> $log"
    case "$what" in
      probe)    "$PY" experiment/notice/probe_preparation.py --layout "$layout" --arm "$arm" 2>&1 | grep -v "WARNING\|Submoduled\|Importing\|wandb:" | tee "$log" ;;
      behavior) "$PY" experiment/notice/behavior_as_notice.py --layout "$layout" --arm "$arm" 2>&1 | grep -v "WARNING\|Submoduled\|Importing\|wandb:" | tee "$log" ;;
    esac
  done
done
