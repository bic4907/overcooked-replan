#!/usr/bin/env bash
# The Inversion probes with the informed agent in seat 1 -- the supplier,
# whose switch to plate work the paper's endpoint reads -- alongside the
# seat-0 cells run by run_all.sh.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY="${PY:-/home/cilab/anaconda3/envs/overcooked-replan/bin/python}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
mkdir -p outputs/notice/logs
run() { # layout arm seat
  out="outputs/notice/probe_$1_$2_seat$3.jsonl"
  if [ -f "$out" ]; then echo "== probe $1 $2 seat$3 already done"; return; fi
  echo "== probe $1 $2 seat$3"
  "$PY" experiment/notice/probe_preparation.py --layout "$1" --arm "$2" --informed-seat "$3" 2>&1 \
    | grep -v "WARNING\|Submoduled\|Importing\|wandb:" | tee "outputs/notice/logs/probe_$1_$2_seat$3.log"
}
for arm in rnn fcp; do
  run distance_0 $arm 1; run distance_1 $arm 1
done
