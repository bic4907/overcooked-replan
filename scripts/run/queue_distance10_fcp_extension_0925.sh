#!/usr/bin/env bash
# Wait for the shared partner population and free aica GPUs 0/1, then train seeds 4-9.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PILOT="$ROOT/campaigns/distance10-priority-0925-fcp-pilot4/main/both"
CNN_EVAL_DONE="$ROOT/orchestrator_distance10_cnn_full10_eval.done"
FCP_EXTENSION_DONE="$ROOT/orchestrator_distance10_fcp_extension6_train.done"

while [[ ! -f "$FCP_EXTENSION_DONE" ]]; do
    population_complete=1
    for seed in 100 101 102 103 104 105; do
        [[ -f "$PILOT/state/population_distance_10_seed${seed}.done" ]] || population_complete=0
    done
    free_gpus="$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits |
        awk -F, '$1 == 0 || $1 == 1 { gsub(/ /, "", $2); if ($2 < 500) n++ } END { print n + 0 }')"
    if [[ "$population_complete" == 1 && -f "$CNN_EVAL_DONE" && "$free_gpus" == 2 ]]; then
        export GPUS="0 1"
        bash "$ROOT/scripts/run/orchestrate_distance10_fcp_extension_0925.sh" train
        break
    fi
    sleep 30
done
