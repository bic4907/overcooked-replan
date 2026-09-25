#!/usr/bin/env bash
# Queue the both-observer 30M-step distance_0 RNN baseline after the active
# distance_9 RNN campaign releases HPC GPUs 0-3.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if ! grep -q '^LAYOUT_REVISION: dual-handoff-private-pots-11x7-v1$' "$ROOT/conf/scenario/distance_0.yaml"; then
    echo 'This private-pot distance_0 campaign is retired; distance_0 is the original 0921 layout again.' >&2
    exit 2
fi
PREDECESSOR=/home/jovyan/handoff-d9-pilot4-0925/orchestrator_d9_rnn_seed10.done
QUEUE_LOG="$ROOT/orchestrator_distance0_rnn_queue.log"
TRAIN_LOG="$ROOT/orchestrator_distance0_rnn_train.log"
EVAL_LOG="$ROOT/orchestrator_distance0_rnn_eval.log"
TRAIN_DONE="$ROOT/orchestrator_distance0_rnn_train.done"
EVAL_DONE="$ROOT/orchestrator_distance0_rnn_eval.done"

while [[ ! -f "$PREDECESSOR" ]]; do
    printf '[%s] waiting for distance_9 RNN evaluation\n' "$(date -Is)" >> "$QUEUE_LOG"
    sleep 300
done
while ! nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
    | head -n 4 | awk '$1 >= 1000 { busy=1 } END { exit busy }'; do
    printf '[%s] waiting for HPC GPUs 0-3\n' "$(date -Is)" >> "$QUEUE_LOG"
    sleep 300
done

export PYTHON="${PYTHON:-/home/jovyan/overcooked-replan-venv/bin/python}"
export GPU_ALLOWLIST="0 1 2 3"
export GPUS="0 1 2 3"
export OBSERVER=both
export LAYOUTS=distance_0
export ALGORITHMS=rnn
export SEED_IDS="0 1 2 3 4 5 6 7 8 9"
export CAMPAIGN=distance0-privatepots-0925-both-rnn10
export TRAIN_PROJECT_OVERRIDE=overcooked-v3-ippo-rnn-0921_train
export EVAL_PROJECT_OVERRIDE=overcooked-v3-ippo-rnn-0921_eval
export LAYOUT_REVISION_REQUIRE=dual-handoff-private-pots-11x7-v1
export EVAL_EPISODES=20

if [[ ! -f "$TRAIN_DONE" ]]; then
    bash "$ROOT/scripts/run/run_topology_inversion.sh" main train > "$TRAIN_LOG" 2>&1
    touch "$TRAIN_DONE"
fi
if [[ ! -f "$EVAL_DONE" ]]; then
    bash "$ROOT/scripts/run/run_topology_inversion.sh" main eval > "$EVAL_LOG" 2>&1
    touch "$EVAL_DONE"
fi
