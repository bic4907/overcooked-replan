#!/usr/bin/env bash
# Partition observer-none RNN seeds across HPC and aica without overlapping runs.
set -euo pipefail

HOST_PART="${1:?use hpc or aica}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REVISION=dual-handoff-recipe-priority-11x7-v1
grep -qx "LAYOUT_REVISION: $REVISION" "$ROOT/conf/scenario/distance_0.yaml" || {
    echo 'distance_0 revision mismatch' >&2; exit 2;
}
case "$HOST_PART" in
    hpc)
        export PYTHON=/home/jovyan/overcooked-replan-venv/bin/python
        export GPU_ALLOWLIST="0 1 2 3" GPUS="0 1 2 3"
        export SEED_IDS="0 1 2 3"
        ;;
    aica)
        export PYTHON=/home/inchang/overcooked-replan-venv/bin/python
        export GPU_ALLOWLIST="0 1 6 7" GPUS="0 1 6 7"
        export SEED_IDS="4 5 6 7 8 9"
        ;;
    *) echo 'use hpc or aica' >&2; exit 2 ;;
esac

export OBSERVER=none LAYOUTS=distance_0 ALGORITHMS=rnn
export EVAL_EPISODES=20 LAYOUT_REVISION_REQUIRE="$REVISION"
export CAMPAIGN="distance0-priority-0926-rnn-observer-none-$HOST_PART"
export PROJECT_PREFIX=overcooked-v3-distance0-priority-0926-observer-none
export TRAIN_PROJECT_OVERRIDE=overcooked-v3-ippo-rnn-observer-none-0921_train

DONE="$ROOT/orchestrator_distance0_priority_observer_none_${HOST_PART}_train.done"
if [[ ! -f "$DONE" ]]; then
    bash "$ROOT/scripts/run/run_topology_inversion.sh" main train \
        > "$ROOT/orchestrator_distance0_priority_observer_none_${HOST_PART}_train.log" 2>&1
    touch "$DONE"
fi
