#!/usr/bin/env bash
# Fresh 10-seed canonical distance_0 baseline in the original 0921 projects.
# Run cnn/fcp on aica and rnn on HPC from isolated source snapshots.
set -euo pipefail

KIND="${1:?use cnn, rnn, or fcp}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REVISION=dual-handoff-recipe-priority-11x7-v1
if ! grep -qx "LAYOUT_REVISION: $REVISION" "$ROOT/conf/scenario/distance_0.yaml"; then
    echo 'distance_0 scenario revision does not match the priority map' >&2
    exit 2
fi

case "$KIND" in
    cnn)
        export GPUS="0 1" GPU_ALLOWLIST="0 1 6 7"
        export PYTHON="${PYTHON:-/home/inchang/overcooked-replan-venv/bin/python}"
        export TRAIN_PROJECT_OVERRIDE=overcooked-v3-ippo-0921_train
        export EVAL_PROJECT_OVERRIDE=overcooked-v3-ippo-0921_eval
        ;;
    fcp)
        export GPUS="6 7" GPU_ALLOWLIST="0 1 6 7"
        export PYTHON="${PYTHON:-/home/inchang/overcooked-replan-venv/bin/python}"
        export TRAIN_PROJECT_OVERRIDE=overcooked-v3-fcp-0921_train
        export EVAL_PROJECT_OVERRIDE=overcooked-v3-fcp-0921_eval
        export POPULATION_PROJECT_OVERRIDE=overcooked-v3-fcp-0921-population
        export POPULATION_WANDB_MODE=online
        ;;
    rnn)
        export GPUS="0 1 2 3" GPU_ALLOWLIST="0 1 2 3"
        export PYTHON="${PYTHON:-/home/jovyan/overcooked-replan-venv/bin/python}"
        export TRAIN_PROJECT_OVERRIDE=overcooked-v3-ippo-rnn-0921_train
        export EVAL_PROJECT_OVERRIDE=overcooked-v3-ippo-rnn-0921_eval
        ;;
    *) echo 'use cnn, rnn, or fcp' >&2; exit 1 ;;
esac

export OBSERVER=both LAYOUTS=distance_0 ALGORITHMS="$KIND"
export SEED_IDS="0 1 2 3 4 5 6 7 8 9"
export POP_SEED_IDS="100 101 102 103 104 105"
export EVAL_EPISODES=20 LAYOUT_REVISION_REQUIRE="$REVISION"
export CAMPAIGN="distance0-priority-0926-${KIND}-full10"
export PROJECT_PREFIX=overcooked-v3-distance0-priority-0926

TRAIN_DONE="$ROOT/orchestrator_distance0_priority_${KIND}_train.done"
EVAL_DONE="$ROOT/orchestrator_distance0_priority_${KIND}_eval.done"
if [[ ! -f "$TRAIN_DONE" ]]; then
    bash "$ROOT/scripts/run/run_topology_inversion.sh" main train \
        > "$ROOT/orchestrator_distance0_priority_${KIND}_train.log" 2>&1
    touch "$TRAIN_DONE"
fi
if [[ ! -f "$EVAL_DONE" ]]; then
    bash "$ROOT/scripts/run/run_topology_inversion.sh" main eval \
        > "$ROOT/orchestrator_distance0_priority_${KIND}_eval.log" 2>&1
    touch "$EVAL_DONE"
fi
