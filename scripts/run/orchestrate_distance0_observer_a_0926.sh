#!/usr/bin/env bash
# Repeat the original 0921 IPPO-RNN observer-a experiment on the new distance_0.
# Launch only after the canonical both-observer CNN/RNN/FCP baseline is complete.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REVISION=dual-handoff-recipe-priority-11x7-v1
if ! grep -qx "LAYOUT_REVISION: $REVISION" "$ROOT/conf/scenario/distance_0.yaml"; then
    echo 'distance_0 scenario revision does not match the priority map' >&2
    exit 2
fi

export PYTHON="${PYTHON:-/home/jovyan/overcooked-replan-venv/bin/python}"
export GPU_ALLOWLIST="0 1 2 3" GPUS="0 1 2 3"
export OBSERVER=agent_0 LAYOUTS=distance_0 ALGORITHMS=rnn
export SEED_IDS="0 1 2 3 4 5 6 7 8 9"
export EVAL_EPISODES=20 LAYOUT_REVISION_REQUIRE="$REVISION"
export CAMPAIGN=distance0-priority-0926-rnn-observer-a-full10
export PROJECT_PREFIX=overcooked-v3-distance0-priority-0926-observer-a
export TRAIN_PROJECT_OVERRIDE=overcooked-v3-ippo-rnn-observer-a-0921_train
export EVAL_PROJECT_OVERRIDE=overcooked-v3-ippo-rnn-observer-a-0921_eval

TRAIN_DONE="$ROOT/orchestrator_distance0_priority_observer_a_train.done"
EVAL_DONE="$ROOT/orchestrator_distance0_priority_observer_a_eval.done"
if [[ ! -f "$TRAIN_DONE" ]]; then
    bash "$ROOT/scripts/run/run_topology_inversion.sh" main train \
        > "$ROOT/orchestrator_distance0_priority_observer_a_train.log" 2>&1
    touch "$TRAIN_DONE"
fi
if [[ ! -f "$EVAL_DONE" ]]; then
    bash "$ROOT/scripts/run/run_topology_inversion.sh" main eval \
        > "$ROOT/orchestrator_distance0_priority_observer_a_eval.log" 2>&1
    touch "$EVAL_DONE"
fi
