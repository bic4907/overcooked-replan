#!/usr/bin/env bash
# Replace the original 0921 observer-none RNN distance_0 experiment on the
# canonical priority layout. Run only from an isolated HPC source snapshot.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REVISION=dual-handoff-recipe-priority-11x7-v1
if ! grep -qx "LAYOUT_REVISION: $REVISION" "$ROOT/conf/scenario/distance_0.yaml"; then
    echo 'distance_0 scenario revision mismatch' >&2
    exit 2
fi

export PYTHON="${PYTHON:-/home/jovyan/overcooked-replan-venv/bin/python}"
export GPU_ALLOWLIST="0 1 2 3" GPUS="0 1 2 3"
export OBSERVER=none LAYOUTS=distance_0 ALGORITHMS=rnn
export SEED_IDS="0 1 2 3 4 5 6 7 8 9"
export EVAL_EPISODES=20 LAYOUT_REVISION_REQUIRE="$REVISION"
export CAMPAIGN=distance0-priority-0926-rnn-observer-none-full10
export PROJECT_PREFIX=overcooked-v3-distance0-priority-0926-observer-none
export TRAIN_PROJECT_OVERRIDE=overcooked-v3-ippo-rnn-observer-none-0921_train
export EVAL_PROJECT_OVERRIDE=overcooked-v3-ippo-rnn-observer-none-0921_test10

TRAIN_DONE="$ROOT/orchestrator_distance0_priority_observer_none_train.done"
EVAL10_DONE="$ROOT/orchestrator_distance0_priority_observer_none_test10.done"
EVAL6_DONE="$ROOT/orchestrator_distance0_priority_observer_none_test.done"

if [[ ! -f "$TRAIN_DONE" ]]; then
    bash "$ROOT/scripts/run/run_topology_inversion.sh" main train \
        > "$ROOT/orchestrator_distance0_priority_observer_none_train.log" 2>&1
    touch "$TRAIN_DONE"
fi
if [[ ! -f "$EVAL10_DONE" ]]; then
    bash "$ROOT/scripts/run/run_topology_inversion.sh" main eval \
        > "$ROOT/orchestrator_distance0_priority_observer_none_test10.log" 2>&1
    touch "$EVAL10_DONE"
fi
if [[ ! -f "$EVAL6_DONE" ]]; then
    cd "$ROOT"
    export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    export XLA_PYTHON_CLIENT_MEM_FRACTION=0.75
    env -u LD_LIBRARY_PATH "$PYTHON" -u baselines/IPPO/eval_crossplay_overcooked_v3.py \
        "cilab-overcooked/$TRAIN_PROJECT_OVERRIDE" \
        --algorithms IPPO --layout distance_0 --seeds 0 1 2 3 4 5 \
        --transition-observer none --episodes 20 --max-steps 450 \
        --layout-revision "$REVISION" --gpus 0 --workers-per-gpu 4 \
        --wandb-mode online \
        --output-project cilab-overcooked/overcooked-v3-ippo-rnn-observer-none-0921_test \
        --output-dir "$ROOT/campaigns/$CAMPAIGN/main/none/evaluation6/rnn/distance_0" \
        --save-adaptation-traces --adaptation-horizon 75 --adaptation-window 30 \
        --drop-baseline-window 60 --drop-horizon 60 \
        > "$ROOT/orchestrator_distance0_priority_observer_none_test.log" 2>&1
    touch "$EVAL6_DONE"
fi
