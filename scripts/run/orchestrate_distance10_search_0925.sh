#!/usr/bin/env bash
# Four-seed diagnostic on a separate W&B project; training resumes by markers.
set -euo pipefail

KIND="${1:?use cnn, rnn, or fcp}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
case "$KIND" in
    cnn)
        export GPUS="0 1" GPU_ALLOWLIST="0 1 6 7"
        export PYTHON="${PYTHON:-/home/inchang/overcooked-replan-venv/bin/python}"
        ;;
    fcp)
        export GPUS="6 7" GPU_ALLOWLIST="0 1 6 7"
        export PYTHON="${PYTHON:-/home/inchang/overcooked-replan-venv/bin/python}"
        ;;
    rnn)
        export GPUS="0 1 2 3" GPU_ALLOWLIST="0 1 2 3"
        export PYTHON="${PYTHON:-/home/jovyan/overcooked-replan-venv/bin/python}"
        ;;
    *) echo "use cnn, rnn, or fcp" >&2; exit 1 ;;
esac

export OBSERVER=both
export LAYOUTS=distance_10
export ALGORITHMS="$KIND"
export SEED_IDS="0 1 2 3"
export POP_SEED_IDS="100 101 102 103 104 105"
export EVAL_EPISODES=20
export LAYOUT_REVISION_REQUIRE=dual-handoff-recipe-priority-11x7-v1
export POPULATION_WANDB_MODE=online
export PROJECT_PREFIX=overcooked-v3-mapsearch-0925-d10-priority-pilot4
export CAMPAIGN="distance10-priority-0925-${KIND}-pilot4"

TRAIN_DONE="$ROOT/orchestrator_distance10_${KIND}_train.done"
EVAL_DONE="$ROOT/orchestrator_distance10_${KIND}_eval.done"
if [[ ! -f "$TRAIN_DONE" ]]; then
    bash "$ROOT/scripts/run/run_topology_inversion.sh" main train \
        > "$ROOT/orchestrator_distance10_${KIND}_train.log" 2>&1
    touch "$TRAIN_DONE"
fi
if [[ ! -f "$EVAL_DONE" ]]; then
    bash "$ROOT/scripts/run/run_topology_inversion.sh" main eval \
        > "$ROOT/orchestrator_distance10_${KIND}_eval.log" 2>&1
    touch "$EVAL_DONE"
fi
