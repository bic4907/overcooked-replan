#!/usr/bin/env bash
# Run the 30M-step private-pot distance_0 baseline with the original
# both-agent transition warning condition. Existing agent_0 pilots are separate.
set -euo pipefail

KIND="${1:?use cnn or fcp}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHON="${PYTHON:-/home/inchang/overcooked-replan-venv/bin/python}"
export GPU_ALLOWLIST="0 1 6 7"
export OBSERVER=both
export LAYOUTS=distance_0
export POP_SEED_IDS="100 101 102 103 104 105"
export EVAL_EPISODES=20
export LAYOUT_REVISION_REQUIRE=dual-handoff-private-pots-11x7-v1

case "$KIND" in
    cnn)
        export GPUS="0 1"
        export ALGORITHMS=cnn
        export SEED_IDS="0 1 2 3 4 5 6 7 8 9"
        export CAMPAIGN=distance0-privatepots-0925-both-cnn10
        export TRAIN_PROJECT_OVERRIDE=overcooked-v3-ippo-0921_train
        export EVAL_PROJECT_OVERRIDE=overcooked-v3-ippo-0921_eval
        ;;
    fcp)
        export GPUS="6 7"
        export ALGORITHMS=fcp
        export SEED_IDS="0 1 2 3 4 5 6 7 8 9"
        export CAMPAIGN=distance0-privatepots-0925-both-fcp10
        export TRAIN_PROJECT_OVERRIDE=overcooked-v3-fcp-0921_train
        export EVAL_PROJECT_OVERRIDE=overcooked-v3-fcp-0921_eval
        export POPULATION_PROJECT_OVERRIDE=overcooked-v3-fcp-0921-population
        export POPULATION_WANDB_MODE=online
        ;;
    *) echo "use cnn or fcp" >&2; exit 1 ;;
esac

TRAIN_LOG="$ROOT/orchestrator_distance0_${KIND}_train.log"
EVAL_LOG="$ROOT/orchestrator_distance0_${KIND}_eval.log"
TRAIN_DONE="$ROOT/orchestrator_distance0_${KIND}_train.done"
EVAL_DONE="$ROOT/orchestrator_distance0_${KIND}_eval.done"
if [[ ! -f "$TRAIN_DONE" ]]; then
    bash "$ROOT/scripts/run/run_topology_inversion.sh" main train > "$TRAIN_LOG" 2>&1
    touch "$TRAIN_DONE"
fi
if [[ ! -f "$EVAL_DONE" ]]; then
    bash "$ROOT/scripts/run/run_topology_inversion.sh" main eval > "$EVAL_LOG" 2>&1
    touch "$EVAL_DONE"
fi
