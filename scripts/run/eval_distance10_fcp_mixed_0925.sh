#!/usr/bin/env bash
# Evaluate both ordered directions of FCP x IPPO-CNN/RNN on distance_10.
set -euo pipefail

KIND="${1:?use cnn or rnn}"
SCALE="${2:?use pilot4 or full10}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-$HOME/overcooked-replan-venv/bin/python}"
ENTITY=cilab-overcooked
REVISION=dual-handoff-recipe-priority-11x7-v1

case "$KIND" in cnn|rnn) ;; *) echo "use cnn or rnn" >&2; exit 1 ;; esac
case "$SCALE" in
    pilot4) SEEDS=(0 1 2 3) ;;
    full10) SEEDS=(0 1 2 3 4 5 6 7 8 9) ;;
    *) echo "use pilot4 or full10" >&2; exit 1 ;;
esac
read -r -a GPU_LIST <<< "${GPUS:?set GPU IDs explicitly}"
(( ${#GPU_LIST[@]} > 0 )) || exit 1
if [[ -n "${GPU_ALLOWLIST:-}" ]]; then
    for gpu in "${GPU_LIST[@]}"; do
        case " $GPU_ALLOWLIST " in
            *" $gpu "*) ;;
            *) echo "GPU $gpu is outside GPU_ALLOWLIST=$GPU_ALLOWLIST" >&2; exit 1 ;;
        esac
    done
fi

PREFIX="overcooked-v3-mapsearch-0925-d10-priority-${SCALE}"
OUTPUT_DIR="$ROOT/campaigns/distance10-priority-0925-mixed-${SCALE}/${KIND}-fcp"
FCP_SOURCES=(--additional-source "$ENTITY/$PREFIX-fcp_train" distance_10)
if [[ "$SCALE" == full10 ]]; then
    # Pilot seeds 0-3 and extension seeds 4-9 share the same population.
    FCP_SOURCES=(
        --additional-source "$ENTITY/overcooked-v3-mapsearch-0925-d10-priority-pilot4-fcp_train" distance_10
        --additional-source "$ENTITY/$PREFIX-fcp_train" distance_10
    )
fi
mkdir -p "$OUTPUT_DIR"
if [[ -f "$OUTPUT_DIR/eval.done" ]]; then
    echo "Already evaluated: $OUTPUT_DIR"
    exit 0
fi

if [[ -z "${WANDB_API_KEY:-}" && -f "$HOME/.netrc" ]]; then
    WANDB_API_KEY="$("$PYTHON" -c 'import netrc; a=netrc.netrc().authenticators("api.wandb.ai"); print(a[2] if a else "")')"
    export WANDB_API_KEY
fi
[[ -n "${WANDB_API_KEY:-}" ]] || { echo "W&B credentials required" >&2; exit 1; }
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.75
cd "$ROOT"

env -u LD_LIBRARY_PATH "$PYTHON" -u \
    baselines/IPPO/eval_crossplay_overcooked_v3.py \
    "$ENTITY/$PREFIX-${KIND}_train" \
    "${FCP_SOURCES[@]}" \
    --algorithms IPPO FCP --layout distance_10 \
    --layout-revision "$REVISION" --seeds "${SEEDS[@]}" \
    --transition-observer both --episodes 20 --max-steps 450 \
    --vmap-indices 0 --gpus "${GPU_LIST[@]}" --workers-per-gpu 4 \
    --wandb-mode online --output-project "$ENTITY/$PREFIX-${KIND}-fcp_eval" \
    --output-dir "$OUTPUT_DIR" --save-adaptation-traces \
    --adaptation-horizon 75 --adaptation-window 30 \
    --drop-baseline-window 60 --drop-horizon 60 \
    > "$OUTPUT_DIR/eval.log" 2>&1
touch "$OUTPUT_DIR/eval.done"
