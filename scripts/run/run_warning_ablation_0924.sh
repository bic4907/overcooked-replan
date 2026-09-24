#!/usr/bin/env bash
# Evaluate trained distance_2/3 checkpoints with phase warnings hidden.
# Run only after the corresponding main training is complete.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
PYTHON="${PYTHON:-$HOME/overcooked-replan-venv/bin/python}"
read -r -a GPU_LIST <<< "${GPUS:-0 1 2 3}"
read -r -a KIND_LIST <<< "${KINDS:-fcp rnn}"
CAMPAIGN_MAIN="${CAMPAIGN_MAIN:-handoff-site-0924-v4-d23-seed10}"
CAMPAIGN_CNN="${CAMPAIGN_CNN:-handoff-site-0924-v4-d23-cnn10}"
RUN_ROOT="${DIAGNOSTIC_ROOT:-$ROOT/campaigns/handoff-site-0924-v4-d23-warning-ablation}"
mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/evaluation"

jobs=()
names=()
slot=0
for kind in "${KIND_LIST[@]}"; do
    case "$kind" in
        fcp) algorithm=FCP; campaign="$CAMPAIGN_MAIN" ;;
        rnn) algorithm=IPPO; campaign="$CAMPAIGN_MAIN" ;;
        cnn) algorithm=IPPO; campaign="$CAMPAIGN_CNN" ;;
        *) echo "Unsupported algorithm: $kind" >&2; exit 2 ;;
    esac
    for layout in distance_2 distance_3; do
        output_dir="$RUN_ROOT/evaluation/$kind/$layout"
        if [[ -f "$output_dir/summary.json" ]]; then
            echo "[$(date -Is)] already complete: $kind $layout"
            continue
        fi
        if ((slot >= ${#GPU_LIST[@]})); then
            echo "Supply at least one GPU per pending map/algorithm job" >&2
            exit 2
        fi
        gpu="${GPU_LIST[$slot]}"
        slot=$((slot + 1))
        project_prefix="overcooked-v3-${campaign}-main-agent_0-${kind}"
        name="${kind}_${layout}"
        echo "[$(date -Is)] start $name GPU=$gpu"
        env -u LD_LIBRARY_PATH "$PYTHON" -u \
            baselines/IPPO/eval_crossplay_overcooked_v3.py \
            "cilab-overcooked/${project_prefix}_train" \
            --algorithms "$algorithm" --layout "$layout" \
            --seeds 0 1 2 3 4 5 6 7 8 9 \
            --transition-observer agent_0 --eval-transition-observer none \
            --episodes 20 --max-steps 450 --gpus "$gpu" \
            --workers-per-gpu 4 --wandb-mode online \
            --output-project "cilab-overcooked/${project_prefix}_warning_ablation_0924" \
            --output-dir "$output_dir" --run-label "${kind}-no-warning" \
            --save-adaptation-traces --adaptation-horizon 75 \
            --adaptation-window 30 --drop-baseline-window 60 \
            --drop-horizon 60 > "$RUN_ROOT/logs/$name.log" 2>&1 &
        jobs+=("$!")
        names+=("$name")
    done
done

failed=0
for index in "${!jobs[@]}"; do
    if wait "${jobs[$index]}"; then
        echo "[$(date -Is)] done ${names[$index]}"
    else
        echo "[$(date -Is)] FAILED ${names[$index]}" >&2
        failed=1
    fi
done
exit "$failed"
