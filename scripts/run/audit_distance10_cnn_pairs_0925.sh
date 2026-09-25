#!/usr/bin/env bash
# Replay one CNN seed's SP and its lowest-scoring ordered XP partner.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-$HOME/overcooked-replan-venv/bin/python}"
GPU="${GPU:?set a currently free GPU ID}"
case " $GPU_ALLOWLIST " in
    *" $GPU "*) ;;
    *) echo "GPU $GPU is outside GPU_ALLOWLIST=$GPU_ALLOWLIST" >&2; exit 1 ;;
esac

OUTPUT="$ROOT/campaigns/distance10-priority-0925-cnn-behavior"
mkdir -p "$OUTPUT"
if [[ -z "${WANDB_API_KEY:-}" && -f "$HOME/.netrc" ]]; then
    WANDB_API_KEY="$("$PYTHON" -c 'import netrc; a=netrc.netrc().authenticators("api.wandb.ai"); print(a[2] if a else "")')"
    export WANDB_API_KEY
fi
[[ -n "${WANDB_API_KEY:-}" ]] || { echo "W&B credentials required" >&2; exit 1; }
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.75
cd "$ROOT"

SOURCE=overcooked-v3-mapsearch-0925-d10-priority-full10-cnn_train
TARGET=overcooked-v3-mapsearch-0925-d10-priority-full10-behavior_eval
SEED7=7ureyn4o
SEED1=iwccpeix
for pair in sp xp; do
    if [[ "$pair" == sp ]]; then partner="$SEED7"; else partner="$SEED1"; fi
    [[ -f "$OUTPUT/cnn_s7_${pair}.done" ]] && continue
    env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" -u \
        baselines/IPPO/eval_wandb_crossplay_overcooked_v3.py \
        --run-ids "$SEED7" "$partner" --source-project "$SOURCE" \
        --project "$TARGET" --layout distance_10 --episodes 1 \
        --max-steps 450 --seed 0 --wandb-mode online \
        --video "$OUTPUT/cnn_s7_${pair}.mp4" \
        --metrics-json "$OUTPUT/cnn_s7_${pair}.json" \
        > "$OUTPUT/cnn_s7_${pair}.log" 2>&1
    touch "$OUTPUT/cnn_s7_${pair}.done"
done
