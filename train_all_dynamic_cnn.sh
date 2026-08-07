#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_DIR}"

TRAIN_SEEDS="${TRAIN_SEEDS:-0 1}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-3e7}"
LOG_INTERVAL="${LOG_INTERVAL:-10}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-50}"
SAVE_PATH="${SAVE_PATH:-models}"
WANDB_MODE="${WANDB_MODE:-disabled}"
GPU_ID="${GPU_ID:-0}"

read -r -a train_seeds <<< "${TRAIN_SEEDS}"
if [[ ${#train_seeds[@]} -eq 0 ]]; then
    echo "TRAIN_SEEDS must contain at least one seed" >&2
    exit 2
fi

layouts=(
    dynamic_easy_0
    dynamic_easy_1
    dynamic_easy_2
    dynamic_easy_3
    dynamic_easy_4
    dynamic_medium_0
    dynamic_medium_1
    dynamic_medium_2
    dynamic_medium_3
    dynamic_medium_4
    dynamic_hard_0
    dynamic_hard_1
    dynamic_hard_2
    dynamic_hard_3
    dynamic_hard_4
)

for layout in "${layouts[@]}"; do
    for seed in "${train_seeds[@]}"; do
        echo "[GPU ${GPU_ID}][seed ${seed}] ===== Training CNN: ${layout} ====="

        if ! CUDA_VISIBLE_DEVICES="${GPU_ID}" \
            XLA_PYTHON_CLIENT_PREALLOCATE=false \
            python -u baselines/IPPO/ippo_overcooked.py \
                ARCHITECTURE=cnn \
                ENV_NAME=overcooked_dynamic \
                ENV_KWARGS.layout="${layout}" \
                SEED="${seed}" \
                NUM_SEEDS=1 \
                TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS}" \
                LOG_INTERVAL="${LOG_INTERVAL}" \
                CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL}" \
                WANDB_MODE="${WANDB_MODE}" \
                SAVE_PATH="${SAVE_PATH}" \
                hydra.run.dir="outputs/ippo_v1/cnn/${layout}/seed${seed}" \
                2>&1 | sed -u "s/^/[GPU ${GPU_ID}][seed ${seed}] /"
        then
            echo "[GPU ${GPU_ID}][seed ${seed}] Training failed: ${layout}" >&2
            exit 1
        fi
    done
done

echo "===== All dynamic CNN training runs completed ====="
