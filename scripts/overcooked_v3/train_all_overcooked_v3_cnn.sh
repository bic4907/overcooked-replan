#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_DIR}"

TRAIN_SEEDS="${TRAIN_SEEDS:-0 1}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-3e7}"
LOG_INTERVAL="${LOG_INTERVAL:-10}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-50}"
GPU_ID="${GPU_ID:-0}"
hydra_overrides=("$@")

read -r -a train_seeds <<< "${TRAIN_SEEDS}"
if [[ ${#train_seeds[@]} -eq 0 ]]; then
    echo "TRAIN_SEEDS must contain at least one seed" >&2
    exit 2
fi

layouts=(
    split_0
    split_1
    outage_0
    outage_1
    recipe_switch_0
    recipe_switch_1
    distance_switch_0
    distance_switch_1
)

for layout in "${layouts[@]}"; do
    for seed in "${train_seeds[@]}"; do
        echo "[GPU ${GPU_ID}][seed ${seed}] ===== Training CNN: ${layout} ====="

        if ! CUDA_VISIBLE_DEVICES="${GPU_ID}" \
            XLA_PYTHON_CLIENT_PREALLOCATE=false \
            python -u baselines/IPPO/ippo_overcooked_v3.py \
                scenario="${layout}" \
                ARCHITECTURE=cnn \
                SEED="${seed}" \
                NUM_SEEDS=1 \
                TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS}" \
                LOG_INTERVAL="${LOG_INTERVAL}" \
                CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL}" \
                "${hydra_overrides[@]}" \
                2>&1 | sed -u "s/^/[GPU ${GPU_ID}][seed ${seed}] /"
        then
            echo "[GPU ${GPU_ID}][seed ${seed}] Training failed: ${layout}" >&2
            exit 1
        fi
    done
done

echo "===== All role-scenario CNN training runs completed ====="
