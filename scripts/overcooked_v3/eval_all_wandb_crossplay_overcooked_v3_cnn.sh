#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_DIR}"

WANDB_ENTITY="${WANDB_ENTITY:-cilab-overcooked}"
SOURCE_PROJECT="${SOURCE_PROJECT:-overcooked-v3-role-coordination}"
EVAL_PROJECT="${EVAL_PROJECT:-overcooked-v3-crossplay}"
LAYOUT="${LAYOUT:-}"
EPISODES="${EPISODES:-10}"
MAX_STEPS="${MAX_STEPS:-400}"
EVAL_SEED="${EVAL_SEED:-0}"
JAX_PLATFORM="${JAX_PLATFORM:-cpu}"
PAIR_MODE="${PAIR_MODE:-all}"
MATRIX_ROOT="${MATRIX_ROOT:-evaluation/overcooked_v3/crossplay/matrices}"

if [[ $# -gt 0 ]]; then
    run_ids=("$@")
elif [[ -n "${RUN_IDS:-}" ]]; then
    read -r -a run_ids <<< "${RUN_IDS}"
else
    echo "Usage: $0 RUN_ID [RUN_ID ...]" >&2
    echo "Or set RUN_IDS='RUN_ID_A RUN_ID_B ...'." >&2
    exit 2
fi

if [[ ${#run_ids[@]} -lt 2 ]]; then
    echo "At least two W&B run IDs are required." >&2
    exit 2
fi
if [[ "${PAIR_MODE}" != "all" && "${PAIR_MODE}" != "cross-only" ]]; then
    echo "PAIR_MODE must be 'all' or 'cross-only'." >&2
    exit 2
fi

common_args=(
    --entity "${WANDB_ENTITY}"
    --source-project "${SOURCE_PROJECT}"
    --project "${EVAL_PROJECT}"
    --episodes "${EPISODES}"
    --max-steps "${MAX_STEPS}"
    --seed "${EVAL_SEED}"
)
if [[ -n "${LAYOUT}" ]]; then
    common_args+=(--layout "${LAYOUT}")
fi

matrix_session="${MATRIX_ROOT}/$(date +%Y%m%d-%H%M%S)-$$"
matrix_metrics_dir="${matrix_session}/metrics"
matrix_output="${matrix_session}/crossplay_matrix.txt"
mkdir -p "${matrix_metrics_dir}"

for agent_0_index in "${!run_ids[@]}"; do
    for agent_1_index in "${!run_ids[@]}"; do
        if [[ "${PAIR_MODE}" == "cross-only" && "${agent_0_index}" == "${agent_1_index}" ]]; then
            continue
        fi
        agent_0_run="${run_ids[agent_0_index]}"
        agent_1_run="${run_ids[agent_1_index]}"
        echo "===== W&B cross-play: agent_0=${agent_0_run}, agent_1=${agent_1_run} ====="

        JAX_PLATFORMS="${JAX_PLATFORM}" \
            XLA_PYTHON_CLIENT_PREALLOCATE=false \
            MPLCONFIGDIR=/tmp \
            python -u baselines/IPPO/eval_wandb_crossplay_overcooked_v3.py \
                --run-ids "${agent_0_run}" "${agent_1_run}" \
                --metrics-json "${matrix_metrics_dir}/pair_${agent_0_index}_${agent_1_index}.json" \
                "${common_args[@]}"

        python -u baselines/IPPO/print_crossplay_matrix.py \
            --metrics-dir "${matrix_metrics_dir}" \
            --run-ids "${run_ids[@]}" \
            --pair-mode "${PAIR_MODE}" \
            --output "${matrix_output}"
    done
done

echo "===== Cross-play matrix saved to ${matrix_output} ====="
echo "===== All W&B cross-play evaluations completed in ${EVAL_PROJECT} ====="
