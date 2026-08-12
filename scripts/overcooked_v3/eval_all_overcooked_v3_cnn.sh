#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_DIR}"

SAVES_DIR="${SAVES_DIR:-${PROJECT_DIR}/saves}"
EVALUATION_DIR="${EVALUATION_DIR:-${PROJECT_DIR}/evaluation/overcooked_v3/cnn}"
EPISODES="${EPISODES:-3}"
MAX_STEPS="${MAX_STEPS:-400}"
EVAL_SEED="${EVAL_SEED:-0}"
JAX_PLATFORM="${JAX_PLATFORM:-cpu}"

if [[ ! -d "${SAVES_DIR}" ]]; then
    echo "Experiment directory does not exist: ${SAVES_DIR}" >&2
    echo "Train a model first or set SAVES_DIR explicitly." >&2
    exit 2
fi

layouts=(
    dynamic_00
    dynamic_01
    dynamic_02
    dynamic_03
    dynamic_04
    dynamic_05
    dynamic_06
    dynamic_07
    dynamic_08
    dynamic_09
    dynamic_10
    dynamic_11
    dynamic_12
    dynamic_13
    dynamic_14
)

# label:agent_0_training_seed:agent_1_training_seed
pair_specs=(
    same_seed0:0:0
    same_seed1:1:1
    cross_seed0_seed1:0:1
    cross_seed1_seed0:1:0
)

# Check every final checkpoint before starting, so an unfinished training run
# does not leave behind a misleading partial set of evaluation results.
missing_checkpoint=0
for layout in "${layouts[@]}"; do
    experiment_name="overcooked_v3_${layout#dynamic_}"
    checkpoint_dir="${SAVES_DIR}"

    for seed in 0 1; do
        checkpoint_name="ippo_cnn_${experiment_name}_seed${seed}_vmap0.safetensors"
        if ! find "${checkpoint_dir}" -type f -name "${checkpoint_name}" -print -quit 2>/dev/null | grep -q .; then
            echo "Missing final checkpoint under ${checkpoint_dir}: ${checkpoint_name}" >&2
            missing_checkpoint=1
        fi
    done
done

if [[ ${missing_checkpoint} -ne 0 ]]; then
    echo "Evaluation stopped because one or more final checkpoints are missing." >&2
    exit 1
fi

for layout in "${layouts[@]}"; do
    layout_dir="${EVALUATION_DIR}/${layout}"
    mkdir -p "${layout_dir}"

    for pair_spec in "${pair_specs[@]}"; do
        IFS=: read -r pair_label agent_0_seed agent_1_seed <<< "${pair_spec}"
        output_prefix="${layout_dir}/${layout}_${pair_label}"

        echo "===== Evaluating ${layout}: agent_0=seed${agent_0_seed}, agent_1=seed${agent_1_seed} ====="

        JAX_PLATFORMS="${JAX_PLATFORM}" \
            XLA_PYTHON_CLIENT_PREALLOCATE=false \
            MPLCONFIGDIR=/tmp \
            python -u baselines/IPPO/eval_ippo_overcooked_v3.py \
                --architecture cnn \
                --saves-dir "${SAVES_DIR}" \
                --layout "${layout}" \
                --agent-seeds "${agent_0_seed}" "${agent_1_seed}" \
                --episodes "${EPISODES}" \
                --max-steps "${MAX_STEPS}" \
                --seed "${EVAL_SEED}" \
                --gif "${output_prefix}.gif" \
                2>&1 | tee "${output_prefix}.log"
    done
done

echo "===== All dynamic CNN evaluations completed: ${EVALUATION_DIR} ====="
