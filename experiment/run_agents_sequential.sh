#!/usr/bin/env bash
# Run several W&B grid sweeps sequentially, with one agent per GPU.
#
# Usage:
#   GPUS="0 1 2 3" bash experiment/run_agents_sequential.sh \
#     entity/project/TRAIN_SWEEP_ID \
#     entity/project/EVAL_SWEEP_ID
#
# Every GPU works on the current sweep in parallel. The next sweep starts only
# after every agent for the current grid sweep exits. No --count is used.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPUS="${GPUS:-0}"
read -r -a GPU_LIST <<< "$GPUS"

ACTIVE_PIDS=()

log() {
    echo "[$(date '+%H:%M:%S')] $*"
}

cleanup() {
    local pid
    if [ "${#ACTIVE_PIDS[@]}" -gt 0 ]; then
        log "Stopping active W&B agents..."
        for pid in "${ACTIVE_PIDS[@]}"; do
            kill "$pid" 2>/dev/null || true
        done
        for pid in "${ACTIVE_PIDS[@]}"; do
            wait "$pid" 2>/dev/null || true
        done
    fi
}

handle_signal() {
    cleanup
    exit 130
}

trap handle_signal INT TERM

run_sweep_agents() {
    local sweep_ref="$1"
    local gpu_id
    local pid
    local failed=0

    ACTIVE_PIDS=()
    log "Starting $sweep_ref on GPUs: ${GPU_LIST[*]}"
    for gpu_id in "${GPU_LIST[@]}"; do
        env \
            -u LD_LIBRARY_PATH \
            PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
            CUDA_VISIBLE_DEVICES="$gpu_id" \
            XLA_PYTHON_CLIENT_PREALLOCATE=false \
            PYTHONFAULTHANDLER=1 \
            WANDB_AGENT_DISABLE_FLAPPING=true \
            wandb agent "$sweep_ref" &
        ACTIVE_PIDS+=("$!")
        log "Started GPU $gpu_id agent (pid=${ACTIVE_PIDS[-1]})"
    done

    for pid in "${ACTIVE_PIDS[@]}"; do
        if ! wait "$pid"; then
            failed=1
        fi
    done
    ACTIVE_PIDS=()

    if [ "$failed" -ne 0 ]; then
        log "One or more agents failed for $sweep_ref"
        return 1
    fi
    log "Completed $sweep_ref"
}

if [ "$#" -eq 0 ]; then
    echo "Usage: GPUS=\"0 1 2 3\" bash $0 SWEEP_REF_1 [SWEEP_REF_2 ...]"
    exit 1
fi

if [ "${#GPU_LIST[@]}" -eq 0 ]; then
    echo "GPUS must contain at least one GPU ID"
    exit 1
fi

command -v wandb >/dev/null 2>&1 || {
    echo "wandb is not available in the active environment"
    exit 1
}

cd "$REPO_ROOT"
for sweep_ref in "$@"; do
    if [[ ! "$sweep_ref" =~ ^[^/]+/[^/]+/[^/]+$ ]]; then
        echo "Invalid sweep reference: $sweep_ref"
        echo "Expected ENTITY/PROJECT/SWEEP_ID"
        exit 1
    fi
    run_sweep_agents "$sweep_ref"
done

trap - INT TERM
log "All sweeps completed"
