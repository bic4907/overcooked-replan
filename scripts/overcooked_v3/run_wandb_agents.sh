#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
GPUS="${GPUS:-0}"
DRY_RUN="${DRY_RUN:-0}"

log() {
    printf '[%s] %s\n' "$(date '+%H:%M:%S')" "$*"
}

usage() {
    cat <<'EOF'
Usage:
  GPUS="0 1 2 3" bash scripts/overcooked_v3/run_wandb_agents.sh [SWEEP_PATH ...]

Each sweep path must have the form ENTITY/PROJECT/SWEEP_ID. One W&B agent is
started per GPU. Multiple sweep paths are processed in order.

Environment variables:
  GPUS           Space- or comma-separated GPU IDs (default: 0)
  PYTHON_BIN     Python executable from the installed environment (default: python)
  DRY_RUN=1      Print the assignments without launching W&B agents
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

gpu_spec="${GPUS//,/ }"
read -r -a gpu_list <<< "${gpu_spec}"
if [[ ${#gpu_list[@]} -eq 0 ]]; then
    echo "GPUS must contain at least one GPU ID" >&2
    exit 2
fi

sweep_paths=("$@")
if [[ ${#sweep_paths[@]} -eq 0 ]]; then
    echo "A sweep path is required: ENTITY/PROJECT/SWEEP_ID" >&2
    usage >&2
    exit 2
fi

for sweep_path in "${sweep_paths[@]}"; do
    if [[ ! "${sweep_path}" =~ ^[^/]+/[^/]+/[^/]+$ ]]; then
        echo "Invalid sweep path '${sweep_path}'; expected ENTITY/PROJECT/SWEEP_ID." >&2
        exit 2
    fi
done

if [[ "${DRY_RUN}" != "1" ]]; then
    if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
        echo "Python executable not found: ${PYTHON_BIN}" >&2
        exit 127
    fi
    if ! "${PYTHON_BIN}" -c 'import dotenv, wandb' >/dev/null 2>&1; then
        echo "python-dotenv and wandb are required. Activate the project environment and install .[algs]." >&2
        exit 1
    fi
fi

active_pids=""
stop_agents() {
    if [[ -n "${active_pids}" ]]; then
        kill ${active_pids} 2>/dev/null || true
        wait ${active_pids} 2>/dev/null || true
    fi
}
trap stop_agents INT TERM

run_agents() {
    local sweep_path="$1"
    local status=0
    active_pids=""

    log "Starting ${#gpu_list[@]} agent(s) for ${sweep_path} on GPU(s): ${gpu_list[*]}"
    for gpu_id in "${gpu_list[@]}"; do
        if [[ "${DRY_RUN}" == "1" ]]; then
            log "[GPU ${gpu_id}] would run W&B agent ${sweep_path}"
            continue
        fi

        (
            export CUDA_VISIBLE_DEVICES="${gpu_id}"
            export XLA_PYTHON_CLIENT_PREALLOCATE=false
            "${PYTHON_BIN}" -m dotenv run --no-override -- \
                "${PYTHON_BIN}" -m wandb agent "${sweep_path}"
        ) 2>&1 | sed -u "s/^/[GPU ${gpu_id}] /" &
        active_pids="${active_pids} $!"
    done

    for pid in ${active_pids}; do
        if ! wait "${pid}"; then
            status=1
        fi
    done
    active_pids=""

    if [[ ${status} -ne 0 ]]; then
        echo "At least one W&B agent failed for ${sweep_path}." >&2
        return "${status}"
    fi
    log "Completed sweep ${sweep_path}"
}

for sweep_path in "${sweep_paths[@]}"; do
    run_agents "${sweep_path}"
done
