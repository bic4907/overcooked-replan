#!/usr/bin/env bash
# Run the five observer sweeps across multiple hosts with a global W&B barrier.
#
# Start this script on every participating host. W&B allocates distinct grid
# jobs to each agent. A host may leave a stage before agents on another host,
# so the global barrier must pass before any host advances to the next stage.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
START_AT="${START_AT:-ippo-train}"
FCP_SHARED_ROOT="${FCP_SHARED_ROOT:-${REPO_ROOT}/saves/fcp_observer}"
FCP_POPULATION_ROOT="${FCP_POPULATION_ROOT:-${FCP_SHARED_ROOT}/population}"
SHARE_MARKER="${FCP_SHARED_ROOT}/.distributed-observer-share"
POLL_SECONDS="${SWEEP_BARRIER_POLL_SECONDS:-30}"
TIMEOUT_HOURS="${SWEEP_BARRIER_TIMEOUT_HOURS:-96}"
IPPO_EVAL_SWEEP_REF="${IPPO_EVAL_SWEEP_REF:-cilab-overcooked/overcooked-v3-ippo-rnn-observer_eval/5z9gtfh2}"
TRAIN_JAX_PREALLOCATE="${TRAIN_JAX_PREALLOCATE:-true}"
EVAL_JAX_PREALLOCATE="${EVAL_JAX_PREALLOCATE:-false}"

STAGE_KEYS=(
    ippo-train
    ippo-eval
    fcp-population
    fcp-train
    fcp-eval
)
SWEEP_REFS=(
    cilab-overcooked/overcooked-v3-ippo-rnn-observer_train/ckt0sirk
    "$IPPO_EVAL_SWEEP_REF"
    cilab-overcooked/overcooked-v3-fcp-observer_population/8t665wf5
    cilab-overcooked/overcooked-v3-fcp-observer_train/bbwihz80
    cilab-overcooked/overcooked-v3-fcp-observer_eval/ausz3nxr
)
EXPECTED_RUNS=(144 24 72 144 24)
LAYOUTS=(
    split_0
    split_1
    outage_0
    outage_1
    distance_switch_0
    distance_switch_1
)
OBSERVERS=(none agent_0 agent_1 both)

log() {
    echo "[$(date '+%Y-%m-%dT%H:%M:%S%z')] $*"
}

require_shared_root() {
    if ! timeout 15 test -r "$SHARE_MARKER"; then
        echo "FCP shared root is unavailable: $SHARE_MARKER" >&2
        exit 1
    fi
}

verify_fcp_population() {
    local observer
    require_shared_root
    for observer in "${OBSERVERS[@]}"; do
        "$PYTHON_BIN" "$REPO_ROOT/scripts/verify_easy1_fcp_population.py" \
            "$FCP_POPULATION_ROOT/$observer" \
            --layouts "${LAYOUTS[@]}" \
            --seeds 0 1 2
    done
}

start_index=-1
for index in "${!STAGE_KEYS[@]}"; do
    if [ "${STAGE_KEYS[$index]}" = "$START_AT" ]; then
        start_index=$index
        break
    fi
done
if [ "$start_index" -lt 0 ]; then
    echo "Unknown START_AT stage: $START_AT" >&2
    echo "Expected one of: ${STAGE_KEYS[*]}" >&2
    exit 1
fi

for ((index = start_index; index < ${#STAGE_KEYS[@]}; index++)); do
    stage="${STAGE_KEYS[$index]}"
    sweep_ref="${SWEEP_REFS[$index]}"
    expected="${EXPECTED_RUNS[$index]}"
    stage_preallocate="$TRAIN_JAX_PREALLOCATE"

    if [[ "$stage" == *-eval ]]; then
        stage_preallocate="$EVAL_JAX_PREALLOCATE"
    fi

    if [[ "$stage" == fcp-* ]]; then
        require_shared_root
    fi

    log "Starting stage $stage on GPUs: ${GPUS:-0}"
    GPUS="${GPUS:-0}" \
        XLA_PYTHON_CLIENT_PREALLOCATE="$stage_preallocate" \
        bash "$REPO_ROOT/experiment/run_agents_sequential.sh" "$sweep_ref"

    log "Local agents exited for $stage; waiting on the global barrier"
    "$PYTHON_BIN" "$REPO_ROOT/scripts/wait_wandb_sweep.py" \
        "$sweep_ref" \
        --expected-runs "$expected" \
        --poll-seconds "$POLL_SECONDS" \
        --timeout-hours "$TIMEOUT_HOURS"

    if [ "$stage" = "fcp-population" ]; then
        verify_fcp_population
    fi
    log "Completed stage $stage"
done

log "All distributed observer stages completed"
