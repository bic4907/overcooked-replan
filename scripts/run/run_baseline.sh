#!/usr/bin/env bash
# Launch IPPO/FCP training or evaluation for the 6 selected layouts x 6 seeds
# without relying on W&B sweeps.
#
# Usage:
#   GPUS="0 1 2 3" VENV=~/overcooked-replan-venv \
#     bash scripts/run/run_baseline.sh \
#       {ippo-cnn|ippo-rnn|fcp-population|fcp-train|
#        ippo-cnn-eval|ippo-rnn-eval|fcp-eval}
#
# Training modes dispatch one (scenario, seed) python process per free GPU.
# Evaluation modes dispatch one (layout) worker per free GPU using the
# eval_crossplay entrypoint against the corresponding training project.
set -euo pipefail

EXP="${1:?experiment name required}"
GPUS="${GPUS:-0}"
VENV="${VENV:-$HOME/overcooked-replan-venv}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

read -r -a GPU_LIST <<< "$GPUS"
LAYOUTS=(split_0 split_1 outage_0 outage_1 distance_0 distance_1)
SEEDS=(0 1 2 3 4 5)

MODE="train"
EVAL_EXTRA=()
case "$EXP" in
    ippo-cnn)
        PROGRAM="baselines/IPPO/ippo_overcooked_v3.py"
        PROJECT="overcooked-v3-ippo-0921_train"
        EXTRA=(ARCHITECTURE=cnn)
        ;;
    ippo-rnn)
        PROGRAM="baselines/IPPO/ippo_overcooked_v3.py"
        PROJECT="overcooked-v3-ippo-rnn-0921_train"
        EXTRA=(ARCHITECTURE=rnn)
        ;;
    ippo-rnn-observer-a)
        PROGRAM="baselines/IPPO/ippo_overcooked_v3.py"
        PROJECT="overcooked-v3-ippo-rnn-observer-a-0921_train"
        EXTRA=(
            ARCHITECTURE=rnn
            +TRANSITION_OBSERVER=agent_0
            ++ENV_KWARGS.transition_observer=agent_0
        )
        ;;
    fcp-population)
        PROGRAM="baselines/IPPO/ippo_overcooked_v3.py"
        PROJECT="overcooked-v3-fcp-0921-population"
        EXTRA=(
            ARCHITECTURE=rnn
            SAVES_DIR=saves/fcp_population
            CHECKPOINT_INTERVAL=0
            "CHECKPOINT_FRACTIONS=[0.1,0.5,1.0]"
            upload_final_checkpoint=false
            recording=disabled
        )
        ;;
    fcp-train)
        PROGRAM="baselines/FCP/fcp_overcooked_v3.py"
        PROJECT="overcooked-v3-fcp-0921_train"
        EXTRA=()
        ;;
    ippo-cnn-eval)
        MODE="eval"
        SOURCE_PROJECT="cilab-overcooked/overcooked-v3-ippo-0921_train"
        OUTPUT_PROJECT="cilab-overcooked/overcooked-v3-ippo-0921_eval"
        ALGORITHM="IPPO"
        ;;
    ippo-rnn-eval)
        MODE="eval"
        SOURCE_PROJECT="cilab-overcooked/overcooked-v3-ippo-rnn-0921_train"
        OUTPUT_PROJECT="cilab-overcooked/overcooked-v3-ippo-rnn-0921_eval"
        ALGORITHM="IPPO"
        ;;
    ippo-rnn-observer-a-eval)
        MODE="eval"
        SOURCE_PROJECT="cilab-overcooked/overcooked-v3-ippo-rnn-observer-a-0921_train"
        OUTPUT_PROJECT="cilab-overcooked/overcooked-v3-ippo-rnn-observer-a-0921_eval"
        ALGORITHM="IPPO"
        EVAL_EXTRA=(--transition-observer agent_0)
        ;;
    fcp-eval)
        MODE="eval"
        SOURCE_PROJECT="cilab-overcooked/overcooked-v3-fcp-0921_train"
        OUTPUT_PROJECT="cilab-overcooked/overcooked-v3-fcp-0921_eval"
        ALGORITHM="FCP"
        ;;
    *)
        echo "unknown experiment: $EXP" >&2
        exit 1
        ;;
esac

if [ ! -f "$VENV/bin/activate" ]; then
    echo "venv activate script not found: $VENV/bin/activate" >&2
    exit 1
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
cd "$REPO_ROOT"

LOG_DIR="$REPO_ROOT/logs/$EXP"
mkdir -p "$LOG_DIR"

declare -A GPU_PID
for g in "${GPU_LIST[@]}"; do GPU_PID[$g]=""; done

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

pick_free_gpu() {
    while true; do
        for g in "${GPU_LIST[@]}"; do
            pid="${GPU_PID[$g]}"
            if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then
                echo "$g"
                return
            fi
        done
        sleep 5
    done
}

cleanup() {
    log "Stopping running jobs..."
    for g in "${GPU_LIST[@]}"; do
        pid="${GPU_PID[$g]}"
        [ -n "$pid" ] && kill "$pid" 2>/dev/null || true
    done
    wait || true
}
trap cleanup INT TERM

log "Experiment: $EXP ($MODE)"
log "GPUs: ${GPU_LIST[*]}"
log "Repo: $REPO_ROOT"
log "Venv: $VENV"

launch_train_jobs() {
    for scenario in "${LAYOUTS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            gpu="$(pick_free_gpu)"
            log_file="$LOG_DIR/${scenario}_seed${seed}_gpu${gpu}.log"
            log "launch $EXP scenario=$scenario SEED=$seed GPU=$gpu -> $(basename "$log_file")"
            env \
                -u LD_LIBRARY_PATH \
                PYTHONPATH="$REPO_ROOT" \
                CUDA_VISIBLE_DEVICES="$gpu" \
                XLA_PYTHON_CLIENT_PREALLOCATE=false \
                XLA_PYTHON_CLIENT_MEM_FRACTION=0.75 \
                PYTHONFAULTHANDLER=1 \
                python -u "$PROGRAM" \
                    scenario="$scenario" \
                    SEED="$seed" \
                    ENTITY=cilab-overcooked \
                    PROJECT="$PROJECT" \
                    "${EXTRA[@]}" \
                    >"$log_file" 2>&1 &
            GPU_PID[$gpu]="$!"
        done
    done
}

launch_eval_jobs() {
    for layout in "${LAYOUTS[@]}"; do
        gpu="$(pick_free_gpu)"
        log_file="$LOG_DIR/${layout}_gpu${gpu}.log"
        log "launch $EXP layout=$layout GPU=$gpu -> $(basename "$log_file")"
        env \
            -u LD_LIBRARY_PATH \
            PYTHONPATH="$REPO_ROOT" \
            CUDA_VISIBLE_DEVICES="$gpu" \
            XLA_PYTHON_CLIENT_PREALLOCATE=false \
            XLA_PYTHON_CLIENT_MEM_FRACTION=0.75 \
            PYTHONFAULTHANDLER=1 \
            python -u baselines/IPPO/eval_crossplay_overcooked_v3.py \
                "$SOURCE_PROJECT" \
                --algorithms "$ALGORITHM" \
                --layout "$layout" \
                --episodes 20 \
                --max-steps 450 \
                --workers-per-gpu 8 \
                --entity cilab-overcooked \
                --output-project "$OUTPUT_PROJECT" \
                --gpus "$gpu" \
                "${EVAL_EXTRA[@]}" \
                >"$log_file" 2>&1 &
        GPU_PID[$gpu]="$!"
    done
}

if [ "$MODE" = "train" ]; then
    launch_train_jobs
else
    launch_eval_jobs
fi

log "All jobs dispatched; waiting for stragglers..."
for g in "${GPU_LIST[@]}"; do
    pid="${GPU_PID[$g]}"
    [ -n "$pid" ] && wait "$pid" 2>/dev/null || true
done
trap - INT TERM
log "$EXP complete"
