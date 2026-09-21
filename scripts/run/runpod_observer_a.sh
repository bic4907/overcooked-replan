#!/usr/bin/env bash
# Runs IPPO-RNN observer-A training + eval, then powers off the pod on success.
# Designed for the Runpod host where /opt/overcooked-venv already has python+wandb.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
export GPUS="${GPUS:-0 1 2 3}"
export VENV="${VENV:-/opt/overcooked-venv}"

mkdir -p "$REPO_ROOT/logs"
DRY_LOG="$REPO_ROOT/logs/observer-a-dryrun.log"

if [ -z "${WANDB_API_KEY:-}" ]; then
    echo "WANDB_API_KEY is not set; refusing to run." >&2
    exit 1
fi

# 1. Offline dry-run: tiny 1-update job on GPU 0 to prove the pipeline works.
echo "[dry-run] launching offline sanity check..."
env \
    -u LD_LIBRARY_PATH \
    PYTHONPATH="$REPO_ROOT" \
    CUDA_VISIBLE_DEVICES="$(echo "$GPUS" | awk '{print $1}')" \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.6 \
    WANDB_MODE=offline \
    "$VENV/bin/python" -u "$REPO_ROOT/baselines/IPPO/ippo_overcooked_v3.py" \
        scenario=split_0 \
        SEED=0 \
        ARCHITECTURE=rnn \
        TRANSITION_OBSERVER=agent_0 \
        ENV_KWARGS.transition_observer=agent_0 \
        ENTITY=cilab-overcooked \
        PROJECT=overcooked-v3-ippo-rnn-observer-a-0921_dryrun \
        NUM_ENVS=8 \
        NUM_STEPS=8 \
        NUM_MINIBATCHES=1 \
        UPDATE_EPOCHS=1 \
        TOTAL_TIMESTEPS=128 \
        REW_SHAPING_HORIZON=128 \
        LOG_INTERVAL=1 \
        upload_final_checkpoint=false \
        recording=disabled \
        >"$DRY_LOG" 2>&1
echo "[dry-run] passed. tail:"; tail -10 "$DRY_LOG"

# 2. Full experiment: 6 layouts x 6 seeds trained on 4 GPUs, then eval matrix.
bash "$HERE/run_baseline.sh" ippo-rnn-observer-a
bash "$HERE/run_baseline.sh" ippo-rnn-observer-a-eval

# 3. Shutdown pod on success.
echo "[done] shutting down pod in 30s..."
sleep 30
if command -v runpodctl >/dev/null 2>&1 && [ -n "${RUNPOD_POD_ID:-}" ]; then
    runpodctl stop pod "$RUNPOD_POD_ID" || true
fi
shutdown -h now 2>/dev/null || poweroff -f 2>/dev/null || true
