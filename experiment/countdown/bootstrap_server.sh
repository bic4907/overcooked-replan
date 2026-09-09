#!/usr/bin/env bash
# Bring a fresh GPU box up to the countdown experiment's branch and verify it
# can actually train before any W&B agent touches a sweep.
#
#   bash bootstrap_server.sh                    # clone into /workspace, branch dev/needs
#   REPO_DIR=~/work BRANCH=main bash bootstrap_server.sh
#
# A grid sweep never re-issues a parameter combination it has handed out, so an
# agent that dies at CUDA init burns runs that stay dead until the crashed runs
# are deleted. That is why this script ends with a real training run rather than
# an import check: on a Volta box the import succeeds and training still fails.
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/bic4907/overcooked-replan.git}"
REPO_DIR="${REPO_DIR:-/workspace}"
BRANCH="${BRANCH:-dev/needs}"

echo "== 1/5  repository =="
if [ -d "$REPO_DIR/.git" ]; then
    git -C "$REPO_DIR" fetch origin "$BRANCH"
    git -C "$REPO_DIR" checkout "$BRANCH"
    git -C "$REPO_DIR" pull --ff-only origin "$BRANCH"
elif [ -z "$(ls -A "$REPO_DIR" 2>/dev/null)" ]; then
    mkdir -p "$REPO_DIR"
    git clone --branch "$BRANCH" "$REPO_URL" "$REPO_DIR"
else
    # A mounted volume that already holds files - /workspace on RunPod is the
    # usual case. git clone refuses a non-empty target, so seed the repository
    # in place instead. Tracked files are overwritten; anything else is kept.
    echo "   $REPO_DIR is not empty and has no .git - seeding the repo in place"
    git -C "$REPO_DIR" init -q
    git -C "$REPO_DIR" remote add origin "$REPO_URL" 2>/dev/null \
        || git -C "$REPO_DIR" remote set-url origin "$REPO_URL"
    git -C "$REPO_DIR" fetch --depth 1 origin "$BRANCH"
    git -C "$REPO_DIR" checkout -f -B "$BRANCH" FETCH_HEAD
fi
cd "$REPO_DIR"
echo "   $(git rev-parse --abbrev-ref HEAD) @ $(git rev-parse --short HEAD)"

echo "== 2/5  dependencies =="
# The image built from this repo's Dockerfile already has /opt/overcooked-venv.
if [ -x /opt/overcooked-venv/bin/python ]; then
    echo "   using existing /opt/overcooked-venv"
    export PATH="/opt/overcooked-venv/bin:$PATH"
else
    command -v uv >/dev/null 2>&1 || python -m pip install --no-cache-dir "uv==0.9.7"
    UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-/opt/overcooked-venv}" \
        uv sync --frozen --no-dev --extra algs --extra cuda
    export PATH="${UV_PROJECT_ENVIRONMENT:-/opt/overcooked-venv}/bin:$PATH"
fi

echo "== 3/5  GPUs =="
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader
python - <<'PY'
import jax
print("   jax", jax.__version__, "| devices:", jax.devices())
PY

echo "== 4/5  W&B =="
if [ -n "${WANDB_API_KEY:-}" ]; then
    wandb login --relogin "$WANDB_API_KEY" >/dev/null 2>&1 && echo "   logged in via WANDB_API_KEY"
else
    wandb login >/dev/null 2>&1 || {
        echo "   set WANDB_API_KEY or run 'wandb login' before starting agents" >&2
        exit 1
    }
fi

echo "== 5/5  smoke train (5 updates on GPU 0) =="
# rnn exercises the CNN encoder too, which is where Volta boxes fail with
# "<unknown cudnn status: 5003>".
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python baselines/IPPO/ippo_overcooked_v3.py \
    ENV_KWARGS.layout=split_0 ENV_KWARGS.layout_mode=cyclic ENV_KWARGS.max_steps=450 \
    ENV_KWARGS.phase_steps=150 ENV_KWARGS.reset_on_layout_change=false \
    ENV_KWARGS.include_transition_countdown=false ENV_KWARGS.include_layout_change_mask=false \
    ARCHITECTURE=rnn SEED=0 NUM_SEEDS=1 TOTAL_TIMESTEPS=327680 \
    EXPERIMENT_FOLDER=smoke recording=disabled wandb_mode=disabled \
    upload_final_checkpoint=false

echo
echo "OK. This box can train. Start agents with GPUS matching nvidia-smi above."
