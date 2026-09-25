#!/usr/bin/env bash
# Complete the distance_8, agent_0, 10-seed comparison against the existing RNN10 run.
set -euo pipefail

KIND="${1:?use cnn or fcp}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-/home/inchang/overcooked-replan-venv/bin/python}"
export PYTHON
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export GPU_ALLOWLIST='0 1 6 7'
export OBSERVER=agent_0
export LAYOUTS=distance_8
export EVAL_EPISODES=20
export LAYOUT_REVISION_REQUIRE=dual-handoff-deadline-11x7-v1

case "$KIND" in
  cnn)
    export GPUS='0 1'
    export ALGORITHMS=cnn
    export SEED_IDS='0 1 2 3 4 5 6 7 8 9'
    export CAMPAIGN=handoff-criterion-0925-d8-cnn10
    TRAIN_DONE="$ROOT/orchestrator_d8_criterion_cnn_train.done"
    EVAL_DONE="$ROOT/orchestrator_d8_criterion_cnn_eval.done"
    if [[ ! -f "$TRAIN_DONE" ]]; then
      bash "$ROOT/scripts/run/run_topology_inversion.sh" main train \
        > "$ROOT/orchestrator_d8_criterion_cnn_train.log" 2>&1
      touch "$TRAIN_DONE"
    fi
    if [[ ! -f "$EVAL_DONE" ]]; then
      bash "$ROOT/scripts/run/run_topology_inversion.sh" main eval \
        > "$ROOT/orchestrator_d8_criterion_cnn_eval.log" 2>&1
      touch "$EVAL_DONE"
    fi
    ;;
  fcp)
    PILOT_ROOT=/home/inchang/handoff-d8-pilot4-0925
    PILOT="$PILOT_ROOT/campaigns/handoff-convention-0924-d8-pilot4/main/agent_0"
    for seed in 0 1 2 3; do
      [[ -f "$PILOT/state/fcp_distance_8_seed${seed}.done" ]] || {
        echo "Missing completed FCP pilot seed $seed" >&2; exit 1;
      }
    done
    [[ -d "$PILOT/population" ]] || {
      echo 'Missing FCP partner population' >&2; exit 1;
    }
    export GPUS='6 7'
    export ALGORITHMS=fcp
    export TRAIN_PROJECT_OVERRIDE=overcooked-v3-handoff-convention-0924-d8-pilot4-main-agent_0-fcp_train
    export CAMPAIGN=handoff-criterion-0925-d8-fcp-extension
    TRAIN_DONE="$ROOT/orchestrator_d8_criterion_fcp_train.done"
    EVAL_DONE="$ROOT/orchestrator_d8_criterion_fcp_eval.done"
    if [[ ! -f "$TRAIN_DONE" ]]; then
      SAVE_ROOT="$ROOT/campaigns/$CAMPAIGN/saves/fcp"
      STATE_ROOT="$ROOT/campaigns/$CAMPAIGN/state"
      LOG_ROOT="$ROOT/campaigns/$CAMPAIGN/logs"
      mkdir -p "$SAVE_ROOT" "$STATE_ROOT" "$LOG_ROOT"
      if [[ -z "${WANDB_API_KEY:-}" && -f "$HOME/.netrc" ]]; then
        WANDB_API_KEY="$("$PYTHON" -c 'import netrc; a=netrc.netrc().authenticators("api.wandb.ai"); print(a[2] if a else "")')"
        export WANDB_API_KEY
      fi
      train_seed() {
        local seed="$1" gpu="$2" marker="$STATE_ROOT/fcp_distance_8_seed${seed}.done"
        [[ -f "$marker" ]] && return 0
        (
          cd "$PILOT_ROOT"
          env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES="$gpu" \
            PYTHONPATH="$PILOT_ROOT" "$PYTHON" -u baselines/FCP/fcp_overcooked_v3.py \
            scenario=distance_8 "SEED=$seed" NUM_SEEDS=1 \
            TOTAL_TIMESTEPS=30000000 REW_SHAPING_HORIZON=15000000 \
            recording=disabled wandb_mode=online ENTITY=cilab-overcooked \
            "PROJECT=$TRAIN_PROJECT_OVERRIDE" "SAVES_DIR=$SAVE_ROOT" \
            "EXPERIMENT_FOLDER=$CAMPAIGN-main-agent_0" \
            ++ENV_KWARGS.transition_observer=agent_0 \
            "RUN_NAME=fcp_distance_8_agent_0_seed${seed}" \
            "FCP.population_dir=$PILOT/population" FCP.snapshots_per_policy=3 \
            FCP.minimum_population_size=18 \
            > "$LOG_ROOT/fcp_distance_8_seed${seed}.log" 2>&1
        )
        touch "$marker"
        printf '[%s] done fcp seed=%s gpu=%s\n' "$(date -Is)" "$seed" "$gpu"
      }
      (
        for seed in 4 6 8; do train_seed "$seed" 6; done
      ) > "$ROOT/orchestrator_d8_criterion_fcp_gpu6.log" 2>&1 &
      worker6=$!
      (
        for seed in 5 7 9; do train_seed "$seed" 7; done
      ) > "$ROOT/orchestrator_d8_criterion_fcp_gpu7.log" 2>&1 &
      worker7=$!
      failed=0
      wait "$worker6" || failed=1
      wait "$worker7" || failed=1
      (( failed == 0 ))
      touch "$TRAIN_DONE"
    fi
    if [[ ! -f "$EVAL_DONE" ]]; then
      mkdir -p "$ROOT/campaigns/$CAMPAIGN/evaluation"
      if [[ -z "${WANDB_API_KEY:-}" && -f "$HOME/.netrc" ]]; then
        WANDB_API_KEY="$("$PYTHON" -c 'import netrc; a=netrc.netrc().authenticators("api.wandb.ai"); print(a[2] if a else "")')"
        export WANDB_API_KEY
      fi
      env -u LD_LIBRARY_PATH "$PYTHON" -u \
        "$ROOT/baselines/IPPO/eval_crossplay_overcooked_v3.py" \
        "cilab-overcooked/$TRAIN_PROJECT_OVERRIDE" \
        --algorithms FCP --layout distance_8 \
        --seeds 0 1 2 3 4 5 6 7 8 9 --transition-observer agent_0 \
        --episodes 20 --max-steps 450 \
        --layout-revision "$LAYOUT_REVISION_REQUIRE" \
        --gpus 6 --workers-per-gpu 4 --wandb-mode online \
        --output-project cilab-overcooked/overcooked-v3-handoff-criterion-0925-d8-fcp10-eval \
        --output-dir "$ROOT/campaigns/$CAMPAIGN/evaluation" \
        --save-adaptation-traces --adaptation-horizon 75 --adaptation-window 30 \
        --drop-baseline-window 60 --drop-horizon 60 \
        > "$ROOT/orchestrator_d8_criterion_fcp_eval.log" 2>&1
      touch "$EVAL_DONE"
    fi
    ;;
  *) echo 'use cnn or fcp' >&2; exit 1 ;;
esac
