#!/usr/bin/env bash
# Add FCP seeds 4-9 without repeating the completed pilot population or seeds.
set -euo pipefail

ACTION="${1:?use train or eval}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-$HOME/overcooked-replan-venv/bin/python}"
export PYTHON
export GPUS="${GPUS:-0 1}" GPU_ALLOWLIST="0 1 6 7"
export OBSERVER=both LAYOUTS=distance_10 ALGORITHMS=fcp
export SEED_IDS="4 5 6 7 8 9" POP_SEED_IDS="100 101 102 103 104 105"
export EVAL_EPISODES=20
export LAYOUT_REVISION_REQUIRE=dual-handoff-recipe-priority-11x7-v1
export PROJECT_PREFIX=overcooked-v3-mapsearch-0925-d10-priority-full10
export CAMPAIGN=distance10-priority-0925-fcp-extension6

PILOT="$ROOT/campaigns/distance10-priority-0925-fcp-pilot4/main/both"
export REUSE_POPULATION_DIR="$PILOT/population"
EXTENSION="$ROOT/campaigns/$CAMPAIGN/main/both"
TRAIN_DONE="$ROOT/orchestrator_distance10_fcp_extension6_train.done"
EVAL_DONE="$ROOT/orchestrator_distance10_fcp_full10_eval.done"

case "$ACTION" in
    train)
        for seed in 100 101 102 103 104 105; do
            [[ -f "$PILOT/state/population_distance_10_seed${seed}.done" ]] || {
                echo "Pilot population seed $seed is incomplete" >&2; exit 1;
            }
        done
        [[ -d "$REUSE_POPULATION_DIR" ]] || {
            echo "Pilot population directory is missing" >&2; exit 1;
        }
        if [[ ! -f "$TRAIN_DONE" ]]; then
            bash "$ROOT/scripts/run/run_topology_inversion.sh" main train \
                > "$ROOT/orchestrator_distance10_fcp_extension6_train.log" 2>&1
            touch "$TRAIN_DONE"
        fi
        ;;
    eval)
        [[ -f "$TRAIN_DONE" ]] || { echo "Extension training incomplete" >&2; exit 1; }
        [[ -f "$ROOT/orchestrator_distance10_fcp_train.done" ]] || {
            echo "Pilot FCP training incomplete" >&2; exit 1;
        }
        for seed in 0 1 2 3; do
            [[ -f "$PILOT/state/fcp_distance_10_seed${seed}.done" ]] || exit 1
        done
        for seed in 4 5 6 7 8 9; do
            [[ -f "$EXTENSION/state/fcp_distance_10_seed${seed}.done" ]] || exit 1
        done
        if [[ ! -f "$EVAL_DONE" ]]; then
            if [[ -z "${WANDB_API_KEY:-}" && -f "$HOME/.netrc" ]]; then
                WANDB_API_KEY="$("$PYTHON" -c 'import netrc; a=netrc.netrc().authenticators("api.wandb.ai"); print(a[2] if a else "")')"
                export WANDB_API_KEY
            fi
            [[ -n "${WANDB_API_KEY:-}" ]] || { echo "W&B credentials required" >&2; exit 1; }
            read -r -a GPU_LIST <<< "$GPUS"
            for gpu in "${GPU_LIST[@]}"; do
                case " $GPU_ALLOWLIST " in
                    *" $gpu "*) ;;
                    *) echo "GPU $gpu is outside allowlist" >&2; exit 1 ;;
                esac
            done
            OUT="$ROOT/campaigns/distance10-priority-0925-fcp-full10-combined/evaluation/distance_10"
            mkdir -p "$OUT"
            export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
            export XLA_PYTHON_CLIENT_PREALLOCATE=false
            export XLA_PYTHON_CLIENT_MEM_FRACTION=0.75
            cd "$ROOT"
            env -u LD_LIBRARY_PATH "$PYTHON" -u \
                baselines/IPPO/eval_crossplay_overcooked_v3.py \
                "cilab-overcooked/overcooked-v3-mapsearch-0925-d10-priority-pilot4-fcp_train" \
                --additional-source "cilab-overcooked/$PROJECT_PREFIX-fcp_train" distance_10 \
                --algorithms FCP --layout distance_10 \
                --layout-revision "$LAYOUT_REVISION_REQUIRE" \
                --seeds 0 1 2 3 4 5 6 7 8 9 --transition-observer both \
                --episodes 20 --max-steps 450 --vmap-indices 0 \
                --gpus "${GPU_LIST[@]}" --workers-per-gpu 4 --wandb-mode online \
                --output-project "cilab-overcooked/$PROJECT_PREFIX-fcp_eval" \
                --output-dir "$OUT" --save-adaptation-traces \
                --adaptation-horizon 75 --adaptation-window 30 \
                --drop-baseline-window 60 --drop-horizon 60 \
                > "$ROOT/orchestrator_distance10_fcp_full10_eval.log" 2>&1
            touch "$EVAL_DONE"
        fi
        ;;
    *) echo "use train or eval" >&2; exit 1 ;;
esac
