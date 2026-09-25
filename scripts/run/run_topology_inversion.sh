#!/usr/bin/env bash
# Explicit launch only. Linux/Bash; run from an isolated copy on the GPU host.
# GPU_ALLOWLIST="0 1 6 7" GPUS="0 1 6 7" bash scripts/run/run_topology_inversion.sh pilot train
# LAYOUTS="distance_2 distance_3" GPU_ALLOWLIST="0 1 6 7" GPUS="0 1 6 7" bash scripts/run/run_topology_inversion.sh main train
# ALGORITHMS="fcp" on H100 and ALGORITHMS="rnn" on HPC split ownership.
# Repeat with the same settings and action=eval after training finishes.
set -euo pipefail
# Give each background worker its own process group, including its children.
set -m

PROFILE="${1:?use pilot or main}"
ACTION="${2:?use train or eval}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-$HOME/overcooked-replan-venv/bin/python}"
OBSERVER="${OBSERVER:-agent_0}"
GPUS="${GPUS:?set the currently available GPU IDs explicitly}"
read -r -a GPU_LIST <<< "$GPUS"
(( ${#GPU_LIST[@]} > 0 )) || exit 1
declare -A SEEN_GPUS=()
for gpu in "${GPU_LIST[@]}"; do
    [[ "$gpu" =~ ^[0-9]+$ && -z "${SEEN_GPUS[$gpu]:-}" ]] || {
        echo "GPU IDs must be distinct nonnegative integers." >&2; exit 1;
    }
    SEEN_GPUS[$gpu]=1
done
if [[ -n "${GPU_ALLOWLIST:-}" ]]; then
    for gpu in "${GPU_LIST[@]}"; do
        case " $GPU_ALLOWLIST " in
            *" $gpu "*) ;;
            *) echo "GPU $gpu is outside GPU_ALLOWLIST=$GPU_ALLOWLIST" >&2; exit 1 ;;
        esac
    done
fi
case "$PROFILE" in
    pilot)
        read -r -a MAPS <<< "${LAYOUTS:-distance_2 distance_3}"
        SEEDS=(0)
        POP_SEEDS=(100 101)
        EPISODES=5
        ;;
    main)
        read -r -a MAPS <<< "${LAYOUTS:-distance_2 distance_3}"
        SEEDS=(0 1 2 3 4 5 6 7 8 9)
        POP_SEEDS=(100 101 102 103 104 105)
        EPISODES=20
        ;;
    *) echo "Unknown profile: $PROFILE" >&2; exit 1 ;;
esac
if [[ -n "${SEED_IDS:-}" ]]; then read -r -a SEEDS <<< "$SEED_IDS"; fi
if [[ -n "${POP_SEED_IDS:-}" ]]; then read -r -a POP_SEEDS <<< "$POP_SEED_IDS"; fi
EPISODES="${EVAL_EPISODES:-$EPISODES}"
read -r -a ALGORITHM_LIST <<< "${ALGORITHMS:-rnn fcp}"
(( ${#ALGORITHM_LIST[@]} > 0 )) || exit 1
declare -A SEEN_ALGORITHMS=()
for kind in "${ALGORITHM_LIST[@]}"; do
    case "$kind" in cnn|rnn|fcp) ;; *) echo "Use ALGORITHMS='cnn', 'rnn', or 'fcp'." >&2; exit 1 ;; esac
    [[ -z "${SEEN_ALGORITHMS[$kind]:-}" ]] || { echo "Duplicate algorithm: $kind" >&2; exit 1; }
    SEEN_ALGORITHMS[$kind]=1
done
case "$ACTION" in train|eval) ;; *) echo "Use train or eval" >&2; exit 1 ;; esac
(( ${#MAPS[@]} > 0 )) || exit 1
case "$OBSERVER" in both|agent_0|agent_1|none) ;; *) exit 1 ;; esac
for layout in "${MAPS[@]}"; do
    case "$layout" in distance_0|distance_2|distance_3|distance_7|distance_8|distance_9) ;; *) exit 1 ;; esac
done

CAMPAIGN="${CAMPAIGN:-handoff-site-0924-v4-d23-seed10}"
RUN_ROOT="$ROOT/campaigns/$CAMPAIGN/$PROFILE/$OBSERVER"
PROJECT_PREFIX="${PROJECT_PREFIX:-overcooked-v3-$CAMPAIGN-$PROFILE-$OBSERVER}"
POP_ROOT="${REUSE_POPULATION_DIR:-$RUN_ROOT/population}"
mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/state" "$POP_ROOT"
cd "$ROOT"

# Keep pilot, main, observer conditions, and populations separate. Refuse
# source/settings drift when resuming a directory that already contains runs.
manifest="$(
    printf '%s\n' "$PYTHON" "$OBSERVER" "${MAPS[*]}" "${SEEDS[*]}" "${POP_SEEDS[*]}" "${ALGORITHM_LIST[*]}" "episodes=$EPISODES" 'steps=30000000' "project=$PROJECT_PREFIX" "train_project=${TRAIN_PROJECT_OVERRIDE:-}" "eval_project=${EVAL_PROJECT_OVERRIDE:-}" "population_project=${POPULATION_PROJECT_OVERRIDE:-}" "population_wandb_mode=${POPULATION_WANDB_MODE:-disabled}" "population=$POP_ROOT" "population_source=${POPULATION_SOURCE_LAYOUT:-}" "layout_revision_require=${LAYOUT_REVISION_REQUIRE:-}" "source_layout=${SOURCE_LAYOUT_OVERRIDE:-}"
    scenario_files=()
    for layout in "${MAPS[@]}"; do scenario_files+=("conf/scenario/$layout.yaml"); done
    sha256sum jaxmarl/environments/overcooked_v3/{dynamic_layout_data,dynamic_layouts}.py \
        "${scenario_files[@]}" conf/{ippo,fcp}_overcooked_v3.yaml \
        baselines/IPPO/ippo_overcooked_v3.py baselines/FCP/fcp_overcooked_v3.py \
        scripts/run/run_topology_inversion.sh
)"
if [[ -f "$RUN_ROOT/manifest.txt" ]]; then
    [[ "$(cat "$RUN_ROOT/manifest.txt")" == "$manifest" ]] || {
        echo "Campaign settings/source changed; use a new CAMPAIGN name." >&2; exit 1;
    }
else
    printf '%s\n' "$manifest" > "$RUN_ROOT/manifest.txt"
fi

# The project entrypoints require the key in the environment for online mode.
# Reuse an existing login without displaying its credential.
if [[ -z "${WANDB_API_KEY:-}" && -f "$HOME/.netrc" ]]; then
    WANDB_API_KEY="$("$PYTHON" -c 'import netrc; a=netrc.netrc().authenticators("api.wandb.ai"); print(a[2] if a else "")')"
    export WANDB_API_KEY
fi
if [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "W&B credentials required for checkpoint-based project evaluation." >&2
    exit 1
fi

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.75
WORKERS=()
cleanup() {
    local pid
    for pid in "${WORKERS[@]}"; do kill -- "-$pid" 2>/dev/null || true; done
    wait || true
}
trap 'cleanup; exit 130' INT TERM

train_job() {
    local kind="$1" gpu="$2" layout="$3" seed="$4"
    local project="${TRAIN_PROJECT_OVERRIDE:-$PROJECT_PREFIX-${kind}_train}"
    if [[ "$kind" == population ]]; then
        project="${POPULATION_PROJECT_OVERRIDE:-$PROJECT_PREFIX-population_train}"
    fi
    local program=baselines/IPPO/ippo_overcooked_v3.py
    local extra=(ARCHITECTURE=rnn)
    if [[ "$kind" == cnn ]]; then extra=(ARCHITECTURE=cnn); fi
    local video_mode=disabled run_wandb_mode=online
    # Inspect pilot partners as well as learners; retain a main seed-0 replay.
    if [[ "$PROFILE" == pilot || ( "$kind" != population && "$seed" == 0 ) ]]; then
        video_mode=enabled
    fi
    if [[ "$kind" == population ]]; then
        extra+=("SAVES_DIR=$POP_ROOT" 'CHECKPOINT_FRACTIONS=[0.1,0.5,1.0]'
            upload_final_checkpoint=false)
        if [[ "$PROFILE" == main ]]; then
            run_wandb_mode="${POPULATION_WANDB_MODE:-disabled}"
        fi
    elif [[ "$kind" == fcp ]]; then
        program=baselines/FCP/fcp_overcooked_v3.py
        extra+=("FCP.population_dir=$POP_ROOT" FCP.snapshots_per_policy=3
            "FCP.minimum_population_size=$((${#POP_SEEDS[@]} * 3))")
        if [[ -n "${POPULATION_SOURCE_LAYOUT:-}" ]]; then
            extra+=("+FCP.population_source_layout=$POPULATION_SOURCE_LAYOUT")
        fi
    fi
    local id="${kind}_${layout}_seed${seed}"
    [[ -f "$RUN_ROOT/state/$id.done" ]] && return 0
    printf '[%s] start %s GPU=%s\n' "$(date -Is)" "$id" "$gpu"
    env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES="$gpu" \
        "$PYTHON" -u "$program" "scenario=$layout" "SEED=$seed" NUM_SEEDS=1 \
        TOTAL_TIMESTEPS=30000000 REW_SHAPING_HORIZON=15000000 \
        "recording=$video_mode" "wandb_mode=$run_wandb_mode" ENTITY=cilab-overcooked \
        "PROJECT=$project" "SAVES_DIR=$RUN_ROOT/saves/$kind" \
        "EXPERIMENT_FOLDER=$CAMPAIGN-$PROFILE-$OBSERVER" \
        "++ENV_KWARGS.transition_observer=$OBSERVER" \
        "RUN_NAME=${kind}_${layout}_${OBSERVER}_seed${seed}" \
        "${extra[@]}" > "$RUN_ROOT/logs/$id.log" 2>&1
    touch "$RUN_ROOT/state/$id.done"
    printf '[%s] done %s\n' "$(date -Is)" "$id"
}

eval_job() {
    local kind="$1" gpu="$2" layout="$3" algorithm=IPPO
    if [[ "$layout" == distance_0 && -z "${LAYOUT_REVISION_REQUIRE:-}" ]]; then
        echo "distance_0 evaluation requires LAYOUT_REVISION_REQUIRE" >&2
        exit 1
    fi
    local source_project="${TRAIN_PROJECT_OVERRIDE:-$PROJECT_PREFIX-${kind}_train}"
    local output_project="${EVAL_PROJECT_OVERRIDE:-$PROJECT_PREFIX-${kind}_eval}"
    local source_args=()
    if [[ -n "${LAYOUT_REVISION_REQUIRE:-}" ]]; then
        source_args+=(--layout-revision "$LAYOUT_REVISION_REQUIRE")
    fi
    if [[ -n "${SOURCE_LAYOUT_OVERRIDE:-}" ]]; then
        source_args+=(--source-layout "$SOURCE_LAYOUT_OVERRIDE")
    fi
    [[ "$kind" == fcp ]] && algorithm=FCP
    local id="eval_${kind}_${layout}"
    [[ -f "$RUN_ROOT/state/$id.done" ]] && return 0
    env -u LD_LIBRARY_PATH "$PYTHON" -u baselines/IPPO/eval_crossplay_overcooked_v3.py \
        "cilab-overcooked/$source_project" \
        --algorithms "$algorithm" --layout "$layout" --seeds "${SEEDS[@]}" \
        --transition-observer "$OBSERVER" --episodes "$EPISODES" --max-steps 450 \
        "${source_args[@]}" \
        --gpus "$gpu" --workers-per-gpu 4 --wandb-mode online \
        --output-project "cilab-overcooked/$output_project" \
        --output-dir "$RUN_ROOT/evaluation/$kind/$layout" \
        --save-adaptation-traces --adaptation-horizon 75 --adaptation-window 30 \
        --drop-baseline-window 60 --drop-horizon 60 \
        > "$RUN_ROOT/logs/$id.log" 2>&1
    touch "$RUN_ROOT/state/$id.done"
}

run_stage() {
    local kind="$1" slot gpu layout seed index failed=0 pid
    local selected_seeds=("${SEEDS[@]}")
    [[ "$kind" == population ]] && selected_seeds=("${POP_SEEDS[@]}")
    WORKERS=()
    for slot in "${!GPU_LIST[@]}"; do
        gpu="${GPU_LIST[$slot]}"
        (
            # Keep descendants in this worker's process group for cancellation.
            set +m
            index=0
            for layout in "${MAPS[@]}"; do
                if [[ "$ACTION" == eval ]]; then
                    if (( index % ${#GPU_LIST[@]} == slot )); then
                        eval_job "$kind" "$gpu" "$layout"
                    fi
                    ((index += 1))
                else
                    for seed in "${selected_seeds[@]}"; do
                        if (( index % ${#GPU_LIST[@]} == slot )); then
                            train_job "$kind" "$gpu" "$layout" "$seed"
                        fi
                        ((index += 1))
                    done
                fi
            done
        ) &
        WORKERS+=("$!")
    done
    for pid in "${WORKERS[@]}"; do wait "$pid" || failed=1; done
    WORKERS=()
    (( failed == 0 ))
}

if [[ "$ACTION" == train ]]; then
    if [[ -n "${SEEN_ALGORITHMS[fcp]:-}" && -z "${REUSE_POPULATION_DIR:-}" ]]; then
        run_stage population
    fi
else
    # Never silently evaluate a partial seed set as the completed campaign.
    for kind in "${ALGORITHM_LIST[@]}"; do
        for layout in "${MAPS[@]}"; do
            for seed in "${SEEDS[@]}"; do
                [[ -f "$RUN_ROOT/state/${kind}_${layout}_seed${seed}.done" ]] || {
                    echo "Missing completed training: $kind $layout seed=$seed" >&2
                    exit 1
                }
            done
        done
    done
fi
for kind in "${ALGORITHM_LIST[@]}"; do run_stage "$kind"; done
printf '[%s] %s %s complete\n' "$(date -Is)" "$PROFILE" "$ACTION"
