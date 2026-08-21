#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-${PROJECT_DIR}/.venv/bin/python}"
UV_BIN="${UV_BIN:-uv}"
GPUS="${GPUS:-0}"
POSTPROCESS_GPU="${POSTPROCESS_GPU:-0}"
SCORE_EPISODES="${SCORE_EPISODES:-50}"

POPULATION_OTHER_SWEEP="${POPULATION_OTHER_SWEEP:-cilab-overcooked/overcooked-v3-coot-population/47ubezl3}"
POPULATION_MULTI_RECIPE_SWEEP="${POPULATION_MULTI_RECIPE_SWEEP:-cilab-overcooked/overcooked-v3-coot-population/ueiul4xd}"
CANDIDATE_RESPONSE_OTHER_SWEEP="${CANDIDATE_RESPONSE_OTHER_SWEEP:-cilab-overcooked/overcooked-v3-coot-response-candidates/gojtpiqm}"
CANDIDATE_RESPONSE_MULTI_RECIPE_SWEEP="${CANDIDATE_RESPONSE_MULTI_RECIPE_SWEEP:-cilab-overcooked/overcooked-v3-coot-response-candidates/omh17zwt}"
SELECTED_RESPONSE_SWEEP="${SELECTED_RESPONSE_SWEEP:-cilab-overcooked/overcooked-v3-coot-response/ac9f9ssi}"
TRAIN_SWEEP="${TRAIN_SWEEP:-cilab-overcooked/overcooked-v3-coot-train/e22e0tlo}"
EVAL_SWEEP="${EVAL_SWEEP:-cilab-overcooked/overcooked-v3-coot-eval/agrn47k5}"

OTHER_LAYOUTS=(
    split_0 split_1 split_2
    outage_0 outage_1 outage_2
    distance_switch_0 distance_switch_1 distance_switch_2
)
MULTI_RECIPE_LAYOUTS=(recipe_switch_0 recipe_switch_1 recipe_switch_2)
ALL_LAYOUTS=("${OTHER_LAYOUTS[@]}" "${MULTI_RECIPE_LAYOUTS[@]}")

log() {
    printf '[%s] %s\n' "$(date '+%H:%M:%S')" "$*"
}

run_python() {
    CUDA_VISIBLE_DEVICES="${POSTPROCESS_GPU}" \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
        "${UV_BIN}" run python "$@"
}

run_sweep() {
    local sweep_path="$1"
    log "Running W&B sweep ${sweep_path}"
    GPUS="${GPUS}" PYTHON_BIN="${PYTHON_BIN}" \
        bash scripts/overcooked_v3/run_wandb_agents.sh "${sweep_path}"
}

static_preflight() {
    local sweep
    for sweep in \
        experiment/coot/population.yaml \
        experiment/coot/population_multi_recipe.yaml \
        experiment/coot/response_candidates.yaml \
        experiment/coot/response_candidates_multi_recipe.yaml \
        experiment/coot/response_hsp_only.yaml \
        experiment/coot/train.yaml \
        experiment/coot/eval.yaml; do
        run_python baselines/CooT/preflight_sweep.py "${sweep}" --static-only
    done
}

build_candidate_jobs() {
    local layout candidate_glob
    mkdir -p manifests/coot/catalogs manifests/coot/response_candidates
    for layout in "${ALL_LAYOUTS[@]}"; do
        candidate_glob="saves/coot_population/${layout}_rnn_hsp_population_hsp_*_candidate*_seed0/*candidate*.json"
        log "Merging HSP candidate catalog for ${layout}"
        run_python baselines/CooT/score_hsp_population_overcooked_v3.py \
            --candidate-result "${candidate_glob}" \
            --layout "${layout}" \
            --merge-only \
            --output "manifests/coot/catalogs/${layout}_raw.json" \
            --overwrite
        run_python baselines/CooT/build_population_manifest.py response-jobs \
            --hsp-catalog "manifests/coot/catalogs/${layout}_raw.json" \
            --layout "${layout}" \
            --all-hsp-candidates \
            --hsp-skill final \
            --verify-checkpoints \
            --output "manifests/coot/response_candidates/${layout}.json" \
            --overwrite
    done
    run_python baselines/CooT/preflight_sweep.py \
        experiment/coot/response_candidates.yaml
    run_python baselines/CooT/preflight_sweep.py \
        experiment/coot/response_candidates_multi_recipe.yaml
}

score_candidates_and_build_selected_jobs() {
    local layout response_glob
    mkdir -p manifests/coot/catalogs manifests/coot/response_jobs_hsp_only \
        saves/coot_population_scores
    for layout in "${ALL_LAYOUTS[@]}"; do
        response_glob="saves/coot_responses/${layout}_rnn_*/response_job*.json"
        log "Scoring HSP candidates for ${layout}"
        run_python baselines/CooT/score_hsp_population_overcooked_v3.py \
            --catalog "manifests/coot/catalogs/${layout}_raw.json" \
            --response-result "${response_glob}" \
            --layout "${layout}" \
            --episodes "${SCORE_EPISODES}" \
            --cache-dir "saves/coot_population_scores/${layout}" \
            --output "manifests/coot/catalogs/${layout}_scored.json" \
            --overwrite
        run_python baselines/CooT/build_population_manifest.py response-jobs \
            --hsp-catalog "manifests/coot/catalogs/${layout}_scored.json" \
            --layout "${layout}" \
            --allow-hsp-only \
            --hsp-skill mid \
            --verify-checkpoints \
            --output "manifests/coot/response_jobs_hsp_only/${layout}.json" \
            --overwrite
    done
    run_python baselines/CooT/preflight_sweep.py \
        experiment/coot/response_hsp_only.yaml
}

build_datasets() {
    local layout response_glob dataset_metadata
    mkdir -p manifests/coot/train datasets/coot
    for layout in "${ALL_LAYOUTS[@]}"; do
        response_glob="saves/coot_responses/${layout}_rnn_*/response_job*.json"
        run_python baselines/CooT/build_population_manifest.py build-pairs \
            --hsp-catalog "manifests/coot/catalogs/${layout}_scored.json" \
            --layout "${layout}" \
            --allow-hsp-only \
            --response-results "${response_glob}" \
            --verify-checkpoints \
            --output "manifests/coot/train/${layout}.json" \
            --overwrite

        dataset_metadata="datasets/coot/${layout}/metadata.json"
        if [[ -f "${dataset_metadata}" ]]; then
            log "Reusing completed dataset ${dataset_metadata}"
            continue
        fi
        log "Collecting CooT trajectories for ${layout}"
        run_python baselines/CooT/collect_overcooked_v3.py \
            --manifest "manifests/coot/train/${layout}.json" \
            --output-root datasets/coot \
            --layout "${layout}" \
            --overwrite
    done
    run_python baselines/CooT/preflight_sweep.py experiment/coot/train.yaml
}

main() {
    if [[ ! -x "${PYTHON_BIN}" ]]; then
        echo "Python executable not found: ${PYTHON_BIN}" >&2
        exit 127
    fi
    if ! command -v "${UV_BIN}" >/dev/null 2>&1; then
        echo "uv executable not found: ${UV_BIN}" >&2
        exit 127
    fi

    log "Checking all CooT sweep programs, projects, and namespaces"
    static_preflight

    run_sweep "${POPULATION_OTHER_SWEEP}"
    run_sweep "${POPULATION_MULTI_RECIPE_SWEEP}"

    build_candidate_jobs
    run_sweep "${CANDIDATE_RESPONSE_OTHER_SWEEP}"
    run_sweep "${CANDIDATE_RESPONSE_MULTI_RECIPE_SWEEP}"

    score_candidates_and_build_selected_jobs
    run_sweep "${SELECTED_RESPONSE_SWEEP}"

    build_datasets
    run_sweep "${TRAIN_SWEEP}"

    run_python baselines/CooT/preflight_sweep.py experiment/coot/eval.yaml
    run_sweep "${EVAL_SWEEP}"
    log "CooT 12-layout HSP-only pipeline completed through seed-wise SP/XP eval"
}

main "$@"
