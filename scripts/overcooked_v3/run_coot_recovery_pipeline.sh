#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_DIR}"

sweep_paths=(
    "cilab-overcooked/overcooked-v3-coot-response-candidates/f0rdtzx6"
    "cilab-overcooked/overcooked-v3-coot-pipeline/9zy1r9ni"
    "cilab-overcooked/overcooked-v3-coot-response/82wckiy9"
    "cilab-overcooked/overcooked-v3-coot-pipeline/oqrzpu7x"
)

if [[ "${RECOVERY_ONLY:-0}" != "1" ]]; then
    sweep_paths+=(
        "cilab-overcooked/overcooked-v3-coot-train/3yyndhum"
        "cilab-overcooked/overcooked-v3-coot-eval/1hrrw5z6"
    )
fi

exec bash scripts/overcooked_v3/run_wandb_agents.sh "${sweep_paths[@]}"
