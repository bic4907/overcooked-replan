#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_DIR}"

sweep_paths=(
    "cilab-overcooked/overcooked-v3-coot-response-candidates/6lfm72m6"
    "cilab-overcooked/overcooked-v3-coot-pipeline/w7qigk8w"
    "cilab-overcooked/overcooked-v3-coot-response/93fcaxnk"
    "cilab-overcooked/overcooked-v3-coot-pipeline/fqdzmy74"
)

if [[ "${RECOVERY_ONLY:-0}" != "1" ]]; then
    sweep_paths+=(
        "cilab-overcooked/overcooked-v3-coot-train/e22e0tlo"
        "cilab-overcooked/overcooked-v3-coot-eval/agrn47k5"
    )
fi

exec bash scripts/overcooked_v3/run_wandb_agents.sh "${sweep_paths[@]}"
