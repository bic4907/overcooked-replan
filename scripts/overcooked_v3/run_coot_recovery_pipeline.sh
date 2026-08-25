#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_DIR}"

sweep_paths=(
    "cilab-overcooked/overcooked-v3-coot-response-candidates/sc4vcnkw"
    "cilab-overcooked/overcooked-v3-coot-pipeline/nam6at9w"
    "cilab-overcooked/overcooked-v3-coot-response/mq11r1ym"
    "cilab-overcooked/overcooked-v3-coot-pipeline/q6xqy5re"
)

if [[ "${RECOVERY_ONLY:-0}" != "1" ]]; then
    sweep_paths+=(
        "cilab-overcooked/overcooked-v3-coot-train/vxwsqkok"
        "cilab-overcooked/overcooked-v3-coot-eval/g98cqm98"
    )
fi

exec bash scripts/overcooked_v3/run_wandb_agents.sh "${sweep_paths[@]}"
