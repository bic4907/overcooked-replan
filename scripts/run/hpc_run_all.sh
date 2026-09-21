#!/usr/bin/env bash
# Runs FCP population then FCP best-response on hpc (A100 x 4).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export GPUS="${GPUS:-0 1 2 3}"
export VENV="${VENV:-$HOME/overcooked-replan-venv}"

bash "$HERE/run_baseline.sh" fcp-population
bash "$HERE/run_baseline.sh" fcp-train
