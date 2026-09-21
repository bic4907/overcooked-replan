#!/usr/bin/env bash
# Runs FCP population, then FCP best-response training, then FCP eval on hpc.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export GPUS="${GPUS:-0 1 2 3}"
export VENV="${VENV:-$HOME/overcooked-replan-venv}"

bash "$HERE/run_baseline.sh" fcp-population
bash "$HERE/run_baseline.sh" fcp-train
bash "$HERE/run_baseline.sh" fcp-eval
