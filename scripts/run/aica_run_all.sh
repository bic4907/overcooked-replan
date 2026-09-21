#!/usr/bin/env bash
# Runs IPPO-CNN then IPPO-RNN on aica (H100 x 8).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export GPUS="${GPUS:-0 1 2 3 4 5 6 7}"
export VENV="${VENV:-$HOME/overcooked-replan-venv}"

bash "$HERE/run_baseline.sh" ippo-cnn
bash "$HERE/run_baseline.sh" ippo-rnn
