#!/usr/bin/env bash
# Runs IPPO-CNN training + eval, then IPPO-RNN training + eval on aica.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export GPUS="${GPUS:-0 1 2 3 4 5 6 7}"
export VENV="${VENV:-$HOME/overcooked-replan-venv}"

bash "$HERE/run_baseline.sh" ippo-cnn
bash "$HERE/run_baseline.sh" ippo-cnn-eval
bash "$HERE/run_baseline.sh" ippo-rnn
bash "$HERE/run_baseline.sh" ippo-rnn-eval
