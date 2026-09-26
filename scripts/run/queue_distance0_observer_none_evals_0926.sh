#!/usr/bin/env bash
# Poll exact new checkpoints, then run original 0921 observer-none eval variants.
set -euo pipefail

VARIANT="${1:?use test or test10}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REVISION=dual-handoff-recipe-priority-11x7-v1
grep -qx "LAYOUT_REVISION: $REVISION" "$ROOT/conf/scenario/distance_0.yaml" || exit 2
case "$VARIANT" in
    test)
        SEEDS=(0 1 2 3 4 5)
        PROJECT=overcooked-v3-ippo-rnn-observer-none-0921_test
        GPUID=0
        ;;
    test10)
        SEEDS=(0 1 2 3 4 5 6 7 8 9)
        PROJECT=overcooked-v3-ippo-rnn-observer-none-0921_test10
        GPUID=0
        ;;
    *) echo 'use test or test10' >&2; exit 2 ;;
esac

case "${EVAL_HOST:?set EVAL_HOST=aica or hpc}" in
    aica) PYTHON=/home/inchang/overcooked-replan-venv/bin/python ;;
    hpc) PYTHON=/home/jovyan/overcooked-replan-venv/bin/python ;;
    *) exit 2 ;;
esac
case " ${GPU_ALLOWLIST:?set GPU_ALLOWLIST} " in *" $GPUID "*) ;; *) exit 2 ;; esac
DONE="$ROOT/orchestrator_distance0_priority_observer_none_${VARIANT}.done"
[[ -f "$DONE" ]] && exit 0

READY="$ROOT/artifacts/wandb/distance0_priority_observer_none_ready_0926.json"
while true; do
    if "$PYTHON" -u "$ROOT/scripts/run/verify_distance0_observer_none_ready_0926.py" "$READY"; then
        break
    else
        status=$?
        (( status == 10 )) || exit "$status"
    fi
    sleep 60
done

if [[ -z "${WANDB_API_KEY:-}" && -f "$HOME/.netrc" ]]; then
    WANDB_API_KEY="$("$PYTHON" -c 'import netrc; a=netrc.netrc().authenticators("api.wandb.ai"); print(a[2] if a else "")')"
    export WANDB_API_KEY
fi
[[ -n "${WANDB_API_KEY:-}" ]] || { echo 'W&B credentials missing' >&2; exit 1; }
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.75
cd "$ROOT"
env -u LD_LIBRARY_PATH "$PYTHON" -u baselines/IPPO/eval_crossplay_overcooked_v3.py \
    cilab-overcooked/overcooked-v3-ippo-rnn-observer-none-0921_train \
    --algorithms IPPO --layout distance_0 --seeds "${SEEDS[@]}" \
    --transition-observer none --episodes 20 --max-steps 450 \
    --layout-revision "$REVISION" --gpus "$GPUID" --workers-per-gpu 4 \
    --wandb-mode online --output-project "cilab-overcooked/$PROJECT" \
    --output-dir "$ROOT/campaigns/distance0-priority-0926-rnn-observer-none-split/evaluation/$VARIANT/distance_0" \
    --save-adaptation-traces --adaptation-horizon 75 --adaptation-window 30 \
    --drop-baseline-window 60 --drop-horizon 60
touch "$DONE"
