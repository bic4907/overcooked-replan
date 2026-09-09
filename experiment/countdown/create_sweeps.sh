#!/usr/bin/env bash
# Create the W&B sweeps for the transition-countdown experiment and print the
# launch chain.
#
# Usage:
#   bash experiment/countdown/create_sweeps.sh                    # all 12 cells
#   bash experiment/countdown/create_sweeps.sh switchmap          # one condition
#   bash experiment/countdown/create_sweeps.sh switchmap fcp      # one algorithm
#   bash experiment/countdown/create_sweeps.sh switchmap fcp cdon # one cell
#
#   STAGE=train bash experiment/countdown/create_sweeps.sh        # training only
#   STAGE=eval  bash experiment/countdown/create_sweeps.sh        # scoring only
#
# STAGE (default "all") separates the two halves. Scoring reads finished runs
# out of W&B, so it does not have to follow its cell immediately - deferring
# every eval to the end keeps the training chain uninterrupted.
#
# Both countdown arms of a cell share one training project and one evaluation
# project, so there are 6 of each rather than 12. ALGORITHM carries the arm
# (IPPO-cdon / IPPO-cdoff / FCP-cdon / FCP-cdoff / IPPO-SP-*), which is what
# keeps them apart: the evaluator identifies runs by ALGORITHM, de-duplicates on
# (algorithm, layout, seed), and each eval sweep passes exactly one arm so it
# never tries to pair policies whose observation widths differ.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENTITY="${WANDB_ENTITY:-cilab-overcooked}"

CONDITION_FILTER="${1:-}"
ALGO_FILTER="${2:-}"
COUNTDOWN_FILTER="${3:-}"
STAGE="${STAGE:-all}"

if [[ "$STAGE" != "all" && "$STAGE" != "train" && "$STAGE" != "eval" ]]; then
    echo "STAGE must be one of: all, train, eval" >&2
    exit 1
fi

command -v wandb >/dev/null 2>&1 || {
    echo "wandb is not available in the active environment" >&2
    exit 1
}

# Create one sweep and echo only its ENTITY/PROJECT/ID reference.
create_sweep() {
    local project="$1"
    local config="$2"
    local output
    output="$(wandb sweep --entity "$ENTITY" --project "$project" "$config" 2>&1)"
    local sweep_id
    sweep_id="$(sed -n 's/.*Creating sweep with ID: \([A-Za-z0-9]*\).*/\1/p' <<<"$output" | tail -1)"
    if [ -z "$sweep_id" ]; then
        echo "Failed to create sweep from $config" >&2
        echo "$output" >&2
        exit 1
    fi
    echo "${ENTITY}/${project}/${sweep_id}"
}

cd "$REPO_ROOT"

DIR="experiment/countdown/v3"
CHAINS=()
EVAL_REFS=()

for condition in multimap switchmap; do
    if [ -n "$CONDITION_FILTER" ] && [ "$condition" != "$CONDITION_FILTER" ]; then
        continue
    fi
    for algo in ippo_cnn ippo_rnn fcp; do
        if [ -n "$ALGO_FILTER" ] && [ "$algo" != "$ALGO_FILTER" ]; then
            continue
        fi
        for countdown in cdon cdoff; do
            if [ -n "$COUNTDOWN_FILTER" ] && [ "$countdown" != "$COUNTDOWN_FILTER" ]; then
                continue
            fi
            # One project per (condition, algo); the arm lives in ALGORITHM.
            project="overcooked-v3-${condition}-${algo}-cd"
            cell="overcooked-v3-${condition}-${algo}-${countdown}"
            prefix="${DIR}/${condition}_${algo}_${countdown}"
            refs=()

            if [ "$STAGE" != "eval" ]; then
                if [ "$algo" = "fcp" ]; then
                    echo "--- ${cell}: FCP population ---" >&2
                    refs+=("$(create_sweep "$project" "${prefix}_population.yaml")")
                fi
                echo "--- ${cell}: train ---" >&2
                refs+=("$(create_sweep "$project" "${prefix}_train.yaml")")
            fi
            if [ "$STAGE" != "train" ]; then
                echo "--- ${cell}: common eval ---" >&2
                eval_ref="$(create_sweep \
                    "${project}-eval" \
                    "${prefix}_eval.yaml")"
                EVAL_REFS+=("$eval_ref")
                [ "$STAGE" = "all" ] && refs+=("$eval_ref")
            fi

            if [ "${#refs[@]}" -gt 0 ]; then
                CHAINS+=("GPUS=\"\${GPUS:-0 1 2 3 4 5 6 7}\" bash experiment/run_agents_sequential.sh ${refs[*]}")
            fi
        done
    done
done

echo
if [ "$STAGE" = "eval" ]; then
    echo "# Scoring only. The eval sweeps are independent of each other,"
    echo "# so one command runs them all back to back."
    echo "GPUS=\"\${GPUS:-0 1 2 3 4 5 6 7}\" bash experiment/run_agents_sequential.sh ${EVAL_REFS[*]}"
else
    echo "# Run each line to completion; the sweeps inside a line are ordered."
    printf '%s\n' "${CHAINS[@]}"
fi
