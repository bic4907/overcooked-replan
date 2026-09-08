#!/usr/bin/env bash

# Run the `_1` baseline suite on one eight-GPU Runpod worker. The two roles are
# deliberately separate because FCP needs its local population checkpoints.
set -Eeuo pipefail

ROLE="${1:?usage: $0 <ippo|rnn-recovery1|fcp|fcp-post|fcp-recovery1|outage1-v2-ippo-rnn|outage1-v2-fcp>}"
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export GPUS="${GPUS:-0 1 2 3 4 5 6 7}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export WANDB_MODE=online

# W&B's API client accepts ~/.netrc, but the training entrypoints deliberately
# require WANDB_API_KEY before enabling online logging. Export the same token
# without printing it so sweep children do not silently fall back to offline.
if [[ -z "${WANDB_API_KEY:-}" && -f "${HOME}/.netrc" ]]; then
    WANDB_API_KEY="$(awk '$1 == "password" {print $2; exit}' "${HOME}/.netrc")"
    export WANDB_API_KEY
fi
if [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "WANDB_API_KEY is unavailable; refusing to start an offline sweep" >&2
    exit 1
fi

cd "${REPO_ROOT}"

case "${ROLE}" in
    ippo)
        bash experiment/run_agents_sequential.sh \
            cilab-overcooked/overcooked-v3-ippo-easy1_train/z7rfiie5 \
            cilab-overcooked/overcooked-v3-ippo-rnn-easy1_train/mglz038p \
            cilab-overcooked/overcooked-v3-ippo-easy1_eval/j9y1n577 \
            cilab-overcooked/overcooked-v3-ippo-rnn-easy1_eval/7wgsgrh5
        ;;
    rnn-recovery1)
        bash experiment/run_agents_sequential.sh \
            cilab-overcooked/overcooked-v3-ippo-rnn-easy1_train/zegiopgh \
            cilab-overcooked/overcooked-v3-ippo-rnn-easy1-recovery1_eval/wvet20aa \
            cilab-overcooked/overcooked-v3-ippo-easy1_eval/j9y1n577
        ;;
    fcp)
        bash experiment/run_agents_sequential.sh \
            cilab-overcooked/overcooked-v3-fcp-easy1_population/46qfe84z
        python scripts/verify_easy1_fcp_population.py \
            saves/fcp_easy1/fcp_population
        bash experiment/run_agents_sequential.sh \
            cilab-overcooked/overcooked-v3-fcp-easy1_train/ayst2t4g \
            cilab-overcooked/overcooked-v3-fcp-easy1_eval/040dx6ha
        ;;
    fcp-post)
        python scripts/verify_easy1_fcp_population.py \
            saves/fcp_easy1/fcp_population
        bash experiment/run_agents_sequential.sh \
            cilab-overcooked/overcooked-v3-fcp-easy1_train/ayst2t4g \
            cilab-overcooked/overcooked-v3-fcp-easy1_eval/040dx6ha
        ;;
    fcp-recovery1)
        python scripts/verify_easy1_fcp_population.py \
            saves/fcp_easy1/fcp_population
        bash experiment/run_agents_sequential.sh \
            cilab-overcooked/overcooked-v3-fcp-easy1-recovery1_train/iq5qa8n0 \
            cilab-overcooked/overcooked-v3-fcp-easy1-recovery1_eval/6wtspgb1
        ;;
    outage1-v2-ippo-rnn)
        bash experiment/run_agents_sequential.sh \
            cilab-overcooked/overcooked-v3-ippo-outage1-v2_train/8ixm9jvx \
            cilab-overcooked/overcooked-v3-ippo-outage1-v2_eval/2laq6eg0 \
            cilab-overcooked/overcooked-v3-ippo-rnn-outage1-v2_train/nlxajili \
            cilab-overcooked/overcooked-v3-ippo-rnn-outage1-v2_eval/9k8ezs3b
        ;;
    outage1-v2-fcp)
        bash experiment/run_agents_sequential.sh \
            cilab-overcooked/overcooked-v3-fcp-outage1-v2_population/xfkub379
        python scripts/verify_easy1_fcp_population.py \
            saves/fcp_outage1_v2/fcp_population \
            --layouts outage_1
        bash experiment/run_agents_sequential.sh \
            cilab-overcooked/overcooked-v3-fcp-outage1-v2_train/hqovh4vz \
            cilab-overcooked/overcooked-v3-fcp-outage1-v2_eval/zzc8mi9z
        ;;
    *)
        echo "Unknown role: ${ROLE}" >&2
        exit 2
        ;;
esac

touch "/workspace/easy1-${ROLE}.complete"
