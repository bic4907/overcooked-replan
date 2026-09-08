#!/usr/bin/env bash

# Run the `_1` baseline suite on one eight-GPU Runpod worker. The two roles are
# deliberately separate because FCP needs its local population checkpoints.
set -Eeuo pipefail

ROLE="${1:?usage: $0 <ippo|fcp>}"
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export GPUS="${GPUS:-0 1 2 3 4 5 6 7}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

cd "${REPO_ROOT}"

case "${ROLE}" in
    ippo)
        bash experiment/run_agents_sequential.sh \
            cilab-overcooked/overcooked-v3-ippo-easy1_train/ranyyq1p \
            cilab-overcooked/overcooked-v3-ippo-rnn-easy1_train/mglz038p \
            cilab-overcooked/overcooked-v3-ippo-easy1_eval/j9y1n577 \
            cilab-overcooked/overcooked-v3-ippo-rnn-easy1_eval/7wgsgrh5
        ;;
    fcp)
        bash experiment/run_agents_sequential.sh \
            cilab-overcooked/overcooked-v3-fcp-easy1_population/zoaxacrm \
            cilab-overcooked/overcooked-v3-fcp-easy1_train/33zac3rf \
            cilab-overcooked/overcooked-v3-fcp-easy1_eval/2vep0tev
        ;;
    *)
        echo "Unknown role: ${ROLE}; expected ippo or fcp" >&2
        exit 2
        ;;
esac

touch "/workspace/easy1-${ROLE}.complete"
