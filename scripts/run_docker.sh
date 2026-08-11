#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_NAME="${IMAGE_NAME:-ghkdrmaghks/overcooked-replan:latest}"
BASE_IMAGE="${BASE_IMAGE:-nvcr.io/nvidia/jax:26.04-py3}"
INSTALL_EXTRAS="${INSTALL_EXTRAS:-algs}"
DOCKER_SHM_SIZE="${DOCKER_SHM_SIZE:-16g}"
DOCKER_GPUS="${DOCKER_GPUS:-all}"
REBUILD_IMAGE="${REBUILD_IMAGE:-0}"
NAS_PROJECT_DIR="${NAS_PROJECT_DIR:-/mnt/nas/overcooked-replan}"

build_image() {
    docker build \
        --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
        --build-arg "UID=$(id -u)" \
        --build-arg "GID=$(id -g)" \
        --build-arg "INSTALL_EXTRAS=${INSTALL_EXTRAS}" \
        --tag "${IMAGE_NAME}" \
        "${PROJECT_DIR}"
}

if [[ "${REBUILD_IMAGE}" == "1" ]]; then
    build_image
elif ! docker image inspect "${IMAGE_NAME}" >/dev/null 2>&1; then
    docker pull "${IMAGE_NAME}"
fi

docker_args=(
    --rm
    --init
    --user "$(id -u):$(id -g)"
    --volume /etc/passwd:/etc/passwd:ro
    --volume /etc/group:/etc/group:ro
    --env HOME=/tmp
    --shm-size "${DOCKER_SHM_SIZE}"
    --volume "${PROJECT_DIR}:/workspace"
    --workdir /workspace
)

if [[ -d "${NAS_PROJECT_DIR}" ]]; then
    docker_args+=(--volume "${NAS_PROJECT_DIR}:${NAS_PROJECT_DIR}")
else
    echo "Warning: NAS model directory was not found: ${NAS_PROJECT_DIR}" >&2
    echo "Training and automatic evaluation require this directory." >&2
fi

if [[ -t 0 && -t 1 ]]; then
    docker_args+=(-it)
fi

if [[ "${DOCKER_GPUS}" != "none" ]]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        docker_args+=(--gpus "${DOCKER_GPUS}")
    else
        echo "Warning: nvidia-smi was not found; running without GPU access." >&2
    fi
fi

for variable in \
    CUDA_VISIBLE_DEVICES \
    JAX_PLATFORMS \
    TF_GPU_ALLOCATOR \
    WANDB_API_KEY \
    WANDB_ENTITY \
    WANDB_MODE \
    WANDB_PROJECT \
    XLA_FLAGS
do
    if printenv "${variable}" >/dev/null 2>&1; then
        docker_args+=(--env "${variable}")
    fi
done

if [[ $# -eq 0 ]]; then
    set -- bash
fi

exec docker run "${docker_args[@]}" "${IMAGE_NAME}" "$@"
