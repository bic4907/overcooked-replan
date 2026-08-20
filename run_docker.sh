#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(pwd -P)}"
IMAGE_NAME="${IMAGE_NAME:-overcooked-replan:uv-cuda12}"
BASE_IMAGE="${BASE_IMAGE:-python:3.12-slim-bookworm}"
UV_VERSION="${UV_VERSION:-0.9.7}"
DOCKER_SHM_SIZE="${DOCKER_SHM_SIZE:-16g}"
DOCKER_GPUS="${DOCKER_GPUS:-all}"
REBUILD_IMAGE="${REBUILD_IMAGE:-0}"
REINSTALL_PROJECT="${REINSTALL_PROJECT:-0}"
ENV_FILE="${ENV_FILE:-${PROJECT_DIR}/.env}"

build_image() {
    docker build \
        --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
        --build-arg "UID=$(id -u)" \
        --build-arg "GID=$(id -g)" \
        --build-arg "UV_VERSION=${UV_VERSION}" \
        --tag "${IMAGE_NAME}" \
        "${PROJECT_DIR}"
}

if [[ "${REBUILD_IMAGE}" == "1" ]]; then
    build_image
elif ! docker image inspect "${IMAGE_NAME}" >/dev/null 2>&1; then
    build_image
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

if [[ -f "${ENV_FILE}" ]]; then
    docker_args+=(--env-file "${ENV_FILE}")
fi

if [[ -n "${SAVES_DIR:-}" && "${SAVES_DIR}" == /* ]]; then
    if [[ ! -d "${SAVES_DIR}" ]]; then
        echo "SAVES_DIR does not exist: ${SAVES_DIR}" >&2
        exit 2
    fi
    docker_args+=(--volume "${SAVES_DIR}:${SAVES_DIR}")
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
    SAVES_DIR \
    TF_GPU_ALLOCATOR \
    WANDB_API_KEY \
    WANDB_ENTITY \
    WANDB_MODE \
    WANDB_PROJECT \
    WANDB_DIR \
    XLA_PYTHON_CLIENT_PREALLOCATE \
    XLA_FLAGS
do
    if printenv "${variable}" >/dev/null 2>&1; then
        docker_args+=(--env "${variable}")
    fi
done

if [[ $# -eq 0 ]]; then
    set -- bash
fi

if [[ "${REINSTALL_PROJECT}" == "1" ]]; then
    reinstall_project_cmd='
set -Eeuo pipefail

uv sync --frozen --no-dev --extra algs --extra cuda

exec "$@"
'
    exec docker run "${docker_args[@]}" "${IMAGE_NAME}" bash -c "${reinstall_project_cmd}" bash "$@"
fi

exec docker run "${docker_args[@]}" "${IMAGE_NAME}" "$@"
