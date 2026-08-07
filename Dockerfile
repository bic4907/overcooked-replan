ARG BASE_IMAGE=nvcr.io/nvidia/jax:26.04-py3
FROM ${BASE_IMAGE}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG UID=1000
ARG GID=1000
ARG USERNAME=runner
ARG INSTALL_EXTRAS=algs

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        fontconfig \
        fonts-dejavu-core \
        git \
        libgl1 \
        libglib2.0-0 \
        tmux \
    && rm -rf /var/lib/apt/lists/*

# Match the host UID/GID so bind-mounted checkpoints are not owned by root.
RUN groupadd --non-unique --gid "${GID}" "${USERNAME}" \
    && useradd \
        --create-home \
        --no-log-init \
        --non-unique \
        --uid "${UID}" \
        --gid "${GID}" \
        --shell /bin/bash \
        "${USERNAME}"

WORKDIR /workspace
COPY --chown=${UID}:${GID} . /workspace

# Preserve the CUDA-enabled JAX stack supplied by the NVIDIA base image while
# installing this project and its algorithm dependencies.
RUN python -m pip freeze \
        | grep -iE '^(jax|jaxlib|jax-cuda[0-9]+)' \
        > /tmp/jax-constraints.txt \
    && python -m pip install \
        --no-cache-dir \
        --constraint /tmp/jax-constraints.txt \
        -e ".[${INSTALL_EXTRAS}]" \
    && python -m pip check \
    && rm /tmp/jax-constraints.txt

ENV PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    TF_FORCE_GPU_ALLOW_GROWTH=true

USER ${USERNAME}

CMD ["/bin/bash"]
