ARG BASE_IMAGE=python:3.12-slim-bookworm
FROM ${BASE_IMAGE}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG UID=1000
ARG GID=1000
ARG USERNAME=runner
ARG UV_VERSION=0.9.7

ENV DEBIAN_FRONTEND=noninteractive \
    UV_PROJECT_ENVIRONMENT=/opt/overcooked-venv \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    TF_FORCE_GPU_ALLOW_GROWTH=true \
    TF_CPP_MIN_LOG_LEVEL=3 \
    GLOG_minloglevel=3

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        fontconfig \
        fonts-dejavu-core \
        git \
        libgl1 \
        libglib2.0-0 \
        tmux \
    && rm -rf /var/lib/apt/lists/* \
    && python -m pip install --no-cache-dir "uv==${UV_VERSION}"

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

# Cache the locked Python/CUDA dependencies independently from source edits.
COPY --chown=${UID}:${GID} pyproject.toml uv.lock README.md LICENSE ./
RUN uv sync \
        --frozen \
        --no-dev \
        --no-install-project \
        --extra algs \
        --extra cuda

COPY --chown=${UID}:${GID} . /workspace
RUN uv sync \
        --frozen \
        --no-dev \
        --extra algs \
        --extra cuda \
    && chown -R "${UID}:${GID}" /opt/overcooked-venv

ENV PATH="/opt/overcooked-venv/bin:${PATH}"

# Login shells source /etc/profile, which unconditionally overwrites PATH and
# would drop the venv - leaving the base image python, where wandb and jax are
# not installed. A `wandb/` run-log directory in the working tree then gets
# imported as an empty namespace package, so the failure surfaces late as
# `wandb.__file__ is None` instead of ModuleNotFoundError.
RUN printf 'export PATH="/opt/overcooked-venv/bin:$PATH"\n' \
        > /etc/profile.d/10-overcooked-venv.sh

USER ${USERNAME}

CMD ["/bin/bash"]
