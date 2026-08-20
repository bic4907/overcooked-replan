# Docker workflow

The root `Dockerfile` reproduces the direct-training environment from
`uv.lock`: Python 3.12, the `algs` dependencies, and the CUDA 12 JAX wheels are
installed into `/opt/overcooked-venv`. The host only needs a compatible NVIDIA
driver and NVIDIA Container Toolkit.

The image intentionally does not inherit the host `LD_LIBRARY_PATH`. JAX GPU
preallocation is disabled by default, matching the stable direct-run command.

## Build

Build directly from the repository root:

```bash
docker build \
  --build-arg UID="$(id -u)" \
  --build-arg GID="$(id -g)" \
  --tag overcooked-replan:uv-cuda12 \
  .
```

`run_docker.sh` builds this image automatically when it is missing. Force a
rebuild after changing `pyproject.toml`, `uv.lock`, or the Dockerfile:

```bash
REBUILD_IMAGE=1 ./run_docker.sh bash
```

## Run

The helper passes `.env`, mounts the repository at `/workspace`, uses the host
UID/GID, and enables available GPUs:

```bash
./run_docker.sh bash
```

Verify the locked environment and GPU before a long run:

```bash
./run_docker.sh python -c \
  'import jax; print(jax.__version__); print(jax.devices())'
```

Run training:

```bash
CUDA_VISIBLE_DEVICES=0 ./run_docker.sh \
  python -u baselines/IPPO/ippo_overcooked_v3.py \
  scenario=splitsig_0 SEED=0 NUM_SEEDS=1
```

The source tree is editable at `/workspace`, so ordinary Python source changes
do not require reinstalling dependencies. If project metadata changed without
rebuilding, resync the locked environment once:

```bash
REINSTALL_PROJECT=1 ./run_docker.sh \
  python -c 'import jaxmarl; print(jaxmarl.__file__)'
```

`saves/` and `wandb/` persist because the complete repository is bind-mounted
from the host. An absolute host `SAVES_DIR` is mounted additionally when that
environment variable is set and the directory exists.
