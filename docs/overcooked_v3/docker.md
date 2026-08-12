# Docker workflow

The root `Dockerfile` extends NVIDIA's JAX CUDA image and preserves its
CUDA-enabled JAX stack while installing this project with the `algs` extra.

## Build

Build directly from the repository root:

```bash
docker build \
  --build-arg BASE_IMAGE=nvcr.io/nvidia/jax:26.04-py3 \
  --build-arg UID="$(id -u)" \
  --build-arg GID="$(id -g)" \
  --tag bic4907/overcooked:cu13 \
  .
```

The image suppresses non-fatal XLA autotuning diagnostics while preserving
fatal errors. Do not pass an unsupported `XLA_FLAGS` value from the host.

## Run

The root helper loads `.env`, mounts the repository, enables available GPUs,
and uses the local image by default:

```bash
./run_docker.sh bash
```

Run a short training job:

```bash
./run_docker.sh python -u baselines/IPPO/ippo_overcooked_v3.py \
  scenario=splitsig_0 \
  SEED=0 \
  NUM_SEEDS=1
```

Set `REBUILD_IMAGE=1` to rebuild before running:

```bash
REBUILD_IMAGE=1 ./run_docker.sh bash
```

When the repository is bind-mounted over `/workspace`, set
`REINSTALL_PROJECT=1` if the editable package must be reinstalled inside the
container.

```bash
REINSTALL_PROJECT=1 ./run_docker.sh python -c 'import jaxmarl; print(jaxmarl.__file__)'
```
