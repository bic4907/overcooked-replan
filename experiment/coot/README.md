# CooT W&B sweep commands

## HSP population

```bash
cd "$(git rev-parse --show-toplevel)"

uv run python baselines/CooT/preflight_sweep.py --static-only \
  experiment/coot/population.yaml
uv run wandb sweep --entity cilab-overcooked \
  --project overcooked-v3-coot-population \
  experiment/coot/population.yaml
GPUS=0,1 bash scripts/overcooked_v3/run_wandb_agents.sh \
  cilab-overcooked/overcooked-v3-coot-population/<SWEEP_ID>

uv run python baselines/CooT/preflight_sweep.py --static-only \
  experiment/coot/population_multi_recipe.yaml
uv run wandb sweep --entity cilab-overcooked \
  --project overcooked-v3-coot-population \
  experiment/coot/population_multi_recipe.yaml
GPUS=0,1 bash scripts/overcooked_v3/run_wandb_agents.sh \
  cilab-overcooked/overcooked-v3-coot-population/<SWEEP_ID>
```

## Candidate responses

```bash
uv run python baselines/CooT/preflight_sweep.py \
  experiment/coot/response_candidates.yaml
uv run wandb sweep --entity cilab-overcooked \
  --project overcooked-v3-coot-response-candidates \
  experiment/coot/response_candidates.yaml
GPUS=0,1 bash scripts/overcooked_v3/run_wandb_agents.sh \
  cilab-overcooked/overcooked-v3-coot-response-candidates/<SWEEP_ID>

uv run python baselines/CooT/preflight_sweep.py \
  experiment/coot/response_candidates_multi_recipe.yaml
uv run wandb sweep --entity cilab-overcooked \
  --project overcooked-v3-coot-response-candidates \
  experiment/coot/response_candidates_multi_recipe.yaml
GPUS=0,1 bash scripts/overcooked_v3/run_wandb_agents.sh \
  cilab-overcooked/overcooked-v3-coot-response-candidates/<SWEEP_ID>
```

## Selected population responses

```bash
uv run python baselines/CooT/preflight_sweep.py \
  experiment/coot/response.yaml
uv run wandb sweep --entity cilab-overcooked \
  --project overcooked-v3-coot-response \
  experiment/coot/response.yaml
GPUS=0,1 bash scripts/overcooked_v3/run_wandb_agents.sh \
  cilab-overcooked/overcooked-v3-coot-response/<SWEEP_ID>
```

## HSP-only responses

```bash
uv run python baselines/CooT/preflight_sweep.py \
  experiment/coot/response_hsp_only.yaml
uv run wandb sweep --entity cilab-overcooked \
  --project overcooked-v3-coot-response \
  experiment/coot/response_hsp_only.yaml
GPUS=0,1 bash scripts/overcooked_v3/run_wandb_agents.sh \
  cilab-overcooked/overcooked-v3-coot-response/<SWEEP_ID>
```

## CooT training

```bash
uv run python baselines/CooT/preflight_sweep.py \
  experiment/coot/train.yaml
uv run wandb sweep --entity cilab-overcooked \
  --project overcooked-v3-coot-train \
  experiment/coot/train.yaml
GPUS=0,1 bash scripts/overcooked_v3/run_wandb_agents.sh \
  cilab-overcooked/overcooked-v3-coot-train/<SWEEP_ID>
```

## Seed-wise SP/XP evaluation

```bash
uv run python baselines/CooT/preflight_sweep.py \
  experiment/coot/eval.yaml
uv run wandb sweep --entity cilab-overcooked \
  --project overcooked-v3-coot-eval \
  experiment/coot/eval.yaml
GPUS=0,1 bash scripts/overcooked_v3/run_wandb_agents.sh \
  cilab-overcooked/overcooked-v3-coot-eval/<SWEEP_ID>
```
