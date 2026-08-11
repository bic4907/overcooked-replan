# Overcooked V3 Role-Scenario Sweep

Run all commands from the repository root with the project Python environment
activated. W&B credentials, team entity, and project are loaded from `.env`.

```dotenv
WANDB_API_KEY=your-api-key
WANDB_ENTITY=your-team-slug
WANDB_PROJECT=overcooked-v3-role-coordination
WANDB_MODE=online
```

## Validate

This checks the sweep config without creating anything in W&B.

```bash
python scripts/overcooked_v3/create_wandb_sweep.py \
  --config sweeps/overcooked_v3_role_scenarios.yaml \
  --dry-run
```

## Create Sweep

This creates the 20-run grid covering four role scenarios and five seeds. The
resulting `ENTITY/PROJECT/SWEEP_ID` is saved to `sweeps/.last_sweep_id`.

```bash
python scripts/overcooked_v3/create_wandb_sweep.py \
  --config sweeps/overcooked_v3_role_scenarios.yaml
```

## Run Sweep on Multiple GPUs

The runner reads `sweeps/.last_sweep_id` and starts one W&B agent per GPU.

```bash
GPUS="0 1 2 3" \
bash scripts/overcooked_v3/run_wandb_agents.sh
```

## Create and Run

Run this block to create a new sweep and immediately start four agents.

```bash
set -euo pipefail

python scripts/overcooked_v3/create_wandb_sweep.py \
  --config sweeps/overcooked_v3_role_scenarios.yaml

GPUS="0 1 2 3" \
bash scripts/overcooked_v3/run_wandb_agents.sh
```

## Resume or Run a Known Sweep

Resume the most recently created sweep:

```bash
GPUS="0 1 2 3" bash scripts/overcooked_v3/run_wandb_agents.sh
```

Run a sweep by its full W&B path:

```bash
GPUS="0 1 2 3" \
bash scripts/overcooked_v3/run_wandb_agents.sh \
  TEAM/PROJECT/SWEEP_ID
```

Final-episode recording is controlled by the sweep's Hydra parameter:

```yaml
recording:
  value: enabled  # Change to disabled to skip MP4 recording and upload.
```
