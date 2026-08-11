# Overcooked V3 Role-Scenario Sweep

The sweep definition is stored in
`experiment/sweeps/overcooked_v3_role_scenarios.yaml`. Create the sweep on a
Mac, then run its agents on the GPU server.

## 1. Create the Sweep on macOS

Install the project, authenticate W&B once, and create the sweep under the
`inchangbaek4907` entity.

```bash
python -m pip install -e ".[algs]"
wandb login
```

```bash
wandb sweep \
  --entity inchangbaek4907 \
  --project overcooked-v3-role-coordination \
  experiment/sweeps/overcooked_v3_role_scenarios.yaml
```

W&B prints an agent command containing the full sweep path:

```text
wandb agent inchangbaek4907/overcooked-v3-role-coordination/SWEEP_ID
```

Copy `inchangbaek4907/overcooked-v3-role-coordination/SWEEP_ID` to the GPU
server. No generated sweep-ID file needs to be committed or transferred.

## 2. Run the Sweep on the GPU Server

Start one W&B agent per GPU with the full sweep path copied from the Mac:

```bash
GPUS="0 1 2 3" \
bash scripts/overcooked_v3/run_wandb_agents.sh \
  inchangbaek4907/overcooked-v3-role-coordination/SWEEP_ID
```

Comma-separated GPU IDs are also accepted:

```bash
GPUS="0,1" \
bash scripts/overcooked_v3/run_wandb_agents.sh \
  inchangbaek4907/overcooked-v3-role-coordination/SWEEP_ID
```

Run the same command again to process any remaining runs after an interruption.

## 3. Run Existing Sweeps Sequentially

When multiple sweep paths are provided, all agents finish the first sweep
before moving to the next one.

```bash
GPUS="0 1 2 3" \
bash scripts/overcooked_v3/run_wandb_agents.sh \
  inchangbaek4907/overcooked-v3-role-coordination/SWEEP_ID_A \
  inchangbaek4907/overcooked-v3-role-coordination/SWEEP_ID_B
```

## Sweep Contents

| Setting | Value |
| --- | --- |
| Scenarios | `split_no_sig`, `split_sig`, `outage_no_sig`, `outage_sig` |
| Seeds | `0`, `1`, `2`, `3`, `4` |
| Total runs | 20 |
| Policy | IPPO CNN |
| Objective | Maximize `train/episode_return` |
| Final video | `recording=enabled` |

To disable final-episode MP4 recording for the entire sweep, change the YAML
parameter before running `wandb sweep`:

```yaml
recording:
  value: disabled
```
