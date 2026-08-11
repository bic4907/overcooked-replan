```bash
wandb sweep --entity inchangbaek4907 --project overcooked-v3-role-coordination experiment/sweeps/overcooked_v3_role_scenarios.yaml
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
GPUS="0 1 2 3" bash scripts/overcooked_v3/run_wandb_agents.sh \
inchangbaek4907/overcooked-v3-role-coordination/SWEEP_ID
```
