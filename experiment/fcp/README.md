# Fictitious Co-Play

FCP is a two-stage baseline. First, train a population of self-play IPPO
policies and retain snapshots from different training stages. Then train an FCP
best response against frozen partners sampled from that population. Run the
seed-wise FCP cross-play evaluation only after both training stages have
finished. IPPO and FCP are evaluated in separate sweeps rather than compared
inside one payoff matrix.

All commands below assume the repository environment is installed and the
project `.env` contains `WANDB_API_KEY` and, optionally,
`WANDB_ENTITY=inchangbaek4907`. Set `WANDB_SOURCE_PROJECT` to the training
project queried by cross-play evaluation. The evaluation output project comes
from the `wandb sweep --project ...` command.

FCP naming is kept separate from ordinary IPPO runs:

- SP population files: `saves/fcp_population/`
- FCP best-response folders: `saves/<layout>_<arch>_fcp_seed<seed>/`
- FCP checkpoints: `fcp_<arch>_...safetensors`
- W&B self-play population runs/groups: `fcp-self-play-...`
- W&B FCP runs/groups/artifacts: `fcp...`
- W&B sweep names: `fcp-self-play`, `fcp-best-response`, and `fcp-seedwise-xp`

## Quick start: run everything with W&B

Run the following stages in order. Do not start the next stage until every run
in the current sweep has finished.

### Stage 1 — create and run the self-play population sweep

Create it on the Mac:

```bash
wandb sweep --entity inchangbaek4907 --project overcooked-v3-fcp-population experiment/fcp/population.yaml
```

Wait until all population runs have finished and the shared
`saves/fcp_population/` directory contains their checkpoints.

### Stage 2 — create and run the FCP sweep

Create it on the Mac:

```bash
wandb sweep --entity inchangbaek4907 --project overcooked-v3-role-coordination experiment/fcp/train.yaml
```

Wait until all FCP runs and final checkpoint artifact uploads have finished.

### Stage 3 — create and run the FCP seed-wise XP sweep

Create it on the Mac:

```bash
wandb sweep --entity inchangbaek4907 --project overcooked-v3-crossplay experiment/fcp/eval.yaml
```

The resulting matrix contains only ordered FCP seed pairs. The self-play
evaluation sweep at `experiment/self_play/eval.yaml` separately contains only
ordered IPPO seed pairs.
