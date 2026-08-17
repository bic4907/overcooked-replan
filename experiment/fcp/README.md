# Fictitious Co-Play

FCP is a two-stage baseline. First, train a population of self-play IPPO
policies and retain snapshots from different training stages. Then train an FCP
best response against frozen partners sampled from that population. Run the
cross-play evaluation only after both training stages have finished.

All commands below assume the repository environment is installed and the
project `.env` contains `WANDB_API_KEY` and, optionally,
`WANDB_ENTITY=inchangbaek4907`.

FCP naming is kept separate from ordinary IPPO runs:

- SP population files: `saves/fcp_population/`
- FCP best-response folders: `saves/<layout>_<arch>_fcp_seed<seed>/`
- FCP checkpoints: `fcp_<arch>_...safetensors`
- W&B population runs/groups: `fcp-population-...`
- W&B FCP runs/groups/artifacts: `fcp...`
- W&B sweep names: `fcp-population`, `fcp-best-response`, and `fcp-crossplay`

## Quick start: run everything with W&B

Run the following stages in order. Do not start the next stage until every run
in the current sweep has finished.

### Stage 1 — create and run the population sweep

Create it on the Mac:

```bash
wandb sweep --entity inchangbaek4907 --project overcooked-v3-fcp-population experiment/fcp/train_population.yaml
```

Wait until all population runs have finished and the shared
`saves/fcp_population/` directory contains their checkpoints.

### Stage 2 — create and run the FCP sweep

Create it on the Mac:

```bash
wandb sweep --entity inchangbaek4907 --project overcooked-v3-role-coordination experiment/fcp/train.yaml
```

Wait until all FCP runs and final checkpoint artifact uploads have finished.

### Stage 3 — create and run the cross-play sweep

Create it on the Mac:

```bash
wandb sweep --entity inchangbaek4907 --project overcooked-v3-crossplay experiment/fcp/eval.yaml
```

The final matrices and pairwise scores appear in the
`overcooked-v3-crossplay` W&B project.
