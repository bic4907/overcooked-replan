# Fictitious Co-Play

## Population

```bash
wandb sweep --entity cilab-overcooked --project overcooked-v3-fcp-population population.yaml
```

## Train

```bash
wandb sweep --entity cilab-overcooked --project overcooked-v3-role-coordination train.yaml
```

## Evaluate

```bash
wandb sweep --entity cilab-overcooked --project overcooked-v3-crossplay eval.yaml
```
