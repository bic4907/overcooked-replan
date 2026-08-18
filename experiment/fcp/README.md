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

The evaluation command reads FCP checkpoint artifacts from
`inchangbaek4907/overcooked-v3-role-coordination` and writes evaluation runs to
the project selected by the sweep command.

```bash
wandb sweep --entity cilab-overcooked --project overcooked-v3-crossplay eval.yaml
```
