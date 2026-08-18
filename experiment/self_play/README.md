# Self-Play IPPO

## Train

```bash
wandb sweep --entity inchangbaek4907 --project overcooked-v3-role-coordination train.yaml
```

## Evaluate

```bash
wandb sweep --entity inchangbaek4907 --project overcooked-v3-crossplay eval.yaml
```

## Recipe Switch Train

```bash
wandb sweep --entity inchangbaek4907 --project overcooked-v3-role-coordination recipe_switch_train.yaml
```

## Recipe Switch Evaluate

```bash
wandb sweep --entity inchangbaek4907 --project overcooked-v3-crossplay recipe_switch_eval.yaml
```
