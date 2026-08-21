# Self-Play IPPO

The combined sweeps cover all 12 selected role-scenario layouts. Training and
evaluation use separate projects under the `cilab-overcooked` entity.

## Train (12 layouts)

```bash
wandb sweep --entity cilab-overcooked --project overcooked-v3-ippo_train train.yaml
```

## Evaluate (12 layouts, 450 steps)

```bash
wandb sweep --entity cilab-overcooked --project overcooked-v3-ippo_eval eval.yaml
```

## Recipe Switch Train

```bash
wandb sweep --entity cilab-overcooked --project overcooked-v3-ippo_train recipe_switch_train.yaml
```

## Recipe Switch Evaluate

```bash
wandb sweep --entity cilab-overcooked --project overcooked-v3-ippo_eval recipe_switch_eval.yaml
```

## Distance Switch Train

```bash
wandb sweep --entity cilab-overcooked --project overcooked-v3-ippo_train distance_switch_train.yaml
```

## Distance Switch Evaluate

```bash
wandb sweep --entity cilab-overcooked --project overcooked-v3-ippo_eval distance_switch_eval.yaml
```
