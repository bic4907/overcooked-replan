# Self-Play IPPO-RNN

The training sweep covers all 12 selected role-scenario layouts and six seeds
in the dedicated `overcooked-v3-ippo-rnn_train` W&B project.

## Train (12 layouts)

```bash
wandb sweep \
  --entity cilab-overcooked \
  --project overcooked-v3-ippo-rnn_train \
  experiment/self_play_rnn/train.yaml
```

## Evaluate (12 layouts, 450 steps)

Run this sweep after the training sweep has finished and uploaded its final
checkpoint artifacts.

```bash
wandb sweep \
  --entity cilab-overcooked \
  --project overcooked-v3-ippo-rnn_eval \
  experiment/self_play_rnn/eval.yaml
```
