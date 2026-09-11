# `_1` layout sweep suite

This suite runs only `split_1`, `outage_1`, and `distance_switch_1`.
The existing `_0` sweep files are unchanged.

## Run counts

| Suite | Train/population/BR | Evaluation | Total |
| --- | ---: | ---: | ---: |
| IPPO baseline | 18 | 3 | 21 |
| IPPO-RNN baseline | 18 | 3 | 21 |
| FCP baseline | 36 | 3 | 39 |
| IPPO-RNN observer | 72 | 12 | 84 |
| FCP observer | 144 | 12 | 156 |
| **Grand total** | **288** | **33** | **321** |

FCP uses six population seeds (`0..5`) and six best-response seeds (`0..5`).
Each population run saves checkpoints at 10%, 50%, and 100%, giving eighteen
frozen partners per layout (and per observer arm in the observer experiment).

## Sweep creation order

The baseline IPPO and IPPO-RNN suites are independent. For each FCP suite,
finish the population stage before starting best-response training, and keep
both stages on the same persistent filesystem.

```bash
# Baselines
wandb sweep experiment/self_play/train_easy1.yaml
wandb sweep experiment/self_play_rnn/train_easy1.yaml

wandb sweep experiment/fcp/population_easy1.yaml
# after population completion
wandb sweep experiment/fcp/train_easy1.yaml

# Observer experiments
wandb sweep experiment/transition_window_observer/train_easy1.yaml

wandb sweep experiment/fcp_transition_window_observer/population_easy1.yaml
# after population completion
wandb sweep experiment/fcp_transition_window_observer/train_easy1.yaml
```

Create the corresponding `eval_easy1.yaml` sweeps only after their training
projects have finished and uploaded final checkpoint artifacts.
