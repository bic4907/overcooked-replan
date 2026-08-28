# FCP transition-window observer pilot

This is the 3-seed FCP counterpart of the IPPO-RNN transition-window observer
experiment. It compares `none`, `agent_0`, `agent_1`, and `both` over the same
eight layouts.

FCP requires three sequential stages:

1. Train an observer-matched self-play population.
2. Train FCP best responses against that frozen population.
3. Evaluate seedwise FCP self-/cross-play within each observer arm.

## Scale

- Population: 8 layouts x 4 observer arms x 3 seeds = 96 runs.
- Each population run retains 10%, 50%, and final snapshots, yielding nine
  frozen partners per layout and observer arm.
- FCP training: 8 layouts x 4 observer arms x 3 seeds = 96 runs.
- Evaluation: 8 layouts x 4 observer arms = 32 runs; each evaluates the three
  FCP seeds as three SP and six ordered XP pairs.

## Create the sweeps

```bash
wandb sweep \
  --entity cilab-overcooked \
  --project overcooked-v3-fcp-transition-window-3seed_population \
  experiment/fcp_transition_window_observer/population_3seeds.yaml

wandb sweep \
  --entity cilab-overcooked \
  --project overcooked-v3-fcp-transition-window-3seed_train \
  experiment/fcp_transition_window_observer/train_3seeds.yaml

wandb sweep \
  --entity cilab-overcooked \
  --project overcooked-v3-fcp-transition-window-3seed_eval \
  experiment/fcp_transition_window_observer/eval_3seeds.yaml
```

## Run sequentially on one Pod

Population snapshots are local files, so population and FCP training must share
the same persistent workspace. Passing all three sweep references to the
sequential launcher guarantees that the best-response stage starts only after
the population is complete, and evaluation starts only after FCP training.

```bash
GPUS="0 1 2 3" bash experiment/run_agents_sequential.sh \
  cilab-overcooked/overcooked-v3-fcp-transition-window-3seed_population/POPULATION_SWEEP_ID \
  cilab-overcooked/overcooked-v3-fcp-transition-window-3seed_train/TRAIN_SWEEP_ID \
  cilab-overcooked/overcooked-v3-fcp-transition-window-3seed_eval/EVAL_SWEEP_ID
```

The population root is
`saves/fcp_transition_window_observer_3seed/population/<observer>`. Keeping one
directory per observer arm prevents FCP population discovery from mixing
policies trained with different warning visibility.
