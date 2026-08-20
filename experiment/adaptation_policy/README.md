# Adaptation-policy experiments

## 1. Multi-layout self-play

The experiment uses the five original Overcooked-AI layouts:

- `cramped_room`
- `asymm_advantages`
- `coord_ring`
- `forced_coord`
- `counter_circuit`

All layouts are embedded at their original top-left coordinates in the same
width 9 x height 7 wall canvas. This preserves the playable geometry while
giving every CNN policy the same observation shape.

The training sweep contains five single-map specialists and all ten two-map
combinations. A specialist always resets to its one layout. A two-map policy
samples either constituent layout with probability 0.5 at episode reset and
keeps it fixed for that episode. There is no countdown, layout-change mask, or
signal-status channel.

With six seeds, the full training sweep has 90 runs:

```bash
wandb sweep \
  --entity cilab-overcooked \
  --project overcooked-v3-adaptation-policy \
  experiment/adaptation_policy/motive/multilayout_sp_train.yaml

wandb agent cilab-overcooked/overcooked-v3-adaptation-policy/<SWEEP_ID>
```

For a pilot, keep the two specialists and one multi-layout entry for a single
pair in `ENV_KWARGS.layout`.

After training finishes, launch the evaluation sweep:

```bash
wandb sweep \
  --entity cilab-overcooked \
  --project overcooked-v3-multilayout-sp-eval \
  experiment/adaptation_policy/motive/multilayout_sp_eval.yaml

wandb agent cilab-overcooked/overcooked-v3-multilayout-sp-eval/<SWEEP_ID>
```

The evaluation sweep has 45 matrix runs. Specialists are evaluated on all five
maps, while each multi-layout policy is evaluated on both constituent maps.
Within each evaluation matrix, identical checkpoints are SP and different
training seeds are XP. Each ordered checkpoint pair runs 50 deterministic
episodes.

A one-update multi-layout smoke test is:

```bash
JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false \
python -u baselines/IPPO/ippo_overcooked_v3.py \
  ENV_KWARGS.layout=multilayout_coord_ring__forced_coord \
  ENV_KWARGS.layout_mode=episode_random \
  ENV_KWARGS.include_transition_countdown=false \
  ENV_KWARGS.include_layout_change_mask=false \
  ENV_KWARGS.include_signal_status=false \
  NUM_ENVS=2 NUM_STEPS=2 NUM_MINIBATCHES=1 UPDATE_EPOCHS=1 \
  TOTAL_TIMESTEPS=4 REW_SHAPING_HORIZON=4 \
  recording=disabled upload_final_checkpoint=false wandb_mode=disabled
```

## 2. Switch-trained existing agent (no explicit co-player action)

The first switch-trained condition reuses the existing shared IPPO CNN. Each
two-layout environment cycles every 100 steps, so a 400-step episode follows
`A -> B -> A -> B`. At each boundary the new kitchen starts from its own agent
spawn positions with empty inventories, pots, counters, and floor objects. The
episode clock and policy state are not reset. The normal observation exposes
the visible map and the other player's position, orientation, and inventory,
but it does not append the co-player's action. Transition countdown,
layout-change mask, and signal status are disabled.

The sweep contains all ten original-layout pairs with six seeds, for 60 runs:

```bash
wandb sweep \
  --entity cilab-overcooked \
  --project overcooked-v3-switch-trained-no-coplayer-action_train \
  experiment/adaptation_policy/motive/switch_trained_no_coplayer_action.yaml

wandb agent cilab-overcooked/overcooked-v3-switch-trained-no-coplayer-action_train/<SWEEP_ID>
```

After all 60 training runs have final checkpoint artifacts, create the 10-run
seed-wise SP/XP evaluation sweep in its separate project:

```bash
wandb sweep \
  --entity cilab-overcooked \
  --project overcooked-v3-switch-trained-no-coplayer-action_eval \
  experiment/adaptation_policy/motive/switch_trained_no_coplayer_action_eval.yaml

wandb agent cilab-overcooked/overcooked-v3-switch-trained-no-coplayer-action_eval/<SWEEP_ID>
```

## 2b. Switch-trained agent with previous co-player action

This condition matches 2a except that both agents receive a six-way one-hot
encoding of the teammate's previous action. The action history is empty at an
episode reset and is cleared whenever a fresh layout starts.

```bash
wandb sweep \
  --entity cilab-overcooked \
  --project overcooked-v3-switch-trained-coplayer-action_train \
  experiment/adaptation_policy/motive/switch_trained_coplayer_action.yaml
```

After 2b training completes, create its separate evaluation sweep:

```bash
wandb sweep \
  --entity cilab-overcooked \
  --project overcooked-v3-switch-trained-coplayer-action_eval \
  experiment/adaptation_policy/motive/switch_trained_coplayer_action_eval.yaml
```

## Sequential multi-GPU sweep launcher

The launcher assigns one W&B agent per listed GPU, waits for the current grid
sweep to be exhausted, and then starts the next sweep. For example, this keeps
2a evaluation behind the 2a training completion barrier:

```bash
GPUS="0 1 2 3" bash \
  experiment/adaptation_policy/motive/run_agents_sequential.sh \
  cilab-overcooked/overcooked-v3-switch-trained-no-coplayer-action_train/<TRAIN_SWEEP_ID> \
  cilab-overcooked/overcooked-v3-switch-trained-no-coplayer-action_eval/<EVAL_SWEEP_ID>
```
