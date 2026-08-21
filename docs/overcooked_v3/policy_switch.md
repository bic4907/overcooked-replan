# Overcooked V3 Policy-Switch Baseline

This baseline works with every registered Overcooked V3 dynamic layout. It
creates one policy head for each unique phase, where a phase is identified by
its map tiles, agent start positions, and active recipe. All heads are jointly
trained on the full dynamic episode. An internal phase marker routes each
transition through the active head, and inactive heads receive no gradient for
that transition. This exposes a newly activated policy to the real inherited
inventory, pot contents, loose objects, and agent positions instead of a clean
static reset. Repeated phases reuse the same policy, so an `A -> B -> A` layout
stores two policies rather than three.

At evaluation time, `state.layout_index` selects the phase policy immediately
after every environment transition. Recurrent hidden state is reset whenever
the selected policy changes.

## Train

```bash
python baselines/PolicySwitch/train_overcooked_v3.py scenario=split_0
```

`TOTAL_TIMESTEPS` is the shared full-episode training budget. Each final
`*_vmapN.safetensors` keeps the existing checkpoint format with contiguous
top-level trees named `policy_0`, `policy_1`, and so on. When
`upload_final_checkpoint=true`, the combined file and resolved Hydra config are
uploaded together as the W&B `final` checkpoint artifact.

## Evaluate a local checkpoint

```bash
python baselines/PolicySwitch/eval_overcooked_v3.py --checkpoints saves/EXPERIMENT/CHECKPOINT.safetensors --layout split_0 --episodes 3 --wandb-mode disabled
```

Pass two checkpoint paths to run cross-play. Passing one path uses the same
combined checkpoint for both agents.

## Evaluate W&B runs

```bash
python baselines/PolicySwitch/eval_overcooked_v3.py --run-ids RUN_A RUN_B --entity cilab-overcooked --source-project overcooked-v3-policyswitch_train --project overcooked-v3-policyswitch_eval --layout split_0 --episodes 3 --wandb-mode online
```
