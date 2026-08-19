# Overcooked V3 Policy-Switch Baseline

This baseline works with every registered Overcooked V3 dynamic layout. It
creates one transition-free training layout for each unique phase, where a
phase is identified by its map tiles, agent start positions, and active recipe.
Each phase policy is trained independently with the existing IPPO
implementation. Repeated phases reuse the same policy, so an `A -> B -> A`
layout stores two policies rather than three.

At evaluation time, `state.layout_index` selects the phase policy immediately
after every environment transition. Recurrent hidden state is reset whenever
the selected policy changes.

## Train

```bash
python baselines/PolicySwitch/train_overcooked_v3.py scenario=splitnosig_0
```

`TOTAL_TIMESTEPS` is applied separately to every unique phase policy. Each
final `*_vmapN.safetensors` contains contiguous top-level trees named
`policy_0`, `policy_1`, and so on. When `upload_final_checkpoint=true`, the
combined file and resolved Hydra config are uploaded together as the W&B
`final` checkpoint artifact.

## Evaluate a local checkpoint

```bash
python baselines/PolicySwitch/eval_overcooked_v3.py --checkpoints saves/EXPERIMENT/CHECKPOINT.safetensors --layout splitnosig_0 --episodes 3 --wandb-mode disabled
```

Pass two checkpoint paths to run cross-play. Passing one path uses the same
combined checkpoint for both agents.

## Evaluate W&B runs

```bash
python baselines/PolicySwitch/eval_overcooked_v3.py --run-ids RUN_A RUN_B --entity inchangbaek4907 --source-project overcooked-v3-role-coordination --project overcooked-v3-crossplay --layout splitnosig_0 --episodes 3 --wandb-mode online
```
