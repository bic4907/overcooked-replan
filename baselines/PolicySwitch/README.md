# PolicySwitch-IPPO

This baseline trains one independent IPPO policy for every unique phase of an
Overcooked V3 dynamic layout. A phase-specific training environment keeps the
original tiles, recipe, and agent starts but disables transitions. Repeated
phases reuse one policy. Evaluation selects `policy_0..N` from the current
`state.layout_index`.

```bash
python train_overcooked_v3.py scenario=splitnosig_0
```

```bash
python eval_overcooked_v3.py --checkpoints CHECKPOINT.safetensors --layout splitnosig_0 --episodes 3 --wandb-mode disabled
```

See [the full workflow](../../docs/overcooked_v3/policy_switch.md).
