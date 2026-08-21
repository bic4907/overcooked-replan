# PolicySwitch-IPPO

This baseline trains one IPPO policy head for every unique phase of an
Overcooked V3 dynamic layout. The heads are jointly trained on the full dynamic
episode, so each one learns from the inventory, pots, objects, and agent
positions it actually inherits at a transition. Only the active head receives
the transition's gradient. Repeated phases reuse one head. Evaluation selects
`policy_0..N` from the current `state.layout_index`.

```bash
python train_overcooked_v3.py scenario=split_0
```

```bash
python eval_overcooked_v3.py --checkpoints CHECKPOINT.safetensors --layout split_0 --episodes 3 --wandb-mode disabled
```

See [the full workflow](../../docs/overcooked_v3/policy_switch.md).
