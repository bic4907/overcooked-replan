# Observer experiment sweeps

This suite reruns the observer ablation from scratch on the six selected layouts:

- `split_0`, `split_1`
- `outage_0`, `outage_1`
- `distance_switch_0`, `distance_switch_1`

Every layout uses the four transition-signal visibility arms `none`, `agent_0`,
`agent_1`, and `both`. The new W&B projects and checkpoint roots are isolated
from the historical observer runs, whose `_1` labels predate the current map
revisions.

## Run inventory

| Suite | Stage | Runs |
| --- | --- | ---: |
| IPPO-RNN | train | 144 |
| IPPO-RNN | eval | 24 |
| FCP | population | 72 |
| FCP | best response | 144 |
| FCP | eval | 24 |
| **Total** |  | **408** |

FCP population and best-response agents must share the same persistent
filesystem. The population stage writes nine frozen partners per
layout/observer arm under `saves/fcp_observer/population`.

## W&B projects

- `overcooked-v3-ippo-rnn-observer_train`
- `overcooked-v3-ippo-rnn-observer_eval`
- `overcooked-v3-fcp-observer_population`
- `overcooked-v3-fcp-observer_train`
- `overcooked-v3-fcp-observer_eval`

## Dependency order

The two chains are independent:

1. IPPO-RNN train -> IPPO-RNN eval
2. FCP population -> FCP best response -> FCP eval

Create the first-stage sweeps with:

```bash
wandb sweep --entity cilab-overcooked \
  --project overcooked-v3-ippo-rnn-observer_train \
  experiment/transition_window_observer/train_all.yaml

wandb sweep --entity cilab-overcooked \
  --project overcooked-v3-fcp-observer_population \
  experiment/fcp_transition_window_observer/population_all.yaml
```

Create each downstream sweep only after its required checkpoints are complete.

## Created first-stage sweeps

Created on 2026-09-09. No agents were attached at creation time.

| Stage | Sweep ID | Expected runs |
| --- | --- | ---: |
| IPPO-RNN train | `ckt0sirk` | 144 |
| FCP population | `8t665wf5` | 72 |

Agent paths:

```text
cilab-overcooked/overcooked-v3-ippo-rnn-observer_train/ckt0sirk
cilab-overcooked/overcooked-v3-fcp-observer_population/8t665wf5
```
