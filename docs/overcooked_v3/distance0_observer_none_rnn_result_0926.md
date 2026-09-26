# Canonical distance_0: IPPO-RNN observer-none replacement

The replacement used `dual-handoff-recipe-priority-11x7-v1`, 30M training steps
per seed, `transition_observer=none`, and fresh seeds 0–9 split across HPC
(0–3) and aica (4–9). Evaluation selected the ten new finished checkpoint
artifacts by exact run ID, revision, and observer configuration.

| Evaluation | Ordered pairs | Episodes per pair | SP | XP | SP−XP |
| --- | ---: | ---: | ---: | ---: | ---: |
| [0921 test](https://wandb.ai/cilab-overcooked/overcooked-v3-ippo-rnn-observer-none-0921_test/runs/2dl33vzo) | 6×6 | 20 | 160.00 | 134.67 | +25.33 |
| [0921 test10](https://wandb.ai/cilab-overcooked/overcooked-v3-ippo-rnn-observer-none-0921_test10/runs/ytkawt4g) | 10×10 | 20 | 154.00 | 132.89 | +21.11 |

The original 13×6 observer-none `distance_0` was fully backed up before
cleanup: 12 historical W&B runs, 216 files, 372 MB, with verified local,
HPC, and Weka copies. At final audit, the ten old training run IDs were
already absent from W&B; no deletion was attempted for them. The two old
evaluation run IDs were still present, matched the archived legacy
provenance, and were selectively deleted without deleting artifacts. All
12 historical IDs are now absent, while the new ten training runs and two
evaluation runs remain. The [receipt](../../artifacts/wandb/distance0_rnn_observer_none_legacy_0921_deletion_receipt_20260926.jsonl)
records which runs were already absent and which two were deleted.
