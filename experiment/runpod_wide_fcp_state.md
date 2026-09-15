# Wide-map FCP Runpod experiment

Started 2026-09-10 at 18:09 KST on Pod `apik5541lhov7y` (`fcp-wide-final`).
Runtime source commit: `b00490869f1f5769da29986f42ff89fc1f237d73`.
The source archive was exported from that commit, with runtime directories only.
Its SHA-256 was checked on the Pod before extraction:
`596e53ed5927e4dc44ee8bbb3ce2c0176703d597772348a4c2f58ff34c1e7fbc`.

| Layout | Size (width×height) | Revision |
| --- | --- | --- |
| `split_wide` | 13×7 | `split-wide-v3` |
| `outage_wide` | 13×6 | `outage-wide-v3` |
| `distance_switch_wide` | 13×6 | `distance-switch-wide-v2` |

One EU-SE-1 Pod runs 6 × A40 48 GB GPUs using the existing
`bic4907/overcooked:cu13-rp` template (`pgrqqi2fkl`). GPU cost at launch:
$2.94/hour plus storage. It has a 50 GB volume at `/workspace`.
The image provides JAX 0.5.3 and Flax 0.10.5. The launcher enables
`JAX_PLATFORMS=cuda,cpu` so GPU training can use CPU logging callbacks. OpenSSH was installed through the
Runpod SSH proxy to enable direct source transfer; no training dependencies
were changed.

The pipeline executes FCP only, including its required IPPO-RNN population:

1. Population: 3 seeds per map, 30M steps each, snapshots at 10%, 50%, 100%.
2. Validate and upload all 27 population checkpoints.
3. FCP: 6 seeds per map, 30M steps each, final checkpoint uploads.
4. Cross-play: 36 ordered pairs per map, 20 episodes each, 450 steps per episode.
5. Upload all checkpoints, evaluation output, and adaptation traces; terminate Pod.

| Stage | W&B sweep |
| --- | --- |
| Population (9 runs) | [vctouw3e](https://wandb.ai/cilab-overcooked/overcooked-v3-fcp-population/sweeps/vctouw3e) |
| FCP training (18 runs) | [wdeiwyhn](https://wandb.ai/cilab-overcooked/overcooked-v3-fcp_train/sweeps/wdeiwyhn) |

Evaluation runs are written to `cilab-overcooked/overcooked-v3-fcp_eval`, with
run label `FCP-wide`. Training config records layout revision and source commit
in `NOTES`. Runs carry `wide-final-20260910`.

Remote source: `/workspace/fcp-wide`.
Progress: `pipeline_status.json`; launcher PID: `pipeline.pid`; log: `pipeline.log`.
Checkpoints: `saves/fcp_wide`; results: `evaluation/fcp_wide`.
The launcher is detached from SSH and has an eight-hour deadline. Failure or
timeout stops the Pod and preserves its volume; a stopped volume still bills.
Runpod control credentials are in `/tmp`, outside all uploaded directories.

The initial sweeps `cfj0xq7p` and `nl27big5` were cancelled after a metadata
override parsing error, before any training update. They remain available for
diagnosis. The next population sweep `uy5c4mpr` failed because CUDA-only
backend selection excluded the CPU required by JAX logging callbacks; its
unused training sweep `bfq5tc4y` was cancelled. Commit `b004908` fixes that
backend configuration. Replacement sweep references and current launch metadata are
recorded in [runpod_wide_state.json](fcp/runpod_wide_state.json).

Local tests were not requested and were not run. Map phase renders were inspected
before committing; actual GPU training progress is recorded in W&B.

At 18:11 KST, all six active population runs completed update 1
(65,536 environment steps each), with 100% utilization on all six GPUs.
These are seeds 0 and 1 for each map; seed 2 runs follow as GPUs become free.
The pipeline PID is `16293`.
