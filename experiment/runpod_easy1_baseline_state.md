# Runpod `_1` baseline state

Created: 2026-09-08 (Asia/Seoul)

## Pods

| Role | Pod ID | GPU | Count | Cost/hour |
| --- | --- | --- | ---: | ---: |
| FCP | `8rpv077f11ol4x` | PRO 6000 MIG 24GB | 8 | $4.72 |
| IPPO + IPPO-RNN | `aa6v2u0iyhwbtl` | PRO 6000 MIG 24GB | 8 | $4.72 |

Both original pods use `bic4907/overcooked:cu13-rp` and a 100 GB pod volume
mounted at `/workspace`. They were stopped with desired status `EXITED` at
16:01 UTC after the host interruption, preserving their volumes and preventing
unexpected reactivation.

| Recovery role | Pod ID | Datacenter | GPU | Count | Cost/hour |
| --- | --- | --- | --- | ---: | ---: |
| IPPO-RNN | `4y95cc7af0hb03` | EUR-IS-2 | PRO 6000 MIG 24GB | 8 | $4.72 |
| FCP | `6dekvjdn7y1nul` | US-NE-1 | PRO 6000 MIG 24GB | 8 | $4.72 |

The recovery pods were created at 16:02 UTC in separate, non-WA datacenters to
avoid a shared-host failure. Both initially remained in
`initializing/awaiting_container`; do not create further pods while these two
are provisioning. Once ready, validate eight JAX CUDA devices before launching
replacement sweeps. Delete all four pods after verified completion; stopped pod
volumes can still incur storage charges.

Runpod's proxied SSH suffix is pod-specific, not account-wide. The FCP
recovery pod's Connect-tab command is:

```text
ssh 6dekvjdn7y1nul-644122c4@ssh.runpod.io -i ~/.ssh/id_ed25519
```

Read each replacement pod's exact command from the Runpod Connect tab; do not
reuse the original pods' `64411dd2` suffix and do not use direct TCP SSH.

### 2026-09-08 host interruption

At approximately 15:57 UTC, both pods simultaneously changed from `running` to
`initializing` with reason `awaiting_container`. Gateway SSH became unavailable
and W&B stopped advancing. Both pods were exposed through the same public host,
so this is being treated as a shared Runpod host/control-plane interruption, not
as two independent training failures. The last observed authoritative progress
was:

- IPPO-RNN `mglz038p`: 8 finished, 8 running, 16 of 18 assignments started;
  active `_step` values ranged from 134 to 334.
- FCP best response `ayst2t4g`: 8 running; active `_step` values ranged from
  157 to 290.

Restart attempts returned Runpod internal `server_error`/deadline-exceeded
responses. Do not delete either pod or launch duplicate W&B agents while the
old processes cannot be inspected. W&B subsequently marked all eight active
IPPO-RNN runs and all eight active FCP runs `crashed`, which proves the old
processes are terminal. Preserve every finished artifact-bearing run and create
a versioned replacement sweep for only the interrupted/missing configurations.

### FCP population recovery artifact

The original population sweep did not log its checkpoint files to W&B. The
original FCP pod was briefly resumed, and the preserved volume was revalidated
as exactly 27 checkpoints. They were then uploaded to:

```text
cilab-overcooked/overcooked-v3-fcp-easy1-population-recovery/
fcp-easy1-population-checkpoints:latest
```

Recovery run: `uhqx4d03`. The W&B API independently reports 27 `.safetensors`
files and metadata `source_sweep=46qfe84z`, `checkpoint_count=27`. Download the
artifact's `fcp_population/` subtree to
`saves/fcp_easy1/fcp_population/`, rerun the verifier, and only then launch FCP
best-response training. The original FCP pod was stopped again after upload.

## W&B sweeps

| Stage | Project | Sweep ID | Runs |
| --- | --- | --- | ---: |
| IPPO train | `overcooked-v3-ippo-easy1_train` | `z7rfiie5` | 18 |
| IPPO eval | `overcooked-v3-ippo-easy1_eval` | `j9y1n577` | 3 |
| IPPO-RNN train | `overcooked-v3-ippo-rnn-easy1_train` | `mglz038p` | 18 |
| IPPO-RNN eval | `overcooked-v3-ippo-rnn-easy1_eval` | `7wgsgrh5` | 3 |
| FCP population | `overcooked-v3-fcp-easy1_population` | `46qfe84z` | 9 |
| FCP best response | `overcooked-v3-fcp-easy1_train` | `ayst2t4g` | 18 |
| FCP eval | `overcooked-v3-fcp-easy1_eval` | `040dx6ha` | 3 |

The shared-host interruption left no finished runs in the original FCP
best-response sweep. Its authoritative full replacement is:

| Recovery stage | Project | Sweep ID | Runs |
| --- | --- | --- | ---: |
| FCP best response | `overcooked-v3-fcp-easy1-recovery1_train` | `iq5qa8n0` | 18 |
| FCP eval | `overcooked-v3-fcp-easy1-recovery1_eval` | `6wtspgb1` | 3 |

Launch both sequentially with `scripts/runpod_easy1_baselines.sh fcp-recovery1`
after the recovered population tree passes validation.

The original IPPO-RNN sweep retained eight finished artifact-bearing runs. Its
recovery sweep reruns only these ten interrupted or unassigned configurations:
`split_1` seeds 3-5, `outage_1` seeds 3-5, and `distance_switch_1` seeds 2-5.

| Recovery stage | Project | Sweep ID | Runs |
| --- | --- | --- | ---: |
| IPPO-RNN missing train jobs | `overcooked-v3-ippo-rnn-easy1_train` | `zegiopgh` | 10 |
| IPPO-RNN combined eval | `overcooked-v3-ippo-rnn-easy1-recovery1_eval` | `wvet20aa` | 3 |

Launch the recovery training, combined RNN evaluation, and the already-pending
IPPO evaluation sequentially with
`scripts/runpod_easy1_baselines.sh rnn-recovery1`.

The FCP recovery launcher started at 16:20 UTC on pod `6dekvjdn7y1nul`. All
eight W&B agents connected online after validating the 27 population
checkpoints and eight JAX CUDA devices.

Execution order is encoded in `scripts/runpod_easy1_baselines.sh`. Remove both
pods after the evaluation artifacts and W&B summaries have been verified.

The initial sweeps `ranyyq1p` and `zoaxacrm` were cancelled after eight runs in
each were detected logging offline because `WANDB_API_KEY` was not exported.
Their outputs are excluded; the replacement IDs above are the authoritative runs.

The first downstream FCP sweeps (`33zac3rf`, `2vep0tev`) were cancelled before
they started. The launcher validates all 27 population snapshots before starting
the replacement best-response and evaluation sweeps.

The subsequent best-response sweep `74otspw7` was cancelled after one assignment
started without the Runpod launcher environment (one GPU and offline W&B). That
run is excluded. `ayst2t4g` is the clean replacement started through the launcher.
When the population is already complete, use the launcher's `fcp-post` role so
W&B agents do not re-query the closed population sweep before validation.

## Outage candidate v2 rerun

The completed baseline screen showed that `split_1` and `distance_switch_1`
have clear SP-XP gaps for IPPO and IPPO-RNN, while `outage_1` v1 does not:

| Algorithm | Layout | SP | XP | SP-XP |
| --- | --- | ---: | ---: | ---: |
| IPPO | `split_1` | 166.67 | 33.33 | 133.33 |
| IPPO | `outage_1` v1 | 396.67 | 398.67 | -2.00 |
| IPPO | `distance_switch_1` | 393.33 | 163.33 | 230.00 |
| IPPO-RNN | `split_1` | 190.00 | 78.00 | 112.00 |
| IPPO-RNN | `outage_1` v1 | 400.00 | 400.00 | 0.00 |
| IPPO-RNN | `distance_switch_1` | 496.67 | 310.67 | 186.00 |
| FCP | `split_1` | 70.00 | 102.67 | -32.67 |
| FCP | `outage_1` v1 | 340.00 | 338.67 | 1.33 |
| FCP | `distance_switch_1` | 356.67 | 380.67 | -24.00 |

FCP is inverted on two maps that pass strongly for both self-play algorithms.
That systematic result is retained for reporting, but it is not being used to
redesign otherwise valid maps because an FCP best response is trained against
a policy population rather than its own clone; its diagonal `SP` has a
different interpretation from IPPO self-play.

Only `outage_1` was revised. `outage-1-adjacent-relay-v2` keeps the 5x7 outage
mechanic and uses two adjacent handoffs, a one-move onion relay, changed
resource positions, and mirrored lower notches. The fresh versioned sweeps are:

| Stage | Project | Sweep ID | Runs |
| --- | --- | --- | ---: |
| IPPO train | `overcooked-v3-ippo-outage1-v2_train` | `8ixm9jvx` | 6 |
| IPPO eval | `overcooked-v3-ippo-outage1-v2_eval` | `2laq6eg0` | 1 |
| IPPO-RNN train | `overcooked-v3-ippo-rnn-outage1-v2_train` | `nlxajili` | 6 |
| IPPO-RNN eval | `overcooked-v3-ippo-rnn-outage1-v2_eval` | `9k8ezs3b` | 1 |
| FCP population | `overcooked-v3-fcp-outage1-v2_population` | `xfkub379` | 3 |
| FCP best response | `overcooked-v3-fcp-outage1-v2_train` | `hqovh4vz` | 6 |
| FCP eval | `overcooked-v3-fcp-outage1-v2_eval` | `zzc8mi9z` | 1 |

The v2 IPPO rerun completed with SP 406.67, XP 377.33, and a 29.33 gap
(7.2%), passing the absolute 10 and relative 5% screening thresholds.
`outage1-v2-ippo-rnn` runs on pod `aa6v2u0iyhwbtl`; `outage1-v2-fcp` runs on
pod `6dekvjdn7y1nul`.
