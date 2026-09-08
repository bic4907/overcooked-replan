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
