# Runpod `_1` baseline state

Created: 2026-09-08 (Asia/Seoul)

## Pods

| Role | Pod ID | GPU | Count | Cost/hour |
| --- | --- | --- | ---: | ---: |
| FCP | `8rpv077f11ol4x` | PRO 6000 MIG 24GB | 8 | $4.72 |
| IPPO + IPPO-RNN | `aa6v2u0iyhwbtl` | PRO 6000 MIG 24GB | 8 | $4.72 |

Both pods use `bic4907/overcooked:cu13-rp` and a 100 GB pod volume mounted at
`/workspace`. Total live compute cost is $9.44/hour.

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
old processes cannot be inspected. Re-query with backoff. When a pod returns to
`running`, first verify `/workspace`, gateway SSH, tmux, logs, completion
markers, and process state. If the old jobs did not survive, preserve every
finished artifact-bearing run and create a versioned replacement sweep for only
the interrupted/missing configurations after the old W&B runs are terminal.

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
