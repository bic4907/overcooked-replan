# Runpod `_1` baseline state

Created: 2026-09-08 (Asia/Seoul)

## Pods

| Role | Pod ID | GPU | Count | Cost/hour |
| --- | --- | --- | ---: | ---: |
| FCP | `8rpv077f11ol4x` | PRO 6000 MIG 24GB | 8 | $4.72 |
| IPPO + IPPO-RNN | `aa6v2u0iyhwbtl` | PRO 6000 MIG 24GB | 8 | $4.72 |

Both pods use `bic4907/overcooked:cu13-rp` and a 100 GB pod volume mounted at
`/workspace`. Total live compute cost is $9.44/hour.

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
