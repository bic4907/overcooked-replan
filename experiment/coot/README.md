# CooT W&B sweeps

Run every command from the repository root. The project passed to `wandb
sweep` is authoritative; during an agent run W&B intentionally ignores the
Hydra `PROJECT` field. Do not register a CooT sweep in an FCP/IPPO project.

## Fast smoke suite

The first stage has no checkpoint or dataset prerequisite and is safe to run
immediately:

```bash
uv run python baselines/CooT/preflight_sweep.py \
  experiment/coot/smoke_population.yaml
uv run wandb sweep --entity cilab-overcooked \
  --project overcooked-v3-coot-smoke \
  experiment/coot/smoke_population.yaml
```

Start the returned sweep with exactly one agent:

```bash
GPUS=0 bash scripts/overcooked_v3/run_wandb_agents.sh \
  cilab-overcooked/overcooked-v3-coot-smoke/<SWEEP_ID>
```

`smoke_population.yaml` performs two one-step PPO updates. It checks HSP
reward wiring, the recurrent learner, mid/final checkpoints, the candidate
sidecar, and W&B artifact upload. It is plumbing-only and its return must not
be reported as a baseline result.

The remaining smoke files are deliberately stage-separated because their
inputs are produced by the preceding stage:

1. `smoke_response.yaml`: one fixed-partner BR update; requires
   `manifests/coot/smoke/response_jobs/split_0.json` and its partner checkpoint.
2. `smoke_train.yaml`: one train and validation batch for seeds 0 and 1;
   requires a horizon-1 dataset under `datasets/coot_smoke/split_0`.
3. `smoke_eval.yaml`: a 2x2 ordered SP/XP matrix with one-step episodes;
   requires both smoke-train checkpoints.

Run `preflight_sweep.py` before registering each one. It checks the program,
isolated W&B project, job bounds, manifest-relative partner checkpoints,
dataset shards, or seed checkpoints as appropriate. `--static-only` checks the
sweep schema/namespaces before its input artifacts exist.

## Production projects

Each stage has a separate W&B project and local namespace:

| Sweep | W&B project | Local output/input root |
| --- | --- | --- |
| `population*.yaml` | `overcooked-v3-coot-population` | `saves/coot_population` |
| `response_candidates*.yaml` | `overcooked-v3-coot-response-candidates` | candidate manifests + `saves/coot_responses` |
| `response.yaml`, `response_hsp_only.yaml` | `overcooked-v3-coot-response` | stage-specific manifests + `saves/coot_responses` |
| `train.yaml` | `overcooked-v3-coot-train` | `datasets/coot` -> `saves/coot_train` |
| `eval.yaml` | `overcooked-v3-coot-eval` | `saves/coot_train` -> `saves/coot_eval/crossplay` |

Exact HSP+MEP response jobs use
`manifests/coot/response_jobs/<layout>.json`; the explicit HSP-only proxy uses
`manifests/coot/response_jobs_hsp_only/<layout>.json`. Candidate response jobs
use `manifests/coot/response_candidates/<layout>.json`. These roots must not be
interchanged.

Population, response, train, and evaluation files are not one Cartesian
sweep: artifacts must be collected and manifests/datasets built between
stages. See `docs/overcooked_v3/coot.md` for the complete production order.
