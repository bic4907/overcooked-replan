# The cdoff and multimap arms of the 75-step plate benchmark

Two arms added beside the 09-15 baseline, on the same six kitchens, the same
450-step episodes and the same 75-step phase clock. Only the training-time
condition differs, and every cell is scored by the one protocol.

## What the three arms are

The baseline is already in W&B and is not rerun:

| arm | countdown | change mask | observer gate | layout_mode | channels |
| --- | --- | --- | --- | --- | ---: |
| baseline (09-15) | on | on | `both` | `cyclic` | 31 |
| cdoff | on | on | `none` | `cyclic` | 31 |
| multimap | on | on | `both` | `episode_random` | 31 |

All three hand the policy the same 31 channels. The two transition channels
carry a real signal only in the baseline:

* **cdoff** keeps them and holds them at zero through the observer gate, so a
  boundary arrives unannounced. Nothing else leaks the clock: the base
  observation has no step feature, and these six layouts schedule no recipes,
  so there is no next-recipe channel either. A convolutional policy cannot know
  a change is coming; a recurrent one can still count, because the 75-step
  cadence never varies.
* **multimap** draws one of the two distinct kitchens at reset and holds it for
  the episode, so there is no boundary to announce and the channels are
  constant zero by construction. Training never crosses a boundary; evaluation
  always does.

Matching the widths is deliberate. It costs nothing, it lets one evaluator read
every arm, and it means the comparison is about what a policy can read rather
than about the shape of its input.

## Layouts and scale

```
split_0  split_1  outage_0  outage_1  distance_0  distance_1
```

Six layouts x six seeds = 36 runs per sweep; 288 training runs and 36
evaluation runs in all.

## Order

Everything in stage 1 is independent. FCP needs its population first.

| Stage | Sweep file | Project | Runs |
| --- | --- | --- | ---: |
| 1 | `ippo_cnn_cdoff_train.yaml` | `overcooked-v3-ippo-cdoff-cnn-75step-plate-0915_train` | 36 |
| 1 | `ippo_rnn_cdoff_train.yaml` | `overcooked-v3-ippo-cdoff-rnn-75step-plate-0915_train` | 36 |
| 1 | `ippo_cnn_multimap_train.yaml` | `overcooked-v3-ippo-multimap-cnn-75step-plate-0915_train` | 36 |
| 1 | `ippo_rnn_multimap_train.yaml` | `overcooked-v3-ippo-multimap-rnn-75step-plate-0915_train` | 36 |
| 2 | `fcp_cdoff_population.yaml` | `overcooked-v3-fcp-cdoff-75step-plate-0915_population` | 36 |
| 2 | `fcp_multimap_population.yaml` | `overcooked-v3-fcp-multimap-75step-plate-0915_population` | 36 |
| 3 | `fcp_cdoff_train.yaml` | `overcooked-v3-fcp-cdoff-75step-plate-0915_train` | 36 |
| 3 | `fcp_multimap_train.yaml` | `overcooked-v3-fcp-multimap-75step-plate-0915_train` | 36 |
| 4 | the six `*_eval.yaml` files | `..._eval`, one per cell | 6 each |

The population sweeps write to `POPULATION_ROOT` scoped per arm
(`saves/75step_plate_0915/{cdoff,multimap}/fcp_population`), and the FCP sweeps
read the same root back. Keep that scoping. `discover_population_checkpoints`
filters on architecture and layout only, and now that every arm shares one
observation width, an unscoped root no longer fails loudly -- it quietly trains
one arm against the other arm's partners.

## What it costs

Medians measured from the 09-15 runs already in W&B, one GPU per run:

| Cell | Runs | Median per run | Total |
| --- | ---: | ---: | ---: |
| IPPO-CNN (either arm) | 36 | 0.13 h | 5 GPU-h |
| IPPO-RNN (either arm) | 36 | 0.59 h | 21 GPU-h |
| FCP population (either arm) | 36 | 0.56 h | 20 GPU-h |
| FCP best response (either arm) | 36 | 0.68 h | 24 GPU-h |
| Evaluation | 36 | 14 min | 8 GPU-h |

About **140 GPU-hours** of training and **8** of evaluation. On four GPUs that
is a day and a half of wall clock, set by the FCP stages, which cannot start
until their population finishes.

## Running it

```bash
git clone <repo> overcooked-replan && cd overcooked-replan
python -c "import jax; print(jax.devices())"        # CUDA devices, not CPU

wandb sweep --entity cilab-overcooked \
  --project overcooked-v3-ippo-cdoff-cnn-75step-plate-0915_train \
  experiment/countdown/v3/75step_plate/ippo_cnn_cdoff_train.yaml
```

`experiment/run_agents_sequential.sh` takes sweeps in order, runs one agent per
GPU id listed, and starts the next sweep only once every agent on the current
one has exited -- which is what the population-then-FCP ordering needs.

```bash
GPUS="0 1 2 3" bash experiment/run_agents_sequential.sh \
  cilab-overcooked/<project>/<sweep_id> ...
```

Repeat a GPU id to put more than one agent on a card. The CNN runs are small
enough for two or three per card; the RNN and FCP runs are not.

## Checks before leaving it alone

* One run of each sweep reaches `update=1` and logs to the right project.
* `ENV_KWARGS` on a live run shows the `layout_mode` and `transition_observer`
  the sweep intended, not the config defaults.
* A population run uploads snapshots at 10, 50 and 100 percent, and the FCP
  sweep finds 18 partners per layout instead of falling back to a discovery
  error.

## Checks when it finishes

```python
import collections, wandb
api = wandb.Api()
projects = (
    "overcooked-v3-ippo-cdoff-cnn-75step-plate-0915_train",
    "overcooked-v3-ippo-cdoff-rnn-75step-plate-0915_train",
    "overcooked-v3-ippo-multimap-cnn-75step-plate-0915_train",
    "overcooked-v3-ippo-multimap-rnn-75step-plate-0915_train",
    "overcooked-v3-fcp-cdoff-75step-plate-0915_train",
    "overcooked-v3-fcp-multimap-75step-plate-0915_train",
)
for project in projects:
    runs = list(api.runs(f"cilab-overcooked/{project}", per_page=500))
    counts = collections.Counter(
        (str(r.config.get("ALGORITHM")), (r.config.get("ENV_KWARGS") or {}).get("layout"))
        for r in runs if r.state == "finished"
    )
    missing = [key for key, n in counts.items() if n != 6]
    print(project, len(runs), "runs;", "complete" if not missing else f"not six seeds: {missing}")
```

Every `(algorithm, layout)` should have six finished seeds and a `final`
checkpoint artifact. Each evaluation project should hold six runs, one per
kitchen.

## Things that went wrong before

* **Reusing checkpoints across a layout revision.** The kitchens were reindexed
  once and the old policies scored zero without an error. `LAYOUT_REVISION` is
  recorded with every run; refuse to evaluate across revisions.
* **Switching branches while agents run.** The checkout changes under them and
  every in-flight run dies on a missing file. Use a separate worktree for other
  work.
* **Concurrent CPU-heavy jobs.** Two of these on one machine exhausted memory in
  LLVM during compilation; run one per GPU rather than oversubscribing.

## The sweeps as created (2026-09-15)

```
# Stage 1 -- independent
cilab-overcooked/overcooked-v3-ippo-cdoff-cnn-75step-plate-0915_train/mdhmuowq
cilab-overcooked/overcooked-v3-ippo-cdoff-rnn-75step-plate-0915_train/t7j50z3e
cilab-overcooked/overcooked-v3-ippo-multimap-cnn-75step-plate-0915_train/r7uo5jfn
cilab-overcooked/overcooked-v3-ippo-multimap-rnn-75step-plate-0915_train/6wbor815
# Stage 2 -- frozen partners
cilab-overcooked/overcooked-v3-fcp-cdoff-75step-plate-0915_population/ujws5yoj
cilab-overcooked/overcooked-v3-fcp-multimap-75step-plate-0915_population/iun8zonk
# Stage 3 -- best responses, only after stage 2
cilab-overcooked/overcooked-v3-fcp-cdoff-75step-plate-0915_train/hk5zsa0f
cilab-overcooked/overcooked-v3-fcp-multimap-75step-plate-0915_train/imqy6x51
# Stage 4 -- the common protocol
cilab-overcooked/overcooked-v3-ippo-cdoff-cnn-75step-plate-0915_eval/4tytnbl4
cilab-overcooked/overcooked-v3-ippo-cdoff-rnn-75step-plate-0915_eval/nmfhxjcm
cilab-overcooked/overcooked-v3-fcp-cdoff-75step-plate-0915_eval/8fsl159p
cilab-overcooked/overcooked-v3-ippo-multimap-cnn-75step-plate-0915_eval/vozdivgi
cilab-overcooked/overcooked-v3-ippo-multimap-rnn-75step-plate-0915_eval/n30939uq
cilab-overcooked/overcooked-v3-fcp-multimap-75step-plate-0915_eval/15s9dgit
```

On a four-GPU machine, one runner call covers the whole experiment in order:

```bash
GPUS="0 1 2 3" bash experiment/run_agents_sequential.sh \
  cilab-overcooked/overcooked-v3-ippo-cdoff-cnn-75step-plate-0915_train/mdhmuowq \
  cilab-overcooked/overcooked-v3-ippo-multimap-cnn-75step-plate-0915_train/r7uo5jfn \
  cilab-overcooked/overcooked-v3-ippo-cdoff-rnn-75step-plate-0915_train/t7j50z3e \
  cilab-overcooked/overcooked-v3-ippo-multimap-rnn-75step-plate-0915_train/6wbor815 \
  cilab-overcooked/overcooked-v3-fcp-cdoff-75step-plate-0915_population/ujws5yoj \
  cilab-overcooked/overcooked-v3-fcp-multimap-75step-plate-0915_population/iun8zonk \
  cilab-overcooked/overcooked-v3-fcp-cdoff-75step-plate-0915_train/hk5zsa0f \
  cilab-overcooked/overcooked-v3-fcp-multimap-75step-plate-0915_train/imqy6x51 \
  cilab-overcooked/overcooked-v3-ippo-cdoff-cnn-75step-plate-0915_eval/4tytnbl4 \
  cilab-overcooked/overcooked-v3-ippo-cdoff-rnn-75step-plate-0915_eval/nmfhxjcm \
  cilab-overcooked/overcooked-v3-fcp-cdoff-75step-plate-0915_eval/8fsl159p \
  cilab-overcooked/overcooked-v3-ippo-multimap-cnn-75step-plate-0915_eval/vozdivgi \
  cilab-overcooked/overcooked-v3-ippo-multimap-rnn-75step-plate-0915_eval/n30939uq \
  cilab-overcooked/overcooked-v3-fcp-multimap-75step-plate-0915_eval/15s9dgit
```

The two CNN sweeps are small enough to take two agents per card; running them
first under `GPUS="0 0 1 1 2 2 3 3"` in a separate call halves their nine hours
at no risk to the rest. The RNN and FCP sweeps get one agent per card.
