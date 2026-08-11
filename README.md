# Overcooked Replan

This repository extends JaxMARL's Overcooked V2 environment to study dynamic
resource changes and test-time role reconfiguration. The original `overcooked_v2`
implementation remains intact, while the new environment and experiments live
under `overcooked_v3`.

The following role-coordination experiments are currently available:

| Hydra scenario | Environment | Signal counter | Research question |
| --- | --- | --- | --- |
| `split_no_sig` | Kitchen Split | No | Can agents form complementary spatial roles using movement alone? |
| `split_sig` | Kitchen Split | Yes | Does a shared counter reduce role conflicts and incorrect area choices? |
| `outage_no_sig` | Resource Outage | No | Can agents reallocate collection and cooking roles after a resource outage? |
| `outage_sig` | Resource Outage | Yes | Does signaling accelerate role reallocation and recovery? |

Overcooked V3 exposes the upcoming layout transition to every agent. The final
two channels of the default 32-channel observation contain a global continuous
countdown and a binary map-change mask. The countdown decreases from `1.0` to
`0.0` within each phase, while the mask marks tiles whose static object will
change in the next phase. Rendered GIFs show the countdown at 5 FPS and draw an
orange warning border around those tiles.

## Quick start

Python 3.11 or later is required. Create and activate a virtual environment,
then install the project and its training and development dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[algs,dev]"
```

Run the remaining commands from the same activated shell. Verify that the
virtual environment is active with `which python`; it should point to
`.venv/bin/python` inside this repository.

Run a random-policy rollout and save it as a GIF to verify that the environment
works correctly:

```bash
python scripts/overcooked_v3/run_role_scenario.py \
  --layout split_sig \
  --steps 220 \
  --seed 0 \
  --gif evaluation/previews/split_sig.gif
```

The resulting GIF is saved to `evaluation/previews/split_sig.gif`.

## W&B and environment variables

Copy the example environment file:

```bash
cp .env.example .env
```

Add your W&B credentials and settings to `.env`:

```dotenv
WANDB_API_KEY=your-api-key
WANDB_ENTITY=your-team-slug
WANDB_PROJECT=overcooked-v3-role-coordination
WANDB_MODE=online
```

`WANDB_ENTITY` must be a team slug or personal username, not an organization
slug. In a W&B workspace URL such as
`https://wandb.ai/<entity>/<project>`, use the `<entity>` segment. If it points
to an organization, open the target team workspace and use that team's slug.

The training entrypoint automatically loads `.env` from the project root. The
file is excluded from Git. Set `WANDB_MODE=disabled` when W&B is not needed.
Direct training commands do not need a `dotenv run` prefix.

W&B-related settings use the following precedence order. This does not apply to
`SAVES_DIR`.

1. Hydra command-line overrides
2. Existing shell environment variables
3. Values from `.env`
4. Hydra defaults

## Training

### Run one experiment

```bash
python -u baselines/IPPO/ippo_overcooked_v3.py \
  scenario=split_no_sig \
  EXPERIMENT_FOLDER=baseline \
  SEED=0 \
  NUM_SEEDS=1
```

Change only the `scenario` override to run another condition:

```bash
scenario=split_sig
scenario=outage_no_sig
scenario=outage_sig
```

CNN is the default policy architecture. Select the RNN policy as follows:

```bash
python -u baselines/IPPO/ippo_overcooked_v3.py \
  scenario=outage_sig \
  ARCHITECTURE=rnn \
  EXPERIMENT_FOLDER=baseline \
  SEED=0
```

When `scenario` is omitted, the existing `dynamic_00` map is used.

### Short dry run

Before starting a full training run, use this CPU-only configuration to perform
one update and verify the training and output paths:

```bash
JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false \
python -u baselines/IPPO/ippo_overcooked_v3.py \
  scenario=split_no_sig \
  EXPERIMENT_FOLDER=dry-run \
  NUM_ENVS=2 \
  NUM_STEPS=2 \
  NUM_MINIBATCHES=1 \
  UPDATE_EPOCHS=1 \
  TOTAL_TIMESTEPS=4 \
  REW_SHAPING_HORIZON=4 \
  LOG_INTERVAL=1 \
  WANDB_MODE=disabled
```

This command writes the experiment to
`saves/split_no_sig_cnn_dry-run_seed0/`.

### Inspect the resolved Hydra configuration

Print the effective configuration without starting training:

```bash
python baselines/IPPO/ippo_overcooked_v3.py \
  scenario=outage_sig \
  --cfg job --resolve
```

## Experiment names and output paths

Experiments use the following directory structure by default:

```text
saves/
└── <layout>_<architecture>_<experiment-name>_seed<seed>/
    ├── <run>_config.yaml
    ├── <run>_vmap0_update000050.safetensors  # When periodic saves are enabled
    ├── <run>_vmap0.safetensors               # Final checkpoint
    └── <run>_vmap0_final_episode.mp4          # Final deterministic rollout
```

For example, this configuration:

```text
scenario=split_sig ARCHITECTURE=cnn EXPERIMENT_FOLDER=baseline SEED=2
```

creates the following directory:

```text
saves/split_sig_cnn_baseline_seed2/
```

If `EXPERIMENT_FOLDER` is omitted, the directory is
`saves/split_sig_cnn_seed2/`. Include only meaningful experimental axes in the
name. For example, when a normally fixed learning rate becomes an ablation
variable, use a name such as `EXPERIMENT_FOLDER=lr-1e-4`.

Running the same layout, architecture, experiment name, and seed again may
overwrite the existing configuration and checkpoints.

`SAVES_DIR` is managed by Hydra rather than by shell environment variables or
`.env`. Its default value is `saves` in `conf/ippo_overcooked_v3.yaml`. To change
the storage root, pass a Hydra override to the training command. `/mnt/nas` is
not hardcoded anywhere in the training code.

```bash
python -u baselines/IPPO/ippo_overcooked_v3.py \
  scenario=split_sig \
  SAVES_DIR=/mnt/nas/overcooked-replan \
  EXPERIMENT_FOLDER=baseline \
  SEED=0
```

Only experiment configurations and checkpoints are stored under `saves/`.
Auxiliary outputs use separate default directories:

| Output | Default location |
| --- | --- |
| Experiment configurations, checkpoints, and final-rollout videos | `saves/` |
| Hydra single-run logs | `outputs/` |
| Hydra multirun logs | `multirun/` |
| Local W&B files | `wandb/` |
| GIFs and evaluation statistics | `evaluation/` |

## W&B sweep

`sweeps/overcooked_v3_role_scenarios.yaml` defines a 20-run grid over four
scenarios and five seeds. Create it on a Mac with the W&B CLI, replacing
`YOUR_TEAM_SLUG` with the target team entity:

```bash
wandb sweep \
  --entity YOUR_TEAM_SLUG \
  --project overcooked-v3-role-coordination \
  sweeps/overcooked_v3_role_scenarios.yaml
```

W&B prints `YOUR_TEAM_SLUG/overcooked-v3-role-coordination/SWEEP_ID`. Copy that
full path to the GPU server and launch one agent on each GPU:

```bash
GPUS="0 1 2 3" bash scripts/overcooked_v3/run_wandb_agents.sh \
  YOUR_TEAM_SLUG/overcooked-v3-role-coordination/SWEEP_ID
```

Each GPU processes one run at a time until W&B reports that the sweep is
complete. Multiple sweep paths can be supplied; all GPU agents finish the first
sweep before the next one starts:

```bash
GPUS="0 1 2 3" bash scripts/overcooked_v3/run_wandb_agents.sh \
  TEAM/PROJECT/SWEEP_ID_A \
  TEAM/PROJECT/SWEEP_ID_B
```

For example, a sweep run is saved under a directory such as
`saves/split_sig_cnn_role-scenarios_seed0/`.

W&B metrics are grouped by slash-delimited namespaces:

| Namespace | Contents |
| --- | --- |
| `train/...` | Episode return and length, sparse/shaped/combined rewards, PPO losses, entropy, learning rate, update, and environment step |
| `debug/...` | Layout phase, transition fraction and event count, countdown, changed-tile count, and wall/resource/signal tile counts |
| `eval/...` | Return and length of the final recorded episode |
| `visualization/...` | Final-episode MP4 and recording diagnostics |

The role-scenario sweep maximizes `train/episode_return`. Layout snapshots such
as `debug/layout_index` and `debug/transition_countdown` represent the end of
the latest rollout; `debug/layout_change_events` counts all phase transitions
observed during that rollout batch.

At the end of training, the first trained seed runs one deterministic episode.
A compact 5 FPS MP4 is saved in the experiment directory and uploaded as
`visualization/final_episode`. The Hydra default is `recording=enabled`; pass
`recording=disabled` to turn it off. With recording enabled, customize it using
`RECORD_MAX_STEPS`, `RECORD_VIDEO_FPS`, and `RECORD_VIDEO_QUALITY`. Recording is
also skipped when `WANDB_MODE=disabled`.

```bash
python -u baselines/IPPO/ippo_overcooked_v3.py \
  scenario=split_sig \
  recording=disabled \
  SEED=0
```

## Evaluate and render trained policies

Evaluate two policies trained with the same seed and save the first episode as
a GIF:

```bash
JAX_PLATFORMS=cpu MPLCONFIGDIR=/tmp \
python baselines/IPPO/eval_ippo_overcooked_v3.py \
  --layout split_no_sig \
  --architecture cnn \
  --agent-seeds 0 0 \
  --episodes 3 \
  --max-steps 400 \
  --gif evaluation/split_no_sig_same_seed0.gif
```

For cross-play, select policies trained with different seeds:

```bash
JAX_PLATFORMS=cpu MPLCONFIGDIR=/tmp \
python baselines/IPPO/eval_ippo_overcooked_v3.py \
  --layout split_no_sig \
  --architecture cnn \
  --agent-seeds 0 1 \
  --episodes 3 \
  --max-steps 400 \
  --gif evaluation/split_no_sig_cross_seed0_seed1.gif
```

With `--agent-seeds`, the evaluator finds the newest final checkpoint for the
requested layout and seed under `saves/`. To evaluate a specific file, provide
its path explicitly with `--checkpoint`:

```bash
python baselines/IPPO/eval_ippo_overcooked_v3.py \
  --layout split_no_sig \
  --checkpoint saves/split_no_sig_cnn_baseline_seed0/ippo_cnn_overcooked_v3_split_no_sig_seed0_vmap0.safetensors \
  --episodes 1 \
  --render \
  --render-delay 0.2
```

On a headless server, use `--gif` instead of `--render`.

The transition features change the default V3 observation from 30 to 32
channels, so policies trained before these changes require the legacy
observation flag during evaluation:

```bash
python baselines/IPPO/eval_ippo_overcooked_v3.py \
  --layout split_no_sig \
  --checkpoint PATH_TO_OLD_CHECKPOINT.safetensors \
  --legacy-observation
```

For a 31-channel checkpoint trained with the countdown but without the change
mask, use `--no-layout-change-mask` instead.

## Batch training and evaluation of dynamic maps

Train CNN policies on `dynamic_00` through `dynamic_14`:

```bash
TRAIN_SEEDS="0 1" \
TOTAL_TIMESTEPS=3e7 \
bash scripts/overcooked_v3/train_all_overcooked_v3_cnn.sh \
  SAVES_DIR=saves
```

Evaluate same-seed and cross-seed combinations of the trained dynamic-map
policies:

```bash
EVALUATION_DIR=evaluation/overcooked_v3/cnn \
bash scripts/overcooked_v3/eval_all_overcooked_v3_cnn.sh
```

## Tests

Run the Overcooked V2 and V3 regression tests:

```bash
python -m pytest -q tests/overcooked_v3 tests/overcooked_v2
```

Run the code-style checks:

```bash
python -m ruff check .
```

## Repository layout

| Path | Contents |
| --- | --- |
| `jaxmarl/environments/overcooked_v3/` | Overcooked V3 environment implementation |
| `jaxmarl/environments/overcooked_v2/` | Preserved Overcooked V2 implementation |
| `baselines/IPPO/ippo_overcooked_v3.py` | CNN/RNN IPPO training entrypoint |
| `conf/` | Hydra defaults and scenario configurations |
| `sweeps/` | W&B sweep configurations |
| `experiment/` | Copy-and-run experiment commands |
| `scripts/overcooked_v3/` | Rollout, batch-training, and batch-evaluation scripts |
| `docs/overcooked_v3/` | Environment design and detailed workflows |

Additional documentation:

- [Executable W&B sweep commands](experiment/role_scenarios.md)
- [Overcooked V3 documentation](docs/overcooked_v3/index.md)
- [Training and W&B configuration](docs/overcooked_v3/training.md)
- [Environment development and evaluation workflow](docs/overcooked_v3/workflow.md)

## Upstream project

This repository is based on [JaxMARL](https://github.com/FLAIROx/JaxMARL). See
[LICENSE](LICENSE) and the upstream JaxMARL repository for licensing and citation
information.
