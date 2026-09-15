# Overcooked Replan

This repository extends JaxMARL's Overcooked V2 environment to study dynamic
resource changes and test-time role reconfiguration. The original `overcooked_v2`
implementation remains intact, while the new environment and experiments live
under `overcooked_v3`.

The following role-coordination experiments are currently available:

| Hydra scenario | Environment | Research question |
| --- | --- | --- |
| `split_0` | Diagonal Narrow Kitchen Split | Does the compact diagonal arrangement preserve coordination while increasing cross-play difficulty? |
| `split_1` | Kitchen Split | Can agents choose opposite bays before the doorway closes and sustain complementary roles? |
| `outage_0` | Diagonal Narrow Resource Outage | Can agents prepare for the loss of every plate dispenser through compact diagonal routes? |
| `outage_1` | Shared-room Resource Outage | Can agents prepare and ration plates before every dispenser becomes unavailable? |
| `recipe_switch_0` | Mixed Recipe Relay | Can agents reverse supplier–cook roles as the shared recipe follows a fixed A→B→A schedule? |
| `distance_0` | Distance-Driven Role Switch | Can agents exchange cook/server roles when reachable stations swap asymmetric near/far costs? |
| `distance_1` | Wide Distance-Driven Role Switch | Can agents reassign roles when near/far distance differences increase? |
| `split_wide` | Kitchen Split, 11×7 | Can agents coordinate between opposite-edge stations around two interior obstacles? |
| `outage_wide` | Shared-room Resource Outage, 11×6 | Can agents compensate when every onion dispenser disappears around staggered obstacles? |
| `distance_switch_wide` | Distance-Driven Role Switch, 13×6 | Can agents reassign roles when near/far station advantages reverse on a wider map? |
| `split_hard` | Random-order Kitchen Split | Can agents generalize across three maps to a held-out order? |
| `outage_hard` | Random-order Resource Outage | Can agents adapt to three outage maps in an unseen order? |
| `distance_switch_hard` | Random-order Distance Switch | Can agents reassign routes across three cost configurations in an unseen order? |

The selected six-layout benchmark is `split_0`, `split_1`, `outage_0`,
`outage_1`, `distance_0`, and `distance_1`. The `_0` and `_1` suffixes identify
the two retained layouts in each family. Earlier descriptive names remain
registered so existing checkpoints and experiment records can still be reproduced.

Split, Outage, and Distance alternate their A and B maps every 75 steps, with
transitions at steps 75, 150, 225, 300, and 375 in a 450-step episode. Kitchen
Split alternates a central doorway and a handoff counter. Resource Outage uses
one shared room and removes every plate dispenser during phase B; stored plates,
held inventory, and pot contents survive the transition. Distance keeps the
standard three-onion recipe and reverses the short onion-input and serving loops.
Outage uses a two-onion recipe; all other cooking timers remain unchanged.

[Wide maps](docs/overcooked_v3/wide_maps.md) use larger route geometries: Split
11×7, Outage 11×6, and Distance Switch 13×6 (width×height). They use the same
75-step alternation as the selected six layouts.

[Hard mode](docs/overcooked_v3/hard_mode.md) keeps the `_0` footprints and uses
three distinct maps A/B/C. Training samples five orders and evaluation holds out
CBA. Hard mode retains transitions at steps 150 and 300.

Mixed Recipe Relay permanently separates an onion/serving bay from a
tomato/plate bay and exposes exactly two shared handoff counters. Both bays have
pots. The retained layout is former catalog `_7`, reindexed as `_0`. It is a
7×5 tomato-major-first layout. The map stays fixed while the recipe changes at
steps 150 and 300 within a 450-step episode.
Select any layout through its Hydra scenario name, such as
`scenario=outage_0`.

Overcooked V3 exposes upcoming layout transitions to every agent. The final two
channels of the default 31-channel observation contain a global transition
countdown and a binary map-change mask. They stay at zero until 20 steps before
a layout change. Rendered GIFs blink an orange border on changing tiles and draw
the remaining transition step count on each affected tile.
Recipe Relay adds two more preview channels at the recipe indicator, one per
ingredient, producing 33 channels so both agents observe the next recipe as
well as the current one.

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
  --layout split_0 \
  --steps 220 \
  --seed 0 \
  --gif evaluation/previews/split_0.gif
```

The resulting GIF is saved to `evaluation/previews/split_0.gif`.

## Human play and BC demonstrations

Follow the [Korean play-tool manual](docs/overcooked_v3/human_demonstrations.md)
for first-time installation, controls, troubleshooting, and BC dataset export.
From the repository root with its virtual environment activated:

```bash
python -m pip install -e ".[human]"
python scripts/overcooked_v3/collect_human.py \
  --layout split_0 --mode realtime --hz 5 --players player01 player02
```

Press **Space** to start or pause. Red uses **WASD + Q**; blue uses **arrow keys +
right Shift**. The display/input loop targets **60 FPS** while the environment
runs at **5 steps per second**. Use `--mode step` for one person controlling both
agents, selecting their actions before pressing Space for each step.

Add `--random-start --output data/human/random_start` to sample initial positions
and facing within each agent's connected room. Each new episode advances the
seed; the same initial seed reproduces the sequence. Export that folder separately
from fixed-start demonstrations.

Finish a full episode, press **K** to accept it, then **N** for the next episode.
After accepting at least two complete episodes, export the BC dataset:

```bash
python scripts/overcooked_v3/prepare_bc_data.py export data/human \
  --layout split_0 --output data/bc/split_0_v1 --val-fraction 0.2 --seed 0
```

Raw episodes are saved under `data/human/<layout>/`. The export writes `train.npz`,
`val.npz`, and `manifest.json` with an episode-level train/validation split.

## W&B and environment variables

Copy the example environment file:

```bash
cp .env.example .env
```

Add your W&B credentials and settings to `.env`:

```dotenv
WANDB_API_KEY=your-api-key
WANDB_ENTITY=cilab-overcooked
WANDB_MODE=online
```

`WANDB_ENTITY` must be a team slug or personal username, not an organization
slug. In a W&B workspace URL such as
`https://wandb.ai/<entity>/<project>`, use the `<entity>` segment. If it points
to an organization, open the target team workspace and use that team's slug.
Training and evaluation projects are selected by the algorithm-specific config
or sweep; avoid a global `WANDB_PROJECT` override when using the six-project
IPPO/FCP/PolicySwitch split.

The training entrypoint automatically loads `.env` from the project root. The
file is excluded from Git. Set `WANDB_MODE=disabled` when W&B is not needed.
Direct training commands do not need a `dotenv run` prefix.
The V3 trainer defaults to online mode and automatically falls back to offline
mode when `WANDB_API_KEY` is not set.

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
  scenario=split_0 \
  EXPERIMENT_FOLDER=baseline \
  SEED=0 \
  NUM_SEEDS=1
```

Change only the `scenario` override to run another condition:

```bash
scenario=outage_0
```

CNN is the default policy architecture. Select the RNN policy as follows:

```bash
python -u baselines/IPPO/ippo_overcooked_v3.py \
  scenario=outage_0 \
  ARCHITECTURE=rnn \
  EXPERIMENT_FOLDER=baseline \
  SEED=0
```

When `scenario` is omitted, `split_0` is used.

### Short dry run

Before starting a full training run, use this CPU-only configuration to perform
one update and verify the training and output paths:

```bash
JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false \
python -u baselines/IPPO/ippo_overcooked_v3.py \
  scenario=split_0 \
  EXPERIMENT_FOLDER=dry-run \
  NUM_ENVS=2 \
  NUM_STEPS=2 \
  NUM_MINIBATCHES=1 \
  UPDATE_EPOCHS=1 \
  TOTAL_TIMESTEPS=4 \
  REW_SHAPING_HORIZON=4 \
  LOG_INTERVAL=1 \
  wandb_mode=disabled
```

This command writes the experiment to
`saves/split_0_cnn_dry-run_seed0/`.

### Inspect the resolved Hydra configuration

Print the effective configuration without starting training:

```bash
python baselines/IPPO/ippo_overcooked_v3.py \
  scenario=outage_0 \
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
scenario=split_0 ARCHITECTURE=cnn EXPERIMENT_FOLDER=baseline SEED=2
```

creates the following directory:

```text
saves/split_0_cnn_baseline_seed2/
```

If `EXPERIMENT_FOLDER` is omitted, the directory is
`saves/split_0_cnn_seed2/`. Include only meaningful experimental axes in the
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
  scenario=split_0 \
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

See [the self-play guide](experiment/self_play/README.md) for its W&B sweep
commands.

Fictitious Co-Play uses a two-stage self-play-population and best-response
workflow. See [the FCP guide](experiment/fcp/README.md) for its W&B sweep
commands.

`experiment/self_play/train.yaml` defines a 48-run grid over all 8 layouts
and six seeds. Create it on a Mac
with the W&B CLI:

```bash
wandb sweep \
  --entity cilab-overcooked \
  --project overcooked-v3-ippo_train \
  experiment/self_play/train.yaml
```

W&B prints `cilab-overcooked/overcooked-v3-ippo_train/SWEEP_ID`. Copy that
full path to the GPU server and launch one agent on each GPU:

```bash
GPUS="0 1 2 3" bash scripts/overcooked_v3/run_wandb_agents.sh \
  cilab-overcooked/overcooked-v3-ippo_train/SWEEP_ID
```

Each GPU processes one run at a time until W&B reports that the sweep is
complete. Multiple sweep paths can be supplied; all GPU agents finish the first
sweep before the next one starts:

```bash
GPUS="0 1 2 3" bash scripts/overcooked_v3/run_wandb_agents.sh \
  cilab-overcooked/overcooked-v3-ippo_train/SWEEP_ID_A \
  cilab-overcooked/overcooked-v3-ippo_train/SWEEP_ID_B
```

For example, a sweep run is saved under a directory such as
`saves/split_0_cnn_seed0/`.

W&B metrics are grouped by slash-delimited namespaces:

| Namespace | Contents |
| --- | --- |
| `train/...` | Episode return and length, sparse/shaped/combined rewards, PPO losses, entropy, learning rate, update, and environment step |
| `debug/...` | Layout phase and changes, transition countdown, changed-tile count, and global/left/right workload and resource tile counts |
| `eval/...` | Return and length of the final recorded episode |
| `visualization/...` | Final-episode MP4 and recording diagnostics |

The role-scenario sweep maximizes `train/episode_return`. Layout snapshots such
as `debug/layout_index` and `debug/transition_countdown` represent the end of
the latest rollout; `debug/layout_change_events` counts all phase transitions
observed during that rollout batch.

At the end of training, the first trained seed runs one deterministic episode.
A compact 10 FPS MP4 is saved in the experiment directory and uploaded as
`visualization/final_episode`. The Hydra default is `recording=enabled`; pass
`recording=disabled` to turn it off. With recording enabled, customize it using
`RECORD_MAX_STEPS`, `RECORD_VIDEO_FPS`, and `RECORD_VIDEO_QUALITY`. Recording is
also skipped when `wandb_mode=disabled`.

```bash
python -u baselines/IPPO/ippo_overcooked_v3.py \
  scenario=split_0 \
  recording=disabled \
  SEED=0
```

## Evaluate and render trained policies

Evaluate two policies trained with the same seed and save the first episode as
a GIF:

```bash
JAX_PLATFORMS=cpu MPLCONFIGDIR=/tmp \
python baselines/IPPO/eval_ippo_overcooked_v3.py \
  --layout split_0 \
  --architecture cnn \
  --agent-seeds 0 0 \
  --episodes 3 \
  --max-steps 450 \
  --gif evaluation/split_0_same_seed0.gif
```

For cross-play, select policies trained with different seeds:

```bash
JAX_PLATFORMS=cpu MPLCONFIGDIR=/tmp \
python baselines/IPPO/eval_ippo_overcooked_v3.py \
  --layout split_0 \
  --architecture cnn \
  --agent-seeds 0 1 \
  --episodes 3 \
  --max-steps 450 \
  --gif evaluation/split_0_cross_seed0_seed1.gif
```

With `--agent-seeds`, the evaluator finds the newest final checkpoint for the
requested layout and seed under `saves/`. To evaluate a specific file, provide
its path explicitly with `--checkpoint`:

```bash
python baselines/IPPO/eval_ippo_overcooked_v3.py \
  --layout split_0 \
  --checkpoint saves/split_0_cnn_baseline_seed0/ippo_cnn_overcooked_v3_split_0_seed0_vmap0.safetensors \
  --episodes 1 \
  --render \
  --render-delay 0.2
```

On a headless server, use `--gif` instead of `--render`.

The signal-free V3 grid encoding has 29 base channels. Countdown and change-mask
features produce the default 31-channel observation. Disable both transition
features when evaluating a checkpoint trained with the 29-channel encoding:

```bash
python baselines/IPPO/eval_ippo_overcooked_v3.py \
  --layout split_0 \
  --checkpoint PATH_TO_OLD_CHECKPOINT.safetensors \
  --legacy-observation
```

For a 30-channel checkpoint trained with the countdown but without the change
mask, use `--no-layout-change-mask`. Checkpoints from the removed 33-channel
signal-enabled environment are not shape-compatible with the new encoding.

## Batch training and evaluation of role scenarios

Train CNN policies on all four selected Easy role scenarios:

```bash
TRAIN_SEEDS="0 1" \
TOTAL_TIMESTEPS=3e7 \
bash scripts/overcooked_v3/train_all_overcooked_v3_cnn.sh \
  SAVES_DIR=saves
```

Evaluate same-seed and cross-seed combinations of the trained role-scenario
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
| `experiment/` | Copy-and-run commands and W&B sweep YAML configurations |
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

### Human behavior cloning

`split_0` 인간 시연으로 학습한 BC 모델과 재학습/추론 방법은
[BC 사용 문서](baselines/BC/README.md)를 참고하세요.
[저장된 모델과 결과](artifacts/bc/split_0_v1/README.md)도 포함되어 있습니다.
