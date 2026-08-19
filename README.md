# Overcooked Replan

This repository extends JaxMARL's Overcooked V2 environment to study dynamic
resource changes and test-time role reconfiguration. The original `overcooked_v2`
implementation remains intact, while the new environment and experiments live
under `overcooked_v3`.

The following role-coordination experiments are currently available:

| Hydra scenario | Environment | Signal counter | Research question |
| --- | --- | --- | --- |
| `splitnosig_{0..2}` | Kitchen Split | No | Can agents choose opposite bays before the doorway closes and sustain complementary roles? |
| `splitsig_{0..2}` | Kitchen Split | Yes | Does signaling reduce same-side choices before the kitchen splits? |
| `outagenosig_{0..2}` | Resource Outage | No | Can a cook pause local production and supply the other kitchen through a shared handoff counter? |
| `outagesig_{0..2}` | Resource Outage | Yes | Does signaling speed up the switch from parallel cooking to supplier–cook cooperation? |
| `recipe_switch_{0..9}` | Mixed Recipe Relay | No | Can agents reverse supplier–cook roles as the shared recipe follows a fixed A→B→A schedule? |
| `distance_switch_{0..9}` | Distance-Driven Role Switch | No | Can agents exchange cook/server roles when identical reachable stations move between asymmetric near/far positions? |

Kitchen Split starts with one central doorway open for 40 steps. It then becomes
a handoff counter for 160 steps, preventing agents from changing bays. The left
bay has onions and pots; the right bay has plates and serving. Agents must
choose opposite sides before closure and sustain complementary cook–server
roles through the counter. Resource Outage instead keeps two complete kitchens
in disconnected bays; the right onion pile disappears during the outage, so
the left cook must trade off local production against supplying the right bay.
Both conditions keep a recipe indicator at a separate fixed tile. NoSig has a
blank non-storage blocker where Sig provides the activatable public signal,
keeping the remaining geometry equal without displaying a dummy button.

Each category has three cross-play-selected layouts named `_0` through `_2`.
Matching Sig/NoSig indices have identical geometry and resources. Split uses a
7×9 map, while Outage uses a compact 5×7 map whose normal and outage phases last
40 and 160 steps. Outage keeps each onion-to-handoff and handoff-to-pot leg
within one movement step. The central wall always
separates agent movement, so cross-bay assistance is possible only by placing
objects on shared handoff counters. This keeps the right cook productive without
allowing it to walk to the surviving onion pile directly.
Split keeps the standard three-onion recipe, while Outage completes and starts
cooking a pot with two onions. Pot cooking time remains 20 steps in every
scenario.
Outage places two adjacent storage counters above the signal tile, allowing the
left cook to preload two onions for the right cook.
Mixed Recipe Relay permanently separates an onion/serving bay from a
tomato/plate bay and exposes exactly two shared handoff counters. Both bays have
pots. Eight layouts use compact 7×5 or 7×6 maps, while two 9×5 layouts retain a
small amount of routing variation. Variants `_0`–`_4` use
`2 onion + 1 tomato` → `1 onion + 2 tomato` → the
first recipe; variants `_5`–`_9` reverse that order. The map stays fixed while
the recipe changes at deterministic phase boundaries within a 450-step episode.
Select any layout through its Hydra scenario name, such as
`scenario=outagesig_2`. Matching Sig/NoSig layouts with the same index differ
only at the signal tile.

Distance-Driven Role Switch keeps the standard three-onion recipe fixed. Both
agents remain in one connected movement region and can reach every onion pile,
pot, plate pile, and serving station. During each 450-step episode the station
assignment follows A → B → A: onion/pot begin near agent 0 while plate/serving
begin near agent 1, then the two station groups exchange counter slots at step
150 and return at step 300. Ten `distance_switch_0`–`_9` layouts vary the
asymmetric counter geometry while enforcing at least a three-step spawn-to-
station advantage for the locally assigned agent.

Overcooked V3 exposes public signals and upcoming layout transitions to every
agent. The final three channels of the default 33-channel observation contain a
spatial signal timer, a global transition countdown, and a binary map-change
mask. Pressing a signal button sets its public channel to `1.0` for both agents;
it decreases to `0.1` over 10 observed steps and has no reward cost. The
transition channels stay at zero until 20 steps before a layout
change. Rendered GIFs label active buttons as `SIG 10...1`, blink an orange
border on changing tiles, and draw the remaining transition step count on each
affected tile.
Recipe Relay adds two more preview channels at the recipe indicator, one per
ingredient, so both agents observe the next recipe as well as the current one.

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
  --layout splitsig_0 \
  --steps 220 \
  --seed 0 \
  --gif evaluation/previews/splitsig_0.gif
```

The resulting GIF is saved to `evaluation/previews/splitsig_0.gif`.

## W&B and environment variables

Copy the example environment file:

```bash
cp .env.example .env
```

Add your W&B credentials and settings to `.env`:

```dotenv
WANDB_API_KEY=your-api-key
WANDB_ENTITY=inchangbaek4907
WANDB_PROJECT=overcooked-v3-role-coordination
WANDB_SOURCE_PROJECT=overcooked-v3-role-coordination
WANDB_MODE=online
```

`WANDB_ENTITY` must be a team slug or personal username, not an organization
slug. In a W&B workspace URL such as
`https://wandb.ai/<entity>/<project>`, use the `<entity>` segment. If it points
to an organization, open the target team workspace and use that team's slug.

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
  scenario=splitnosig_0 \
  EXPERIMENT_FOLDER=baseline \
  SEED=0 \
  NUM_SEEDS=1
```

Change only the `scenario` override to run another condition:

```bash
scenario=splitsig_0
scenario=outagenosig_0
scenario=outagesig_0
```

CNN is the default policy architecture. Select the RNN policy as follows:

```bash
python -u baselines/IPPO/ippo_overcooked_v3.py \
  scenario=outagesig_0 \
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
  scenario=splitnosig_0 \
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
`saves/splitnosig_0_cnn_dry-run_seed0/`.

### Inspect the resolved Hydra configuration

Print the effective configuration without starting training:

```bash
python baselines/IPPO/ippo_overcooked_v3.py \
  scenario=outagesig_0 \
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
scenario=splitsig_0 ARCHITECTURE=cnn EXPERIMENT_FOLDER=baseline SEED=2
```

creates the following directory:

```text
saves/splitsig_0_cnn_baseline_seed2/
```

If `EXPERIMENT_FOLDER` is omitted, the directory is
`saves/splitsig_0_cnn_seed2/`. Include only meaningful experimental axes in the
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
  scenario=splitsig_0 \
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

`experiment/self_play/train.yaml` defines a 72-run grid over all 12 layouts
and six seeds. Create it on a Mac
with the W&B CLI:

```bash
wandb sweep \
  --entity inchangbaek4907 \
  --project overcooked-v3-role-coordination \
  experiment/self_play/train.yaml
```

W&B prints `inchangbaek4907/overcooked-v3-role-coordination/SWEEP_ID`. Copy that
full path to the GPU server and launch one agent on each GPU:

```bash
GPUS="0 1 2 3" bash scripts/overcooked_v3/run_wandb_agents.sh \
  inchangbaek4907/overcooked-v3-role-coordination/SWEEP_ID
```

Each GPU processes one run at a time until W&B reports that the sweep is
complete. Multiple sweep paths can be supplied; all GPU agents finish the first
sweep before the next one starts:

```bash
GPUS="0 1 2 3" bash scripts/overcooked_v3/run_wandb_agents.sh \
  inchangbaek4907/overcooked-v3-role-coordination/SWEEP_ID_A \
  inchangbaek4907/overcooked-v3-role-coordination/SWEEP_ID_B
```

For example, a sweep run is saved under a directory such as
`saves/splitsig_0_cnn_seed0/`.

W&B metrics are grouped by slash-delimited namespaces:

| Namespace | Contents |
| --- | --- |
| `train/...` | Episode return and length, sparse/shaped/combined rewards, PPO losses, entropy, learning rate, update, and environment step |
| `debug/...` | Layout phase and changes, transition countdown, signal state and activation count, changed-tile count, and global/left/right workload and resource tile counts |
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
  scenario=splitsig_0 \
  recording=disabled \
  SEED=0
```

## Evaluate and render trained policies

Evaluate two policies trained with the same seed and save the first episode as
a GIF:

```bash
JAX_PLATFORMS=cpu MPLCONFIGDIR=/tmp \
python baselines/IPPO/eval_ippo_overcooked_v3.py \
  --layout splitnosig_0 \
  --architecture cnn \
  --agent-seeds 0 0 \
  --episodes 3 \
  --max-steps 400 \
  --gif evaluation/splitnosig_0_same_seed0.gif
```

For cross-play, select policies trained with different seeds:

```bash
JAX_PLATFORMS=cpu MPLCONFIGDIR=/tmp \
python baselines/IPPO/eval_ippo_overcooked_v3.py \
  --layout splitnosig_0 \
  --architecture cnn \
  --agent-seeds 0 1 \
  --episodes 3 \
  --max-steps 400 \
  --gif evaluation/splitnosig_0_cross_seed0_seed1.gif
```

With `--agent-seeds`, the evaluator finds the newest final checkpoint for the
requested layout and seed under `saves/`. To evaluate a specific file, provide
its path explicitly with `--checkpoint`:

```bash
python baselines/IPPO/eval_ippo_overcooked_v3.py \
  --layout splitnosig_0 \
  --checkpoint saves/splitnosig_0_cnn_baseline_seed0/ippo_cnn_overcooked_v3_splitnosig_0_seed0_vmap0.safetensors \
  --episodes 1 \
  --render \
  --render-delay 0.2
```

On a headless server, use `--gif` instead of `--render`.

The signal and transition features change the default V3 observation from 30
to 33 channels, so policies trained before these changes require the legacy
observation flag during evaluation:

```bash
python baselines/IPPO/eval_ippo_overcooked_v3.py \
  --layout splitnosig_0 \
  --checkpoint PATH_TO_OLD_CHECKPOINT.safetensors \
  --legacy-observation
```

For a recent 32-channel checkpoint with transition features but no explicit
signal channel, use `--no-signal-status`. For a 31-channel checkpoint trained
with the countdown but without the change mask or signal channel, combine
`--no-layout-change-mask --no-signal-status`.

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
