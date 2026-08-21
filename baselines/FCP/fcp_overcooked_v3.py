"""Fictitious Co-Play best-response training for Overcooked V3.

Based on the V3 IPPO trainer, with frozen population partners sampled per episode.
"""

import functools
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, NamedTuple, Sequence

import distrax
import flax.linen as nn
import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from omegaconf import OmegaConf

import jaxmarl
import wandb
from jaxmarl._env import load_project_env
from jaxmarl._experiment import experiment_folder
from jaxmarl._wandb import require_sweep_target
from jaxmarl.wrappers.baselines import LogWrapper


TRAIN_METRIC_NAMES = {
    "returned_episode_returns": "episode_return",
    "returned_episode_lengths": "episode_length",
    "returned_episode": "episode_completed",
    "original_reward": "sparse_reward",
    "shaped_reward": "shaped_reward",
    "combined_reward": "combined_reward",
    "anneal_factor": "reward_shaping_factor",
    "total_loss": "total_loss",
    "value_loss": "value_loss",
    "actor_loss": "actor_loss",
    "entropy": "entropy",
    "entropy_coef": "entropy_coefficient",
    "learning_rate": "learning_rate",
    "update_step": "update",
    "env_step": "env_step",
}

DEBUG_METRIC_NAMES = {
    "layout_index": "layout_index",
    "layout_changed": "layout_changed_fraction",
    "layout_change_events": "layout_change_events",
    "recipe_changed": "recipe_changed_fraction",
    "recipe_change_events": "recipe_change_events",
    "recipe_onion_count": "recipe_onion_count",
    "recipe_tomato_count": "recipe_tomato_count",
    "legacy_recipe_deliveries_remaining": "legacy_recipe_deliveries_remaining",
    "steps_until_layout_change": "steps_until_layout_change",
    "transition_countdown": "transition_countdown",
    "layout_change_tile_count": "layout_change_tile_count",
    "wall_tile_count": "wall_tile_count",
    "ingredient_pile_count": "ingredient_pile_count",
    "left_workload_tile_count": "left_workload_tile_count",
    "right_workload_tile_count": "right_workload_tile_count",
    "left_ingredient_pile_count": "left_ingredient_pile_count",
    "right_ingredient_pile_count": "right_ingredient_pile_count",
    "fcp_population_index": "fcp_population_index",
    "fcp_train_agent": "fcp_train_agent",
}

_FCP_CHECKPOINT_PATTERN = re.compile(
    r"^(?P<identity>ippo_(?:cnn|rnn)_overcooked_v3_.+_seed\d+_vmap\d+)"
    r"(?:_update(?P<update>\d+))?\.safetensors$"
)


def _prefixed_wandb_metrics(metric):
    """Split optimization and environment diagnostics into W&B namespaces."""
    prefixed = {
        f"train/{target}": metric[source]
        for source, target in TRAIN_METRIC_NAMES.items()
        if source in metric
    }
    prefixed.update(
        {
            f"debug/{target}": metric[source]
            for source, target in DEBUG_METRIC_NAMES.items()
            if source in metric
        }
    )
    return prefixed


def _timestamp():
    return datetime.now().strftime("%H:%M:%S")


def _resolve_wandb_mode(config, environ=None):
    """Fall back from online to offline when no API key is configured."""
    if environ is None:
        environ = os.environ
    mode = str(config.get("wandb_mode", "online")).lower()
    if mode not in {"online", "offline", "disabled"}:
        raise ValueError("wandb_mode must be online, offline, or disabled")
    if mode == "online" and not environ.get("WANDB_API_KEY", "").strip():
        return "offline"
    return mode


def _wandb_target(config, environ=None):
    """Let a sweep agent own its entity/project instead of overriding it."""
    if environ is None:
        environ = os.environ
    if environ.get("WANDB_SWEEP_ID"):
        return {}
    return {
        "entity": config.get("ENTITY") or None,
        "project": config.get("PROJECT") or None,
    }


def _architecture(config):
    architecture = config.get("ARCHITECTURE", "rnn").lower()
    if architecture not in {"cnn", "rnn"}:
        raise ValueError("ARCHITECTURE must be either 'cnn' or 'rnn'")
    return architecture


def _name_template_context(config):
    """Expose stable identifiers to configurable output-name templates."""
    fcp_config = dict(config.get("FCP") or {})
    return {
        "architecture": _architecture(config),
        "layout": str(config["ENV_KWARGS"]["layout"]),
        "partner_id": str(fcp_config.get("partner_id") or "population"),
        "partner_slug": str(fcp_config.get("partner_slug") or "population"),
        "partner_skill": str(fcp_config.get("partner_skill") or "final"),
        "partner_skill_slug": str(fcp_config.get("partner_skill_slug") or "final"),
        "seed": int(config["SEED"]),
    }


def _format_name_setting(config, key, default):
    """Format an optional naming setting without changing FCP defaults."""
    template = str(config.get(key) or default)
    try:
        return template.format(**_name_template_context(config))
    except (KeyError, ValueError) as error:
        raise ValueError(f"Invalid {key} template {template!r}: {error}") from error


def _checkpoint_prefix(config):
    configured = config.get("CHECKPOINT_PREFIX")
    if configured:
        return _format_name_setting(config, "CHECKPOINT_PREFIX", configured)
    return f"fcp_{_architecture(config)}"


def _checkpoint_update_steps(config):
    """Resolve optional fractional milestones before the final checkpoint."""
    num_updates = int(config["NUM_UPDATES"])
    fractions = config.get("CHECKPOINT_FRACTIONS") or ()
    updates = set()
    for raw_fraction in fractions:
        fraction = float(raw_fraction)
        if not 0.0 < fraction <= 1.0:
            raise ValueError(
                "CHECKPOINT_FRACTIONS values must be greater than 0 and at most 1"
            )
        update = max(1, min(num_updates, int(num_updates * fraction + 0.5)))
        if update < num_updates:
            updates.add(update)
    return tuple(sorted(updates))


def _checkpoint_path(
    config,
    save_dir,
    checkpoint_prefix,
    experiment_name,
    seed_index,
    update_step=None,
):
    suffix = ""
    if update_step is not None:
        suffix = f"_update{int(update_step):06d}"
    return Path(save_dir) / (
        f"{checkpoint_prefix}_{experiment_name}_seed{config['SEED']}_"
        f"vmap{int(seed_index)}{suffix}.safetensors"
    )


def _snapshot_sort_key(path):
    match = _FCP_CHECKPOINT_PATTERN.match(Path(path).name)
    if match is None:
        raise ValueError(f"Unsupported FCP population checkpoint name: {path}")
    update = match.group("update")
    return float("inf") if update is None else int(update)


def _evenly_spaced(items, count):
    """Select chronological snapshots while always retaining the final policy."""
    if count <= 0 or len(items) <= count:
        return list(items)
    if count == 1:
        return [items[-1]]
    indices = [i * (len(items) - 1) // (count - 1) for i in range(count)]
    return [items[index] for index in dict.fromkeys(indices)]


def discover_population_checkpoints(config):
    """Find compatible SP snapshots and select a balanced FCP population."""
    fcp_config = dict(config.get("FCP") or {})
    partner_checkpoint = fcp_config.get("partner_checkpoint")
    if partner_checkpoint:
        # PORTING NOTE: CooT trains one response against one explicit HSP/MEP
        # checkpoint. Its filenames do not follow the IPPO snapshot convention.
        checkpoint_path = Path(partner_checkpoint).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"FCP partner checkpoint not found: {checkpoint_path}"
            )
        return [checkpoint_path]

    population_dir = fcp_config.get("population_dir")
    if not population_dir:
        raise ValueError("FCP.population_dir must point to SP checkpoints")
    population_dir = Path(population_dir).expanduser().resolve()
    if not population_dir.is_dir():
        raise FileNotFoundError(f"FCP population directory not found: {population_dir}")

    architecture = _architecture(config)
    layout = config["ENV_KWARGS"]["layout"]
    expected_prefix = f"ippo_{architecture}_overcooked_v3_{layout}_seed"
    groups = {}
    for path in population_dir.rglob("*.safetensors"):
        match = _FCP_CHECKPOINT_PATTERN.match(path.name)
        if match is None or not path.name.startswith(expected_prefix):
            continue
        groups.setdefault(match.group("identity"), []).append(path)

    if not groups:
        raise FileNotFoundError(
            f"No {architecture.upper()} SP checkpoints for {layout!r} under "
            f"{population_dir}. Train the FCP population sweep first."
        )

    snapshots_per_policy = int(fcp_config.get("snapshots_per_policy", 3))
    selected = []
    for identity in sorted(groups):
        snapshots = sorted(groups[identity], key=_snapshot_sort_key)
        selected.extend(_evenly_spaced(snapshots, snapshots_per_policy))

    max_population_size = fcp_config.get("max_population_size")
    if max_population_size is not None:
        selected = _evenly_spaced(selected, int(max_population_size))
    minimum_population_size = int(fcp_config.get("minimum_population_size", 2))
    if len(selected) < minimum_population_size:
        raise ValueError(
            f"FCP population has {len(selected)} policies; expected at least "
            f"{minimum_population_size}"
        )
    return selected


def load_fcp_population(config):
    """Load selected safetensors checkpoints into one leading population axis."""
    from jaxmarl.wrappers.baselines import load_params

    checkpoint_paths = discover_population_checkpoints(config)
    policies = [load_params(path) for path in checkpoint_paths]
    reference_structure = jax.tree_util.tree_structure(policies[0])
    reference_shapes = [value.shape for value in jax.tree_util.tree_leaves(policies[0])]
    for path, policy in zip(checkpoint_paths[1:], policies[1:]):
        if jax.tree_util.tree_structure(policy) != reference_structure:
            raise ValueError(f"Incompatible parameter tree in {path}")
        shapes = [value.shape for value in jax.tree_util.tree_leaves(policy)]
        if shapes != reference_shapes:
            raise ValueError(f"Incompatible parameter shapes in {path}")
    population = jax.tree.map(lambda *values: jnp.stack(values), *policies)
    return population, checkpoint_paths


def _checkpoint_metadata(config):
    layout_name = config["ENV_KWARGS"]["layout"]
    layout_suffix = layout_name
    if config["ENV_NAME"] == "overcooked_v3":
        layout_suffix = layout_suffix.removeprefix("dynamic_")
    experiment_name = f"{config['ENV_NAME']}_{layout_suffix}"
    save_dir = os.path.join(config["SAVES_DIR"], experiment_folder(config))
    return experiment_name, save_dir


def _wandb_metadata(config):
    """Build stable W&B names while keeping Hydra overrides authoritative."""
    algorithm = str(config.get("ALGORITHM", "IPPO"))
    architecture = _architecture(config)
    layout_name = config["ENV_KWARGS"]["layout"]
    condition = layout_name
    experiment = config.get("EXPERIMENT", "overcooked_v3")
    tags = list(config.get("WANDB_TAGS") or [])
    tags.extend([algorithm, architecture.upper(), "OvercookedV3", experiment])
    tags = list(dict.fromkeys(tags))

    group = str(config.get("WANDB_GROUP") or experiment)
    group_prefix = str(config.get("WANDB_GROUP_PREFIX") or "fcp").strip("- ")
    if group_prefix and not group.casefold().startswith(group_prefix.casefold()):
        group = f"{group_prefix}-{group}"

    default_name = f"{_checkpoint_prefix(config)}_{condition}_seed{config['SEED']}"
    name = str(config.get("RUN_NAME") or default_name)
    run_prefix = str(config.get("RUN_NAME_PREFIX") or "fcp").strip("- ")
    if run_prefix and not name.casefold().startswith(run_prefix.casefold()):
        name = f"{run_prefix}-{name}"
    return name, group, tags


def _log_final_checkpoint_artifact(config, checkpoint_paths, config_path):
    """Log final checkpoints and their resolved config as one W&B artifact."""
    if not config.get("upload_final_checkpoint", False):
        return None
    if wandb.run is None:
        raise RuntimeError("upload_final_checkpoint requires an active W&B run")
    if not checkpoint_paths:
        raise RuntimeError("No final checkpoints were saved for artifact upload")

    artifact_prefix = _format_name_setting(
        config,
        "CHECKPOINT_ARTIFACT_PREFIX",
        "fcp-overcooked-v3",
    ).strip("- ")
    artifact_name = f"{artifact_prefix}-{wandb.run.id}-final-checkpoint"
    fcp_config = dict(config.get("FCP") or {})
    artifact_metadata = {
        "run_id": wandb.run.id,
        "algorithm": str(config.get("ALGORITHM", "IPPO")),
        "architecture": _architecture(config),
        "layout": config["ENV_KWARGS"]["layout"],
        "seed": int(config["SEED"]),
        "num_seeds": int(config["NUM_SEEDS"]),
        "checkpoint_format": "safetensors",
        "fcp_population_size": int(fcp_config["population_size"]),
        "fcp_population_dir": str(fcp_config.get("population_dir") or ""),
    }
    if fcp_config.get("partner_checkpoint"):
        artifact_metadata.update(
            {
                "partner_checkpoint": str(fcp_config["partner_checkpoint"]),
                "partner_sha256": str(fcp_config.get("partner_sha256") or ""),
                "partner_id": str(fcp_config.get("partner_id") or ""),
                "partner_skill": str(fcp_config.get("partner_skill") or ""),
                "partner_population_type": str(fcp_config.get("population_type") or ""),
                "partner_population_split": str(
                    fcp_config.get("population_split") or ""
                ),
                "train_agent_index": (
                    int(fcp_config["train_agent_index"])
                    if fcp_config.get("train_agent_index") is not None
                    else None
                ),
            }
        )
    artifact = wandb.Artifact(
        artifact_name,
        type="checkpoint",
        description=str(
            config.get("CHECKPOINT_ARTIFACT_DESCRIPTION")
            or "Final Overcooked V3 FCP best-response checkpoint(s)."
        ),
        metadata=artifact_metadata,
    )
    for checkpoint_path in checkpoint_paths:
        checkpoint_path = Path(checkpoint_path)
        artifact.add_file(str(checkpoint_path), name=checkpoint_path.name)
    if config_path is not None:
        config_path = Path(config_path)
        artifact.add_file(str(config_path), name=config_path.name)

    logged_artifact = wandb.run.log_artifact(artifact, aliases=["final"])
    wandb.run.summary["checkpoint/artifact_name"] = artifact_name
    wandb.run.summary["checkpoint/uploaded"] = True
    print(
        f"[{_timestamp()}] Queued final checkpoint artifact: {artifact_name}:final",
        flush=True,
    )
    return logged_artifact


def _record_final_episode(config, params, video_path):
    """Roll out the final shared policy once, save an MP4, and upload it."""
    from jaxmarl.environments.overcooked_v3.common import OvercookedActionsEnum
    from jaxmarl.viz.overcooked_v3_visualizer import OvercookedV3Visualizer

    max_steps = int(config.get("RECORD_MAX_STEPS", 400))
    fps = int(config.get("RECORD_VIDEO_FPS", 10))
    quality = int(config.get("RECORD_VIDEO_QUALITY", 5))
    if max_steps <= 0:
        raise ValueError("RECORD_MAX_STEPS must be greater than zero")
    if fps <= 0:
        raise ValueError("RECORD_VIDEO_FPS must be greater than zero")
    if not 0 <= quality <= 10:
        raise ValueError("RECORD_VIDEO_QUALITY must be between 0 and 10")

    env_kwargs = dict(config["ENV_KWARGS"])
    env_kwargs["max_steps"] = max_steps
    env = jaxmarl.make(config["ENV_NAME"], **env_kwargs)
    architecture = _architecture(config)
    network_class = ActorCriticRNN if architecture == "rnn" else ActorCriticCNN
    network = network_class(env.action_space(env.agents[0]).n, config=config)

    @jax.jit
    def select_actions(params, hidden, obs, dones):
        hidden, pi, _ = network.apply(params, hidden, (obs, dones))
        return hidden, pi.mode()

    env_step = jax.jit(env.step_env)
    key = jax.random.PRNGKey(int(config["SEED"]) + 1_000_000)
    key, reset_key = jax.random.split(key)
    obs, state = env.reset(reset_key)
    hidden = ScannedRNN.initialize_carry(env.num_agents, config["GRU_HIDDEN_DIM"])
    last_done = jnp.zeros((env.num_agents,), dtype=jnp.bool_)

    states = [jax.device_get(state)]
    captions = ["step=0 score=0 actions=-/-"]
    episode_return = 0.0
    episode_length = 0

    for step in range(max_steps):
        obs_batch = jnp.stack([obs[agent] for agent in env.agents])
        hidden, actions = select_actions(
            params,
            hidden,
            obs_batch[jnp.newaxis, :],
            last_done[jnp.newaxis, :],
        )
        actions = actions.squeeze(0)
        env_actions = {agent: actions[index] for index, agent in enumerate(env.agents)}
        key, step_key = jax.random.split(key)
        obs, state, rewards, dones, _ = env_step(step_key, state, env_actions)

        episode_return += float(rewards[env.agents[0]])
        episode_length = step + 1
        action_names = [
            OvercookedActionsEnum(int(actions[index])).name
            for index in range(env.num_agents)
        ]
        states.append(jax.device_get(state))
        captions.append(
            f"step={episode_length} score={episode_return:g} "
            f"actions={'/'.join(action_names)}"
        )
        last_done = jnp.asarray([dones[agent] for agent in env.agents])
        if bool(dones["__all__"]):
            break

    video_path = Path(video_path)
    video_path.parent.mkdir(parents=True, exist_ok=True)
    visualizer = OvercookedV3Visualizer(
        tile_size=24,
        seconds_per_step=1.0 / fps,
        transition_warning_steps=env.transition_warning_steps,
    )
    visualizer.save_video(
        states,
        filename=str(video_path),
        agent_view_size=env.agent_view_size,
        captions=captions,
        fps=fps,
        quality=quality,
    )

    layout = config["ENV_KWARGS"]["layout"]
    wandb.log(
        {
            "visualization/final_episode": wandb.Video(
                str(video_path),
                format="mp4",
                caption=(
                    f"{layout} | seed={config['SEED']} | "
                    f"return={episode_return:g} | length={episode_length}"
                ),
            ),
            "eval/final_episode_return": episode_return,
            "eval/final_episode_length": episode_length,
        }
    )
    print(
        f"[{_timestamp()}] Saved and logged final episode: {video_path} "
        f"(return={episode_return:.2f}, length={episode_length})",
        flush=True,
    )
    return video_path, episode_return, episode_length


class ScannedRNN(nn.Module):
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        """Applies the module."""
        rnn_state = carry
        ins, resets = x

        new_carry = self.initialize_carry(ins.shape[0], ins.shape[1])

        rnn_state = jnp.where(
            resets[:, np.newaxis],
            new_carry,
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell(features=ins.shape[1])(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        # Use a dummy key since the default state init fn is just zeros.
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))


class CNN(nn.Module):
    output_size: int = 64
    activation: Callable[..., Any] = nn.relu

    @nn.compact
    def __call__(self, x, train=False):
        x = nn.Conv(
            features=128,
            kernel_size=(1, 1),
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = self.activation(x)
        x = nn.Conv(
            features=128,
            kernel_size=(1, 1),
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = self.activation(x)
        x = nn.Conv(
            features=8,
            kernel_size=(1, 1),
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = self.activation(x)

        x = nn.Conv(
            features=16,
            kernel_size=(3, 3),
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = self.activation(x)

        x = nn.Conv(
            features=32,
            kernel_size=(3, 3),
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = self.activation(x)

        x = nn.Conv(
            features=32,
            kernel_size=(3, 3),
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = self.activation(x)

        x = x.reshape((x.shape[0], -1))

        x = nn.Dense(
            features=self.output_size,
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = self.activation(x)

        return x


class ActorCriticRNN(nn.Module):
    action_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x

        embedding = obs

        if self.config["ACTIVATION"] == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        embed_model = CNN(
            output_size=self.config["GRU_HIDDEN_DIM"],
            activation=activation,
        )
        embedding = jax.vmap(embed_model)(embedding)

        embedding = nn.LayerNorm()(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        actor_mean = nn.Dense(
            self.config["FC_DIM_SIZE"],
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )(embedding)
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)

        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            self.config["FC_DIM_SIZE"],
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )(embedding)
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return hidden, pi, jnp.squeeze(critic, axis=-1)


class ActorCriticCNN(nn.Module):
    """Feed-forward CNN policy with the same call signature as the RNN policy."""

    action_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        obs, _dones = x

        if self.config["ACTIVATION"] == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        embedding = jax.vmap(
            CNN(
                output_size=self.config["FC_DIM_SIZE"],
                activation=activation,
            )
        )(obs)

        actor = nn.Dense(
            self.config["FC_DIM_SIZE"],
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )(embedding)
        actor = activation(actor)
        logits = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
        )(actor)
        pi = distrax.Categorical(logits=logits)

        critic = nn.Dense(
            self.config["FC_DIM_SIZE"],
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )(embedding)
        critic = activation(critic)
        critic = nn.Dense(
            1,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
        )(critic)

        return hidden, pi, jnp.squeeze(critic, axis=-1)


class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        embedding = CNN(self.activation)(x)

        actor_mean = nn.Dense(
            128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(embedding)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(embedding)
        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(embedding)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)


class Transition(NamedTuple):
    global_done: jnp.ndarray
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray
    train_mask: jnp.ndarray


def _masked_mean(values, mask):
    mask = mask.astype(values.dtype)
    count = jnp.maximum(mask.sum(), 1.0)
    return (values * mask).sum() / count


def _masked_normalize(values, mask):
    mean = _masked_mean(values, mask)
    variance = _masked_mean(jnp.square(values - mean), mask)
    return (values - mean) / jnp.sqrt(variance + 1e-8)


def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


def make_train(config):
    env_kwargs = dict(config["ENV_KWARGS"])
    env = jaxmarl.make(config["ENV_NAME"], **env_kwargs)
    architecture = _architecture(config)
    checkpoint_prefix = _checkpoint_prefix(config)
    fcp_config = dict(config.get("FCP") or {})
    population_size = int(fcp_config["population_size"])
    if population_size < 1:
        raise ValueError("FCP.population_size must be positive")

    fixed_train_agent = fcp_config.get("train_agent_index")
    if fixed_train_agent is not None:
        fixed_train_agent = int(fixed_train_agent)
        if not 0 <= fixed_train_agent < env.num_agents:
            raise ValueError(
                "FCP.train_agent_index must be null or a valid environment agent index"
            )

    value_loss_type = str(config.get("VALUE_LOSS", "mse")).lower()
    if value_loss_type not in {"mse", "huber"}:
        raise ValueError("VALUE_LOSS must be either 'mse' or 'huber'")
    value_normalization = str(config.get("VALUE_NORMALIZATION", "none")).lower()
    if value_normalization != "none":
        raise ValueError(
            "This V3 FCP trainer currently supports VALUE_NORMALIZATION=none only"
        )
    # PORTING NOTE: ZSC-Eval's adaptive BR runner enables a running ValueNorm
    # module by default. The existing V3 actor-critic TrainState has no
    # corresponding mutable normalization state, so CooT-BR explicitly uses
    # raw sparse-return targets and records this deviation in its config/result.
    huber_delta = float(config.get("HUBER_DELTA", 10.0))
    if huber_delta <= 0:
        raise ValueError("HUBER_DELTA must be positive")

    entropy_coefs_config = config.get("ENT_COEFS")
    entropy_horizons_config = config.get("ENT_COEF_HORIZONS")
    if (entropy_coefs_config is None) != (entropy_horizons_config is None):
        raise ValueError(
            "ENT_COEFS and ENT_COEF_HORIZONS must either both be set or both be omitted"
        )
    if entropy_coefs_config is None:
        constant_entropy_coef = float(config["ENT_COEF"])

        def entropy_coef_at(_timestep):
            return jnp.asarray(constant_entropy_coef, dtype=jnp.float32)

    else:
        entropy_coefs = tuple(float(value) for value in entropy_coefs_config)
        entropy_horizons = tuple(float(value) for value in entropy_horizons_config)
        if not entropy_coefs:
            raise ValueError("ENT_COEFS must contain at least one coefficient")
        if len(entropy_coefs) != len(entropy_horizons):
            raise ValueError(
                "ENT_COEFS and ENT_COEF_HORIZONS must have the same length"
            )
        if not all(np.isfinite(value) and value >= 0 for value in entropy_coefs):
            raise ValueError("ENT_COEFS values must be finite and non-negative")
        if not all(np.isfinite(value) and value >= 0 for value in entropy_horizons):
            raise ValueError("ENT_COEF_HORIZONS values must be finite and non-negative")
        if any(
            right <= left for left, right in zip(entropy_horizons, entropy_horizons[1:])
        ):
            raise ValueError("ENT_COEF_HORIZONS values must be strictly increasing")
        entropy_coef_points = jnp.asarray(entropy_coefs, dtype=jnp.float32)
        entropy_horizon_points = jnp.asarray(entropy_horizons, dtype=jnp.float32)

        def entropy_coef_at(timestep):
            # The supplementary BR trainer linearly interpolates coefficients
            # at environment-step anchors and holds the endpoint values outside.
            return jnp.interp(
                timestep,
                entropy_horizon_points,
                entropy_coef_points,
            )

    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    checkpoint_interval = int(config.get("CHECKPOINT_INTERVAL", 0))
    if checkpoint_interval < 0:
        raise ValueError("CHECKPOINT_INTERVAL must be greater than or equal to 0")
    checkpoint_updates = _checkpoint_update_steps(config)
    checkpoint_enabled = bool(checkpoint_interval or checkpoint_updates) and (
        config.get("SAVES_DIR") is not None
    )
    if checkpoint_enabled:
        experiment_name, save_dir = _checkpoint_metadata(config)

        def save_intermediate_checkpoint(params, update_step, seed_index):
            from jaxmarl.wrappers.baselines import save_params

            checkpoint_path = _checkpoint_path(
                config,
                save_dir,
                checkpoint_prefix,
                experiment_name,
                int(seed_index),
                int(update_step),
            )
            save_params(params, checkpoint_path)
            print(
                f"[{_timestamp()}] Saved intermediate checkpoint: {checkpoint_path}",
                flush=True,
            )

    env = LogWrapper(env, replace_info=False)

    def create_learning_rate_fn():
        base_learning_rate = config["LR"]

        lr_warmup = config["LR_WARMUP"]
        update_steps = config["NUM_UPDATES"]
        warmup_steps = int(lr_warmup * update_steps)

        steps_per_epoch = config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]

        warmup_fn = optax.linear_schedule(
            init_value=0.0,
            end_value=base_learning_rate,
            transition_steps=warmup_steps * steps_per_epoch,
        )
        cosine_epochs = max(update_steps - warmup_steps, 1)

        print(f"[{_timestamp()}] Update steps: {update_steps}")
        print(f"[{_timestamp()}] Warmup epochs: {warmup_steps}")
        print(f"[{_timestamp()}] Cosine epochs: {cosine_epochs}")

        cosine_fn = optax.cosine_decay_schedule(
            init_value=base_learning_rate, decay_steps=cosine_epochs * steps_per_epoch
        )
        schedule_fn = optax.join_schedules(
            schedules=[warmup_fn, cosine_fn],
            boundaries=[warmup_steps * steps_per_epoch],
        )
        return schedule_fn

    rew_shaping_horizon = int(config["REW_SHAPING_HORIZON"])
    if rew_shaping_horizon < 0:
        raise ValueError("REW_SHAPING_HORIZON must be greater than or equal to 0")
    if rew_shaping_horizon == 0:
        # PORTING NOTE: the released partner-specific BR uses sparse reward only.
        # Avoid passing transition_steps=0 to Optax, and make that intent explicit.
        def rew_shaping_anneal(_timestep):
            return jnp.asarray(0.0, dtype=jnp.float32)

    else:
        rew_shaping_anneal = optax.linear_schedule(
            init_value=1.0,
            end_value=0.0,
            transition_steps=rew_shaping_horizon,
        )

    def train(rng, seed_index, population_params):
        # INIT NETWORK
        population_params = jax.tree.map(jax.lax.stop_gradient, population_params)
        network_class = ActorCriticRNN if architecture == "rnn" else ActorCriticCNN
        network = network_class(env.action_space(env.agents[0]).n, config=config)

        rng, _rng = jax.random.split(rng)
        init_x = (
            jnp.zeros(
                (
                    1,
                    config["NUM_ENVS"],
                    *env.observation_space(env.agents[0]).shape,
                )
            ),
            jnp.zeros((1, config["NUM_ENVS"])),
        )
        init_hstate = ScannedRNN.initialize_carry(
            config["NUM_ENVS"], config["GRU_HIDDEN_DIM"]
        )

        network_params = network.init(_rng, init_hstate, init_x)
        population_leaves = jax.tree_util.tree_leaves(population_params)
        network_leaves = jax.tree_util.tree_leaves(network_params)
        if len(population_leaves) != len(network_leaves):
            raise ValueError("FCP population parameter tree does not match the actor")
        for population_leaf, network_leaf in zip(population_leaves, network_leaves):
            if population_leaf.shape != (population_size, *network_leaf.shape):
                raise ValueError(
                    "FCP population parameter shapes do not match the configured actor"
                )
        if config["ANNEAL_LR"]:
            learning_rate_fn = create_learning_rate_fn()
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate_fn, eps=1e-5),
            )
        else:
            learning_rate_fn = optax.constant_schedule(config["LR"])
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate_fn, eps=1e-5),
            )
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)
        init_hstate = ScannedRNN.initialize_carry(
            config["NUM_ACTORS"], config["GRU_HIDDEN_DIM"]
        )
        init_population_hstate = ScannedRNN.initialize_carry(
            config["NUM_ACTORS"], config["GRU_HIDDEN_DIM"]
        )
        rng, train_agent_rng, population_rng = jax.random.split(rng, 3)
        if fixed_train_agent is None:
            init_train_agent_indices = jax.random.randint(
                train_agent_rng,
                (config["NUM_ENVS"],),
                0,
                env.num_agents,
            )
        else:
            init_train_agent_indices = jnp.full(
                (config["NUM_ENVS"],),
                fixed_train_agent,
                dtype=jnp.int32,
            )
        init_population_indices = jax.random.randint(
            population_rng,
            (config["NUM_ENVS"],),
            0,
            population_size,
        )

        def _make_train_mask(train_agent_indices):
            agent_indices = jnp.arange(env.num_agents)[:, jnp.newaxis]
            return (agent_indices == train_agent_indices[jnp.newaxis, :]).reshape(-1)

        def _compute_population_actions(
            population_indices,
            population_hstate,
            obs_batch,
            last_done,
            action_rng,
        ):
            actor_population_indices = jnp.tile(population_indices, env.num_agents)
            action_rngs = jax.random.split(action_rng, config["NUM_ACTORS"])

            def _compute_one(policy_index, hidden, obs, done, key):
                params = jax.tree.map(
                    lambda values: values[policy_index], population_params
                )
                next_hidden, pi, _ = network.apply(
                    params,
                    hidden[jnp.newaxis, :],
                    (
                        obs[jnp.newaxis, jnp.newaxis, ...],
                        done[jnp.newaxis, jnp.newaxis],
                    ),
                )
                return next_hidden.squeeze(0), pi.sample(seed=key).squeeze()

            return jax.vmap(_compute_one)(
                actor_population_indices,
                population_hstate,
                obs_batch,
                last_done,
                action_rngs,
            )

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                (
                    train_state,
                    env_state,
                    last_obs,
                    last_done,
                    update_step,
                    hstate,
                    population_hstate,
                    train_agent_indices,
                    population_indices,
                    rng,
                ) = runner_state

                # SELECT ACTION
                rng, policy_rng, population_action_rng = jax.random.split(rng, 3)

                # obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
                obs_batch = jnp.stack([last_obs[a] for a in env.agents]).reshape(
                    -1, *env.observation_space(env.agents[0]).shape
                )
                ac_in = (
                    obs_batch[np.newaxis, :],
                    last_done[np.newaxis, :],
                )

                hstate, pi, value = network.apply(train_state.params, hstate, ac_in)
                policy_action = pi.sample(seed=policy_rng).squeeze(0)
                log_prob = pi.log_prob(policy_action[jnp.newaxis, :]).squeeze(0)
                population_hstate, population_action = _compute_population_actions(
                    population_indices,
                    population_hstate,
                    obs_batch,
                    last_done,
                    population_action_rng,
                )
                train_mask = _make_train_mask(train_agent_indices)
                action = jnp.where(train_mask, policy_action, population_action)
                env_act = unbatchify(
                    action, env.agents, config["NUM_ENVS"], env.num_agents
                )

                env_act = {k: v.flatten() for k, v in env_act.items()}

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])

                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0)
                )(rng_step, env_state, env_act)
                original_reward = jnp.array([reward[a] for a in env.agents])

                current_timestep = (
                    update_step * config["NUM_STEPS"] * config["NUM_ENVS"]
                )
                anneal_factor = rew_shaping_anneal(current_timestep)
                reward = jax.tree.map(
                    lambda x, y: x + y * anneal_factor, reward, info["shaped_reward"]
                )

                shaped_reward = jnp.array(
                    [info["shaped_reward"][a] for a in env.agents]
                )
                combined_reward = jnp.array([reward[a] for a in env.agents])

                info["shaped_reward"] = shaped_reward
                info["original_reward"] = original_reward
                info["anneal_factor"] = jnp.full_like(shaped_reward, anneal_factor)
                info["combined_reward"] = combined_reward
                info["fcp_population_index"] = jnp.tile(
                    population_indices, (env.num_agents, 1)
                )
                info["fcp_train_agent"] = jnp.tile(
                    train_agent_indices, (env.num_agents, 1)
                )

                # HSP population training exposes a structured event vector.
                # The standard FCP/BR logger only consumes scalar diagnostics,
                # and cannot flatten that extra feature into one actor axis.
                info.pop("event_vector", None)
                info = jax.tree.map(lambda x: x.reshape((config["NUM_ACTORS"])), info)
                done_batch = batchify(done, env.agents, config["NUM_ACTORS"]).squeeze()
                rng, train_agent_rng, population_rng = jax.random.split(rng, 3)
                if fixed_train_agent is None:
                    sampled_train_agents = jax.random.randint(
                        train_agent_rng,
                        (config["NUM_ENVS"],),
                        0,
                        env.num_agents,
                    )
                else:
                    sampled_train_agents = jnp.full(
                        (config["NUM_ENVS"],),
                        fixed_train_agent,
                        dtype=jnp.int32,
                    )
                sampled_population = jax.random.randint(
                    population_rng,
                    (config["NUM_ENVS"],),
                    0,
                    population_size,
                )
                next_train_agent_indices = jnp.where(
                    done["__all__"], sampled_train_agents, train_agent_indices
                )
                next_population_indices = jnp.where(
                    done["__all__"], sampled_population, population_indices
                )
                transition = Transition(
                    jnp.tile(done["__all__"], env.num_agents),
                    last_done,
                    action,
                    value.squeeze(),
                    batchify(reward, env.agents, config["NUM_ACTORS"]).squeeze(),
                    log_prob,
                    obs_batch,
                    info,
                    train_mask,
                )
                runner_state = (
                    train_state,
                    env_state,
                    obsv,
                    done_batch,
                    update_step,
                    hstate,
                    population_hstate,
                    next_train_agent_indices,
                    next_population_indices,
                    rng,
                )
                return runner_state, transition

            initial_hstate = runner_state[5]
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            (
                train_state,
                env_state,
                last_obs,
                last_done,
                update_step,
                hstate,
                population_hstate,
                train_agent_indices,
                population_indices,
                rng,
            ) = runner_state
            last_obs_batch = jnp.stack([last_obs[a] for a in env.agents]).reshape(
                -1, *env.observation_space(env.agents[0]).shape
            )
            ac_in = (
                last_obs_batch[np.newaxis, :],
                last_done[np.newaxis, :],
            )
            _, _, last_val = network.apply(train_state.params, hstate, ac_in)
            last_val = last_val.squeeze()

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.global_done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # Match ZSC-Eval's partner-specific BR runner: it updates entropy
            # after collecting a rollout, using the completed environment-step
            # count rather than the optimizer/minibatch step.
            entropy_env_step = (
                (update_step + 1) * config["NUM_STEPS"] * config["NUM_ENVS"]
            )
            entropy_coef = entropy_coef_at(entropy_env_step)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    init_hstate, traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, init_hstate, traj_batch, gae, targets):
                        train_mask = jax.lax.stop_gradient(traj_batch.train_mask)
                        # RERUN NETWORK
                        _, pi, value = network.apply(
                            params,
                            init_hstate.squeeze(),
                            (traj_batch.obs, traj_batch.done),
                        )

                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        if value_loss_type == "huber":
                            value_losses = optax.huber_loss(
                                value,
                                targets,
                                delta=huber_delta,
                            )
                            value_losses_clipped = optax.huber_loss(
                                value_pred_clipped,
                                targets,
                                delta=huber_delta,
                            )
                            value_loss_scale = 1.0
                        else:
                            value_losses = jnp.square(value - targets)
                            value_losses_clipped = jnp.square(
                                value_pred_clipped - targets
                            )
                            value_loss_scale = 0.5
                        value_loss = value_loss_scale * _masked_mean(
                            jnp.maximum(value_losses, value_losses_clipped),
                            train_mask,
                        )

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = _masked_normalize(gae, train_mask)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = _masked_mean(loss_actor, train_mask)
                        entropy = _masked_mean(pi.entropy(), train_mask)

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - entropy_coef * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, init_hstate, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, init_hstate, traj_batch, advantages, targets, rng = (
                    update_state
                )
                rng, _rng = jax.random.split(rng)

                init_hstate = jnp.reshape(init_hstate, (1, config["NUM_ACTORS"], -1))
                batch = (
                    init_hstate,
                    traj_batch,
                    # GAE already returns [rollout, actor]. Preserve both axes
                    # when either dimension is one in a smoke run.
                    advantages,
                    targets,
                )
                permutation = jax.random.permutation(_rng, config["NUM_ACTORS"])

                shuffled_batch = jax.tree.map(
                    lambda x: jnp.take(x, permutation, axis=1), batch
                )

                minibatches = jax.tree.map(
                    lambda x: jnp.swapaxes(
                        jnp.reshape(
                            x,
                            [x.shape[0], config["NUM_MINIBATCHES"], -1]
                            + list(x.shape[2:]),
                        ),
                        1,
                        0,
                    ),
                    shuffled_batch,
                )

                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (
                    train_state,
                    init_hstate.squeeze(),
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                )
                return update_state, total_loss

            update_state = (
                train_state,
                initial_hstate,
                traj_batch,
                advantages,
                targets,
                rng,
            )
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            total_loss, (value_loss, actor_loss, entropy) = loss_info
            metric = {
                **traj_batch.info,
                "layout_index": traj_batch.info["layout_index"][-1],
                "recipe_changed": traj_batch.info["recipe_changed"][-1],
                "recipe_onion_count": traj_batch.info["recipe_onion_count"][-1],
                "recipe_tomato_count": traj_batch.info["recipe_tomato_count"][-1],
                "legacy_recipe_deliveries_remaining": traj_batch.info[
                    "legacy_recipe_deliveries_remaining"
                ][-1],
                "steps_until_layout_change": traj_batch.info[
                    "steps_until_layout_change"
                ][-1],
                "transition_countdown": traj_batch.info["transition_countdown"][-1],
                "layout_change_tile_count": traj_batch.info["layout_change_tile_count"][
                    -1
                ],
                "wall_tile_count": traj_batch.info["wall_tile_count"][-1],
                "ingredient_pile_count": traj_batch.info["ingredient_pile_count"][-1],
                "left_workload_tile_count": traj_batch.info["left_workload_tile_count"][
                    -1
                ],
                "right_workload_tile_count": traj_batch.info[
                    "right_workload_tile_count"
                ][-1],
                "left_ingredient_pile_count": traj_batch.info[
                    "left_ingredient_pile_count"
                ][-1],
                "right_ingredient_pile_count": traj_batch.info[
                    "right_ingredient_pile_count"
                ][-1],
                "total_loss": total_loss,
                "value_loss": value_loss,
                "actor_loss": actor_loss,
                "entropy": entropy,
                "entropy_coef": entropy_coef,
                "learning_rate": learning_rate_fn(
                    update_step * config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]
                ),
                "layout_change_events": (
                    traj_batch.info["layout_changed"].sum() / env.num_agents
                ),
                "recipe_change_events": (
                    traj_batch.info["recipe_changed"].sum() / env.num_agents
                ),
            }
            rng = update_state[-1]

            def callback(metric):
                wandb.log(_prefixed_wandb_metrics(metric))
                update = int(metric["update_step"])
                log_interval = int(config.get("LOG_INTERVAL", 10))
                if (
                    update == 1
                    or update % log_interval == 0
                    or update == config["NUM_UPDATES"]
                ):
                    env_step = int(metric["env_step"])
                    progress = 100.0 * update / config["NUM_UPDATES"]
                    sparse_episode_return = float(metric["returned_episode_returns"])
                    sparse_step_reward = float(metric["original_reward"])
                    print(
                        f"[{_timestamp()}] "
                        f"update={update}/{config['NUM_UPDATES']} "
                        f"env_step={env_step} progress={progress:.1f}% "
                        f"entropy_coef={float(metric['entropy_coef']):.5f} "
                        f"sparse_episode_return={sparse_episode_return:.2f} "
                        f"sparse_step_reward={sparse_step_reward:.4f}",
                        flush=True,
                    )

            update_step = update_step + 1
            metric = jax.tree.map(lambda x: jnp.asarray(x).mean(), metric)
            metric["update_step"] = update_step
            metric["env_step"] = update_step * config["NUM_STEPS"] * config["NUM_ENVS"]
            jax.debug.callback(callback, metric)

            if checkpoint_enabled:
                interval_due = (
                    update_step % checkpoint_interval == 0
                    if checkpoint_interval
                    else jnp.bool_(False)
                )
                fraction_due = (
                    jnp.any(update_step == jnp.asarray(checkpoint_updates))
                    if checkpoint_updates
                    else jnp.bool_(False)
                )
                should_save = jnp.logical_and(
                    jnp.logical_or(interval_due, fraction_due),
                    update_step < config["NUM_UPDATES"],
                )

                def checkpoint_branch(_):
                    jax.debug.callback(
                        save_intermediate_checkpoint,
                        train_state.params,
                        update_step,
                        seed_index,
                        ordered=True,
                    )
                    return jnp.int32(0)

                jax.lax.cond(
                    should_save,
                    checkpoint_branch,
                    lambda _: jnp.int32(0),
                    operand=None,
                )

            runner_state = (
                train_state,
                env_state,
                last_obs,
                last_done,
                update_step,
                hstate,
                population_hstate,
                train_agent_indices,
                population_indices,
                rng,
            )
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (
            train_state,
            env_state,
            obsv,
            jnp.zeros((config["NUM_ACTORS"]), dtype=bool),
            0,
            init_hstate,
            init_population_hstate,
            init_train_agent_indices,
            init_population_indices,
            _rng,
        )
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metric}

    return train


def run(config):
    config = OmegaConf.to_container(config, resolve=True)
    requested_wandb_mode = str(config.get("wandb_mode", "online")).lower()
    config["wandb_mode"] = _resolve_wandb_mode(config)
    if requested_wandb_mode == "online" and config["wandb_mode"] == "offline":
        print(
            f"[{_timestamp()}] WANDB_API_KEY is not set; using offline W&B mode",
            flush=True,
        )

    layout_name = config["ENV_KWARGS"]["layout"]
    num_seeds = config["NUM_SEEDS"]
    architecture = _architecture(config)
    checkpoint_prefix = _checkpoint_prefix(config)
    population_params, population_checkpoints = load_fcp_population(config)
    config["FCP"]["population_size"] = len(population_checkpoints)
    config["FCP"]["selected_checkpoints"] = [
        str(path) for path in population_checkpoints
    ]
    print(
        f"[{_timestamp()}] Loaded {len(population_checkpoints)} frozen "
        f"{architecture.upper()} partner policies for {layout_name}",
        flush=True,
    )
    save_dir = None
    config_path = None

    upload_requested = bool(config.get("upload_final_checkpoint", True))
    wandb_enabled = str(config.get("wandb_mode", "disabled")).lower() != "disabled"
    upload_final_checkpoint = upload_requested and wandb_enabled
    if upload_final_checkpoint and config.get("SAVES_DIR") is None:
        raise ValueError("upload_final_checkpoint requires SAVES_DIR")

    if config.get("SAVES_DIR") is not None:
        experiment_name, save_dir = _checkpoint_metadata(config)
        os.makedirs(save_dir, exist_ok=True)
        config_path = os.path.join(
            save_dir,
            f"{checkpoint_prefix}_{experiment_name}_seed{config['SEED']}_config.yaml",
        )
        OmegaConf.save(OmegaConf.create(config), config_path)

    wandb_name, wandb_group, wandb_tags = _wandb_metadata(config)
    wandb_run = wandb.init(
        **_wandb_target(config),
        tags=wandb_tags,
        config=config,
        mode=config["wandb_mode"],
        name=wandb_name,
        group=wandb_group,
        job_type="train",
        notes=config.get("NOTES"),
    )
    if str(config.get("ALGORITHM")) == "CooT-BR":
        require_sweep_target(wandb_run, config)
    wandb.define_metric("train/env_step")
    wandb.define_metric("train/*", step_metric="train/env_step")
    wandb.define_metric("debug/*", step_metric="train/env_step")
    wandb.define_metric("eval/*")

    with jax.disable_jit(False):
        rng = jax.random.PRNGKey(config["SEED"])
        rngs = jax.random.split(rng, num_seeds)
        seed_indices = jnp.arange(num_seeds)
        train_jit = jax.jit(make_train(config))
        train_vmap = jax.vmap(train_jit, in_axes=(0, 0, None))
        out = jax.block_until_ready(train_vmap(rngs, seed_indices, population_params))

    model_state = out["runner_state"][0]
    checkpoint_paths = []
    if save_dir is not None:
        from jaxmarl.wrappers.baselines import save_params

        for i in range(num_seeds):
            params = jax.tree.map(lambda x: x[i], model_state.params)
            checkpoint_path = _checkpoint_path(
                config,
                save_dir,
                checkpoint_prefix,
                experiment_name,
                i,
            )
            save_params(params, checkpoint_path)
            checkpoint_paths.append(checkpoint_path)
            print(f"[{_timestamp()}] Saved checkpoint: {checkpoint_path}")

    if upload_final_checkpoint:
        artifact_checkpoint_paths = list(checkpoint_paths)
        if save_dir is not None and config.get(
            "UPLOAD_INTERMEDIATE_CHECKPOINTS", False
        ):
            intermediate_paths = [
                _checkpoint_path(
                    config,
                    save_dir,
                    checkpoint_prefix,
                    experiment_name,
                    seed_index,
                    update_step,
                )
                for seed_index in range(num_seeds)
                for update_step in _checkpoint_update_steps(config)
            ]
            missing_paths = [path for path in intermediate_paths if not path.is_file()]
            if missing_paths:
                raise FileNotFoundError(
                    "Expected intermediate checkpoint(s) were not saved: "
                    + ", ".join(str(path) for path in missing_paths)
                )
            artifact_checkpoint_paths = [
                *intermediate_paths,
                *artifact_checkpoint_paths,
            ]
        _log_final_checkpoint_artifact(
            config,
            artifact_checkpoint_paths,
            config_path,
        )

    recording_enabled = bool(config.get("RECORD_FINAL_EPISODE", True))
    if recording_enabled and wandb_enabled:
        params = jax.tree.map(lambda x: x[0], model_state.params)
        video_filename = (
            f"{checkpoint_prefix}_{layout_name}_seed{config['SEED']}_"
            "vmap0_final_episode.mp4"
        )
        try:
            if save_dir is not None:
                _record_final_episode(
                    config,
                    params,
                    Path(save_dir) / video_filename,
                )
            else:
                with tempfile.TemporaryDirectory(
                    prefix="overcooked-v3-final-episode-"
                ) as temp_dir:
                    _record_final_episode(
                        config,
                        params,
                        Path(temp_dir) / video_filename,
                    )
        except Exception as error:
            print(
                f"[{_timestamp()}] WARNING: final episode recording failed: {error}",
                flush=True,
            )
            wandb.log({"debug/final_video_failed": 1})
            if wandb.run is not None:
                wandb.run.summary["visualization/final_episode_error"] = str(error)

    run_result = {
        "run_id": str(getattr(wandb_run, "id", "local")),
        "save_dir": str(save_dir) if save_dir is not None else None,
        "config_path": str(config_path) if config_path is not None else None,
        "checkpoint_paths": [str(path) for path in checkpoint_paths],
        "population_checkpoints": [str(path) for path in population_checkpoints],
    }
    # CooT's thin response wrapper writes and uploads an immutable job-result
    # sidecar after the generic trainer has saved its checkpoint. Generic FCP
    # runs retain the original ownership/lifecycle by default.
    if not config.get("DEFER_WANDB_FINISH", False):
        wandb.finish()
    return run_result


@hydra.main(
    version_base=None, config_path="../../conf", config_name="fcp_overcooked_v3"
)
def main(config):
    run(config)


def entrypoint():
    if load_project_env():
        print(f"[{_timestamp()}] Loaded project .env")
    main()


if __name__ == "__main__":
    entrypoint()
