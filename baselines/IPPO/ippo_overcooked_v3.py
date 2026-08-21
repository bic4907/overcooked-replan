import functools
import json
import os
import sys
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

try:
    from baselines.CooT.hsp_population import (
        apply_candidate_rewards,
        resolve_hsp_config,
    )
except ModuleNotFoundError as error:  # Direct script execution from baselines/IPPO.
    if error.name != "baselines":
        raise
    coot_dir = Path(__file__).resolve().parents[1] / "CooT"
    sys.path.insert(0, str(coot_dir))
    from hsp_population import apply_candidate_rewards, resolve_hsp_config
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
    "hsp_event_reward": "hsp_event_reward",
    "hsp_training_reward": "hsp_training_reward",
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
}


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


def _resolve_entropy_schedule(config):
    """Validate an optional piecewise-linear entropy schedule.

    Existing IPPO configs only define ENT_COEF and retain their exact constant
    behavior. ENT_COEFS and ENT_COEF_HORIZONS must be supplied together to opt
    into the supplementary HSP schedule.
    """

    raw_coefs = config.get("ENT_COEFS")
    raw_horizons = config.get("ENT_COEF_HORIZONS")
    if raw_coefs is None and raw_horizons is None:
        return None
    if raw_coefs is None or raw_horizons is None:
        raise ValueError("ENT_COEFS and ENT_COEF_HORIZONS must be configured together")
    coefs = tuple(float(value) for value in raw_coefs)
    horizons = tuple(int(value) for value in raw_horizons)
    if len(coefs) < 2 or len(coefs) != len(horizons):
        raise ValueError(
            "ENT_COEFS and ENT_COEF_HORIZONS must have the same length >= 2"
        )
    if horizons[0] != 0 or any(
        right <= left for left, right in zip(horizons, horizons[1:])
    ):
        raise ValueError("ENT_COEF_HORIZONS must start at 0 and be strictly increasing")
    if any(not np.isfinite(coef) or coef < 0.0 for coef in coefs):
        raise ValueError("ENT_COEFS values must be finite and non-negative")
    return coefs, horizons


def _checkpoint_prefix(config):
    architecture = _architecture(config)
    candidate = resolve_hsp_config(config)
    if candidate is not None:
        return (
            f"hsp_{candidate.profile}_candidate{candidate.candidate_id:04d}_"
            f"ippo_{architecture}"
        )
    return f"ippo_{architecture}"


def _isolate_hsp_output(config, candidate):
    """Give each HSP candidate a collision-free local experiment folder."""
    if candidate is None:
        return config
    base_folder = str(config.get("EXPERIMENT_FOLDER") or "hsp_population")
    candidate_suffix = f"hsp_{candidate.profile}_candidate{candidate.candidate_id:04d}"
    if not base_folder.endswith(candidate_suffix):
        config["EXPERIMENT_FOLDER"] = f"{base_folder}_{candidate_suffix}"
    return config


def _checkpoint_update_steps(config):
    """Resolve fractional checkpoint milestones before the final checkpoint."""
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
    default_name = f"{_checkpoint_prefix(config)}_{condition}_seed{config['SEED']}"
    name = str(config.get("RUN_NAME") or default_name)
    candidate = resolve_hsp_config(config)
    if candidate is not None:
        tags = list(
            dict.fromkeys(
                [
                    *tags,
                    "CooT-Population",
                    "HSP",
                    f"HSP-{candidate.profile}",
                    f"HSP-candidate-{candidate.candidate_id:04d}",
                ]
            )
        )
        scenario_group = str(config.get("WANDB_GROUP") or experiment)
        group = f"coot-hsp-{candidate.profile}-{scenario_group}"
    saves_dir = Path(str(config.get("SAVES_DIR") or ""))
    if saves_dir.name.casefold().replace("-", "_") == "fcp_population":
        tags = list(dict.fromkeys([*tags, "FCP-Self-Play"]))
        group = f"fcp-self-play-{group}"
        name = f"fcp-self-play-{name}"
    return name, group, tags


def _log_final_checkpoint_artifact(
    config, checkpoint_paths, config_path, extra_files=()
):
    """Log final checkpoints and their resolved config as one W&B artifact."""
    if not config.get("upload_final_checkpoint", False):
        return None
    if wandb.run is None:
        raise RuntimeError("upload_final_checkpoint requires an active W&B run")
    if not checkpoint_paths:
        raise RuntimeError("No final checkpoints were saved for artifact upload")

    artifact_name = f"overcooked-v3-{wandb.run.id}-final-checkpoint"
    artifact_metadata = {
        "run_id": wandb.run.id,
        "algorithm": str(config.get("ALGORITHM", "IPPO")),
        "architecture": _architecture(config),
        "layout": config["ENV_KWARGS"]["layout"],
        "seed": int(config["SEED"]),
        "num_seeds": int(config["NUM_SEEDS"]),
        "checkpoint_format": "safetensors",
    }
    candidate = resolve_hsp_config(config)
    if candidate is not None:
        artifact_metadata.update(
            {
                "hsp_profile": candidate.profile,
                "hsp_candidate_id": candidate.candidate_id,
                "hsp_resolved_utility": candidate.metadata(),
                "hsp_checkpoint_fractions": list(
                    config.get("CHECKPOINT_FRACTIONS") or ()
                ),
                "hsp_shared_policy_approximation": True,
            }
        )

    artifact = wandb.Artifact(
        artifact_name,
        type="checkpoint",
        description="Final Overcooked V3 IPPO checkpoint(s).",
        metadata=artifact_metadata,
    )
    for checkpoint_path in checkpoint_paths:
        checkpoint_path = Path(checkpoint_path)
        artifact.add_file(str(checkpoint_path), name=checkpoint_path.name)
    if config_path is not None:
        config_path = Path(config_path)
        artifact.add_file(str(config_path), name=config_path.name)
    for extra_file in extra_files:
        extra_file = Path(extra_file)
        artifact.add_file(str(extra_file), name=extra_file.name)

    logged_artifact = wandb.run.log_artifact(artifact, aliases=["final"])
    wandb.run.summary["checkpoint/artifact_name"] = artifact_name
    wandb.run.summary["checkpoint/uploaded"] = True
    print(
        f"[{_timestamp()}] Queued final checkpoint artifact: {artifact_name}:final",
        flush=True,
    )
    return logged_artifact


def _write_hsp_candidate_result(config, save_dir, checkpoint_paths):
    """Write one immutable, run-scoped population-candidate sidecar.

    Selection features deliberately remain unresolved here: the training scan
    does not run a frozen final response-policy scorer, and its shared-policy
    on-policy event stream is not the BR event statistic used by CooT's greedy
    selector.  A post-training two-seat rollout scorer must populate that
    field before diversity selection.
    """

    candidate = resolve_hsp_config(config)
    if candidate is None:
        return None

    mid_paths = []
    final_paths = []
    for checkpoint_path in checkpoint_paths:
        # Checkpoints are placed beside the sidecar and uploaded at the W&B
        # artifact root. Relative names therefore work both locally and after
        # downloading a distributed sweep artifact on another machine.
        portable_path = Path(checkpoint_path).name
        if "_update" in Path(checkpoint_path).stem:
            mid_paths.append(portable_path)
        else:
            final_paths.append(portable_path)

    run_id = getattr(wandb.run, "id", None) or datetime.now().strftime(
        "%Y%m%d-%H%M%S-%f"
    )
    sidecar_path = Path(save_dir) / (
        f"{_checkpoint_prefix(config)}_seed{config['SEED']}_"
        f"candidate_result_{run_id}.json"
    )
    policy_metadata = {
        "architecture": _architecture(config),
        "activation": str(config["ACTIVATION"]),
        "fc_dim_size": int(config["FC_DIM_SIZE"]),
        "gru_hidden_dim": int(config["GRU_HIDDEN_DIM"]),
        # Supplementary runner.rollout uses deterministic=False for the
        # population trajectories and event-diversity scoring rollouts.
        "stochastic": True,
    }
    mid_policy = {"checkpoint": mid_paths[0], **policy_metadata} if mid_paths else None
    final_policy = (
        {"checkpoint": final_paths[0], **policy_metadata} if final_paths else None
    )
    payload = {
        "schema_version": 1,
        "id": f"hsp_{candidate.candidate_id:04d}",
        "candidate_id": candidate.candidate_id,
        "population_type": "hsp",
        "immutable_run_id": run_id,
        "algorithm": str(config.get("ALGORITHM", "HSP")),
        "layout": config["ENV_KWARGS"]["layout"],
        "scenario": config.get("CONDITION") or config["ENV_KWARGS"]["layout"],
        "seed": int(config["SEED"]),
        "num_seeds": int(config["NUM_SEEDS"]),
        "resolved_utility": candidate.metadata(),
        "partner": {
            "mid": mid_policy,
            "final": final_policy,
        },
        "checkpoints": {
            "mid_fraction": float(config["HSP"].get("MID_CHECKPOINT_FRACTION", 0.5)),
            "mid": mid_paths,
            "final": final_paths,
            "shared_policy_approximation": True,
        },
        "selection_features": None,
        "selection_feature_status": (
            "requires_post_training_two_seat_response_rollout_scorer"
        ),
        "selection_feature_note": (
            "CooT selection uses final response/BR episode event counts; "
            "shared on-policy HSP training events are not that statistic."
        ),
        "porting_note": config["HSP"]["PORTING_NOTE"],
    }
    with sidecar_path.open("x", encoding="utf-8") as sidecar_file:
        json.dump(payload, sidecar_file, indent=2, sort_keys=True)
        sidecar_file.write("\n")
    return sidecar_path


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


def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


def make_train(config):
    hsp_candidate = resolve_hsp_config(config)
    entropy_schedule = _resolve_entropy_schedule(config)
    hsp_randomize_roles = bool(
        hsp_candidate is not None and config["HSP"].get("RANDOMIZE_ROLE", True)
    )
    env_kwargs = dict(config["ENV_KWARGS"])
    env = jaxmarl.make(config["ENV_NAME"], **env_kwargs)
    if hsp_candidate is not None and env.num_agents != 2:
        raise ValueError("HSP population training requires exactly two agents")
    architecture = _architecture(config)
    checkpoint_prefix = _checkpoint_prefix(config)

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

            update = int(update_step)
            vmap_index = int(seed_index)
            checkpoint_path = os.path.join(
                save_dir,
                f"{checkpoint_prefix}_{experiment_name}_seed{config['SEED']}_"
                f"vmap{vmap_index}_update{update:06d}.safetensors",
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

    rew_shaping_anneal = optax.linear_schedule(
        init_value=1.0, end_value=0.0, transition_steps=config["REW_SHAPING_HORIZON"]
    )

    def entropy_coefficient_at_step(env_step):
        if entropy_schedule is None:
            return jnp.asarray(config["ENT_COEF"], dtype=jnp.float32)

        coefs, horizons = entropy_schedule
        step = jnp.asarray(env_step, dtype=jnp.float32)
        coefficient = jnp.asarray(coefs[-1], dtype=jnp.float32)
        for interval_index in range(len(coefs) - 2, -1, -1):
            start_step = float(horizons[interval_index])
            end_step = float(horizons[interval_index + 1])
            fraction = jnp.clip((step - start_step) / (end_step - start_step), 0.0, 1.0)
            interpolated = (1.0 - fraction) * coefs[interval_index] + fraction * coefs[
                interval_index + 1
            ]
            coefficient = jnp.where(step < end_step, interpolated, coefficient)
        return coefficient

    def train(rng, seed_index):
        # INIT NETWORK
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
        if hsp_candidate is not None:
            rng, role_rng = jax.random.split(rng)
            if hsp_randomize_roles:
                biased_agent_indices = jax.random.randint(
                    role_rng,
                    (config["NUM_ENVS"],),
                    minval=0,
                    maxval=2,
                    dtype=jnp.int32,
                )
            else:
                biased_agent_indices = jnp.zeros((config["NUM_ENVS"],), dtype=jnp.int32)

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                if hsp_candidate is None:
                    (
                        train_state,
                        env_state,
                        last_obs,
                        last_done,
                        update_step,
                        hstate,
                        rng,
                    ) = runner_state
                    current_biased_agent_indices = None
                else:
                    (
                        train_state,
                        env_state,
                        last_obs,
                        last_done,
                        update_step,
                        current_biased_agent_indices,
                        hstate,
                        rng,
                    ) = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)

                # obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
                obs_batch = jnp.stack([last_obs[a] for a in env.agents]).reshape(
                    -1, *env.observation_space(env.agents[0]).shape
                )
                ac_in = (
                    obs_batch[np.newaxis, :],
                    last_done[np.newaxis, :],
                )

                hstate, pi, value = network.apply(train_state.params, hstate, ac_in)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
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

                # event_vector has a trailing event dimension and therefore
                # cannot enter the scalar-info reshape below.  Consume it for
                # HSP reward routing, then keep the existing IPPO info shapes
                # unchanged for both HSP and all default runs.
                event_vectors = info.get("event_vector")
                info = {
                    key: value for key, value in info.items() if key != "event_vector"
                }

                current_timestep = (
                    update_step * config["NUM_STEPS"] * config["NUM_ENVS"]
                )
                if hsp_candidate is None:
                    anneal_factor = rew_shaping_anneal(current_timestep)
                    reward = jax.tree.map(
                        lambda x, y: x + y * anneal_factor,
                        reward,
                        info["shaped_reward"],
                    )
                    hsp_event_reward = None
                else:
                    if event_vectors is None:
                        raise ValueError(
                            "HSP requires Overcooked V3 info['event_vector']"
                        )
                    reward, event_components = apply_candidate_rewards(
                        reward,
                        event_vectors,
                        hsp_candidate,
                        current_biased_agent_indices,
                    )
                    hsp_event_reward = jnp.array(
                        [event_components[a] for a in env.agents]
                    )
                    # Built-in V3 shaping is not part of either HSP role's
                    # objective. Table-5 fixed weights are already represented
                    # in the candidate event vector.
                    anneal_factor = jnp.array(0.0, dtype=jnp.float32)

                shaped_reward = jnp.array(
                    [info["shaped_reward"][a] for a in env.agents]
                )
                combined_reward = jnp.array([reward[a] for a in env.agents])

                info["shaped_reward"] = shaped_reward
                info["original_reward"] = original_reward
                info["anneal_factor"] = jnp.full_like(shaped_reward, anneal_factor)
                info["combined_reward"] = combined_reward
                if hsp_event_reward is not None:
                    info["hsp_event_reward"] = hsp_event_reward
                    info["hsp_training_reward"] = combined_reward

                info = jax.tree.map(lambda x: x.reshape((config["NUM_ACTORS"])), info)
                done_batch = batchify(done, env.agents, config["NUM_ACTORS"]).squeeze()
                transition = Transition(
                    jnp.tile(done["__all__"], env.num_agents),
                    last_done,
                    action.squeeze(),
                    value.squeeze(),
                    batchify(reward, env.agents, config["NUM_ACTORS"]).squeeze(),
                    log_prob.squeeze(),
                    obs_batch,
                    info,
                )
                if hsp_candidate is None:
                    runner_state = (
                        train_state,
                        env_state,
                        obsv,
                        done_batch,
                        update_step,
                        hstate,
                        rng,
                    )
                else:
                    if hsp_randomize_roles:
                        rng, role_rng = jax.random.split(rng)
                        sampled_biased_agent_indices = jax.random.randint(
                            role_rng,
                            (config["NUM_ENVS"],),
                            minval=0,
                            maxval=2,
                            dtype=jnp.int32,
                        )
                        next_biased_agent_indices = jnp.where(
                            done["__all__"],
                            sampled_biased_agent_indices,
                            current_biased_agent_indices,
                        )
                    else:
                        next_biased_agent_indices = current_biased_agent_indices
                    runner_state = (
                        train_state,
                        env_state,
                        obsv,
                        done_batch,
                        update_step,
                        next_biased_agent_indices,
                        hstate,
                        rng,
                    )
                return runner_state, transition

            initial_hstate = runner_state[-2]
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            if hsp_candidate is None:
                (
                    train_state,
                    env_state,
                    last_obs,
                    last_done,
                    update_step,
                    hstate,
                    rng,
                ) = runner_state
            else:
                (
                    train_state,
                    env_state,
                    last_obs,
                    last_done,
                    update_step,
                    biased_agent_indices,
                    hstate,
                    rng,
                ) = runner_state
            # ZSC-Eval updates entropy after collecting a rollout, using the
            # completed environment-step count.
            entropy_coefficient = entropy_coefficient_at_step(
                (update_step + 1) * config["NUM_STEPS"] * config["NUM_ENVS"]
            )
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

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    init_hstate, traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, init_hstate, traj_batch, gae, targets):
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
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
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
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - entropy_coefficient * entropy
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
                "entropy_coef": entropy_coefficient,
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

            if hsp_candidate is None:
                runner_state = (
                    train_state,
                    env_state,
                    last_obs,
                    last_done,
                    update_step,
                    hstate,
                    rng,
                )
            else:
                runner_state = (
                    train_state,
                    env_state,
                    last_obs,
                    last_done,
                    update_step,
                    biased_agent_indices,
                    hstate,
                    rng,
                )
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        if hsp_candidate is None:
            runner_state = (
                train_state,
                env_state,
                obsv,
                jnp.zeros((config["NUM_ACTORS"]), dtype=bool),
                0,
                init_hstate,
                _rng,
            )
        else:
            runner_state = (
                train_state,
                env_state,
                obsv,
                jnp.zeros((config["NUM_ACTORS"]), dtype=bool),
                0,
                biased_agent_indices,
                init_hstate,
                _rng,
            )
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metric}

    return train


def run(config):
    config = OmegaConf.to_container(config, resolve=True)
    hsp_candidate = resolve_hsp_config(config)
    if hsp_candidate is not None:
        # PORTING NOTE: a production sweep runs every utility candidate with the
        # same layout/seed. Keep each candidate in its own directory so retries
        # and concurrent W&B agents cannot overwrite another candidate's
        # checkpoint/config while preserving the established filename schema.
        _isolate_hsp_output(config, hsp_candidate)
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
    if hsp_candidate is not None:
        require_sweep_target(wandb_run, config)
    wandb.define_metric("train/env_step")
    wandb.define_metric("train/*", step_metric="train/env_step")
    wandb.define_metric("debug/*", step_metric="train/env_step")
    wandb.define_metric("eval/*")
    if hsp_candidate is not None and wandb.run is not None:
        wandb.run.summary["hsp/profile"] = hsp_candidate.profile
        wandb.run.summary["hsp/candidate_id"] = hsp_candidate.candidate_id
        wandb.run.summary["hsp/candidate_count"] = config["HSP"][
            "RESOLVED_CANDIDATE_COUNT"
        ]
        wandb.run.summary["hsp/sparse_reward_weight"] = (
            hsp_candidate.sparse_reward_weight
        )

    with jax.disable_jit(False):
        rng = jax.random.PRNGKey(config["SEED"])
        rngs = jax.random.split(rng, num_seeds)
        seed_indices = jnp.arange(num_seeds)
        train_jit = jax.jit(make_train(config))
        out = jax.block_until_ready(jax.vmap(train_jit)(rngs, seed_indices))

    model_state = out["runner_state"][0]
    checkpoint_paths = []
    if save_dir is not None:
        from jaxmarl.wrappers.baselines import save_params

        if hsp_candidate is not None:
            for update in _checkpoint_update_steps(config):
                for i in range(num_seeds):
                    intermediate_path = Path(save_dir) / (
                        f"{checkpoint_prefix}_{experiment_name}_"
                        f"seed{config['SEED']}_vmap{i}_"
                        f"update{update:06d}.safetensors"
                    )
                    if not intermediate_path.is_file():
                        raise RuntimeError(
                            "Expected HSP intermediate checkpoint was not saved: "
                            f"{intermediate_path}"
                        )
                    checkpoint_paths.append(intermediate_path)

        for i in range(num_seeds):
            params = jax.tree.map(lambda x: x[i], model_state.params)
            checkpoint_path = os.path.join(
                save_dir,
                f"{checkpoint_prefix}_{experiment_name}_seed{config['SEED']}_"
                f"vmap{i}.safetensors",
            )
            save_params(params, checkpoint_path)
            checkpoint_paths.append(Path(checkpoint_path))
            print(f"[{_timestamp()}] Saved checkpoint: {checkpoint_path}")

    candidate_result_path = None
    if hsp_candidate is not None and save_dir is not None:
        candidate_result_path = _write_hsp_candidate_result(
            config, save_dir, checkpoint_paths
        )
        if wandb.run is not None:
            wandb.run.summary["hsp/candidate_result"] = str(candidate_result_path)

    if upload_final_checkpoint:
        extra_files = (
            (candidate_result_path,) if candidate_result_path is not None else ()
        )
        _log_final_checkpoint_artifact(
            config,
            checkpoint_paths,
            config_path,
            extra_files=extra_files,
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

    wandb.finish()


@hydra.main(
    version_base=None, config_path="../../conf", config_name="ippo_overcooked_v3"
)
def main(config):
    run(config)


def entrypoint():
    if load_project_env():
        print(f"[{_timestamp()}] Loaded project .env")
    main()


if __name__ == "__main__":
    entrypoint()
