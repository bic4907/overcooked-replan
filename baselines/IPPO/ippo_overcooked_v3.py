import functools
import os
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
    "learning_rate": "learning_rate",
    "update_step": "update",
    "env_step": "env_step",
}

DEBUG_METRIC_NAMES = {
    "layout_index": "layout_index",
    "layout_changed": "layout_changed_fraction",
    "layout_change_events": "layout_change_events",
    "steps_until_layout_change": "steps_until_layout_change",
    "transition_countdown": "transition_countdown",
    "layout_change_tile_count": "layout_change_tile_count",
    "wall_tile_count": "wall_tile_count",
    "ingredient_pile_count": "ingredient_pile_count",
    "signal_tile_count": "signal_tile_count",
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


def _architecture(config):
    architecture = config.get("ARCHITECTURE", "rnn").lower()
    if architecture not in {"cnn", "rnn"}:
        raise ValueError("ARCHITECTURE must be either 'cnn' or 'rnn'")
    return architecture


def _checkpoint_prefix(config):
    return f"ippo_{_architecture(config)}"


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
    architecture = _architecture(config)
    layout_name = config["ENV_KWARGS"]["layout"]
    condition = layout_name
    experiment = config.get("EXPERIMENT", "overcooked_v3")
    signal_tag = "Sig" if config.get("SIGNAL_ENABLED", False) else "NoSig"

    tags = list(config.get("WANDB_TAGS") or [])
    tags.extend(["IPPO", architecture.upper(), "OvercookedV3", experiment, signal_tag])
    tags = list(dict.fromkeys(tags))

    group = config.get("WANDB_GROUP") or experiment
    name = config.get("RUN_NAME") or (
        f"{_checkpoint_prefix(config)}_{condition}_seed{config['SEED']}"
    )
    return name, group, tags


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
    env_kwargs = dict(config["ENV_KWARGS"])
    env = jaxmarl.make(config["ENV_NAME"], **env_kwargs)
    architecture = _architecture(config)
    checkpoint_prefix = _checkpoint_prefix(config)

    checkpoint_interval = int(config.get("CHECKPOINT_INTERVAL", 0))
    if checkpoint_interval < 0:
        raise ValueError("CHECKPOINT_INTERVAL must be greater than or equal to 0")
    checkpoint_enabled = checkpoint_interval > 0 and config.get("SAVES_DIR") is not None
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

    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
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
                runner_state = (
                    train_state,
                    env_state,
                    obsv,
                    done_batch,
                    update_step,
                    hstate,
                    rng,
                )
                return runner_state, transition

            initial_hstate = runner_state[-2]
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, last_done, update_step, hstate, rng = (
                runner_state
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
                            - config["ENT_COEF"] * entropy
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
                    advantages.squeeze(),
                    targets.squeeze(),
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
                "steps_until_layout_change": traj_batch.info[
                    "steps_until_layout_change"
                ][-1],
                "transition_countdown": traj_batch.info["transition_countdown"][-1],
                "layout_change_tile_count": traj_batch.info["layout_change_tile_count"][
                    -1
                ],
                "wall_tile_count": traj_batch.info["wall_tile_count"][-1],
                "ingredient_pile_count": traj_batch.info["ingredient_pile_count"][-1],
                "signal_tile_count": traj_batch.info["signal_tile_count"][-1],
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
                "learning_rate": learning_rate_fn(
                    update_step * config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]
                ),
                "layout_change_events": (
                    traj_batch.info["layout_changed"].sum() / env.num_agents
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
            metric = jax.tree.map(lambda x: x.mean(), metric)
            metric["update_step"] = update_step
            metric["env_step"] = update_step * config["NUM_STEPS"] * config["NUM_ENVS"]
            jax.debug.callback(callback, metric)

            if checkpoint_enabled:
                should_save = jnp.logical_and(
                    update_step % checkpoint_interval == 0,
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
            _rng,
        )
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metric}

    return train


def run(config):
    config = OmegaConf.to_container(config, resolve=True)

    layout_name = config["ENV_KWARGS"]["layout"]
    num_seeds = config["NUM_SEEDS"]
    architecture = _architecture(config)
    checkpoint_prefix = _checkpoint_prefix(config)
    save_dir = None

    if config.get("SAVES_DIR") is not None:
        experiment_name, save_dir = _checkpoint_metadata(config)
        os.makedirs(save_dir, exist_ok=True)
        config_path = os.path.join(
            save_dir,
            f"{checkpoint_prefix}_{experiment_name}_seed{config['SEED']}_config.yaml",
        )
        OmegaConf.save(OmegaConf.create(config), config_path)

    wandb_name, wandb_group, wandb_tags = _wandb_metadata(config)
    wandb.init(
        entity=config.get("ENTITY") or None,
        project=config["PROJECT"],
        tags=wandb_tags,
        config=config,
        mode=config["WANDB_MODE"],
        name=wandb_name,
        group=wandb_group,
        job_type="train",
        notes=config.get("NOTES"),
    )
    wandb.define_metric("train/env_step")
    wandb.define_metric("train/*", step_metric="train/env_step")
    wandb.define_metric("debug/*", step_metric="train/env_step")
    wandb.define_metric("eval/*")

    with jax.disable_jit(False):
        rng = jax.random.PRNGKey(config["SEED"])
        rngs = jax.random.split(rng, num_seeds)
        seed_indices = jnp.arange(num_seeds)
        train_jit = jax.jit(make_train(config))
        out = jax.block_until_ready(jax.vmap(train_jit)(rngs, seed_indices))

    model_state = out["runner_state"][0]
    if save_dir is not None:
        from jaxmarl.wrappers.baselines import save_params

        for i in range(num_seeds):
            params = jax.tree.map(lambda x: x[i], model_state.params)
            checkpoint_path = os.path.join(
                save_dir,
                f"{checkpoint_prefix}_{experiment_name}_seed{config['SEED']}_"
                f"vmap{i}.safetensors",
            )
            save_params(params, checkpoint_path)
            print(f"[{_timestamp()}] Saved checkpoint: {checkpoint_path}")

    recording_enabled = bool(config.get("RECORD_FINAL_EPISODE", True))
    wandb_enabled = str(config.get("WANDB_MODE", "disabled")).lower() != "disabled"
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
