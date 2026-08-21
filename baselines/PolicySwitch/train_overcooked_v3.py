"""Train phase-gated IPPO policies on full dynamic Overcooked V3 episodes."""

import sys
from contextlib import contextmanager
from pathlib import Path

import distrax
import flax.linen as nn
import hydra
import jax
import jax.numpy as jnp
import wandb
from omegaconf import OmegaConf

from jaxmarl._env import load_project_env
from jaxmarl._experiment import experiment_folder
from jaxmarl.environments.spaces import Box
from jaxmarl.environments.overcooked_v3 import (
    dynamic_layouts,
    phase_policy_layout_name,
    phase_policy_sequence,
)
from jaxmarl.wrappers.baselines import JaxMARLWrapper

try:
    from baselines.IPPO import ippo_overcooked_v3 as ippo
except ModuleNotFoundError as error:
    if error.name != "baselines":
        raise
    ippo_dir = Path(__file__).resolve().parents[1] / "IPPO"
    sys.path.insert(0, str(ippo_dir))
    import ippo_overcooked_v3 as ippo

try:
    from .policy_switch import (
        ALGORITHM_NAME,
        save_combined_policy_params,
        validate_policy_switch_layout,
    )
except ImportError:  # Direct execution: python baselines/PolicySwitch/<script>.py
    from policy_switch import (
        ALGORITHM_NAME,
        save_combined_policy_params,
        validate_policy_switch_layout,
    )


_BASE_ACTOR_CRITIC_CNN = ippo.ActorCriticCNN


def _checkpoint_prefix(config):
    return f"policy_switch_ippo_{ippo._architecture(config)}"


def _checkpoint_metadata(config):
    layout = config["ENV_KWARGS"]["layout"]
    experiment_name = f"overcooked_v3_{layout}"
    save_dir = Path(config["SAVES_DIR"]) / experiment_folder(config)
    return experiment_name, save_dir


class PhasePolicyObservationWrapper(JaxMARLWrapper):
    """Append an internal one-hot marker for the active policy head."""

    def __init__(self, env, policy_sequence):
        super().__init__(env)
        self.policy_sequence = tuple(int(index) for index in policy_sequence)
        self.policy_count = max(self.policy_sequence) + 1
        self._phase_to_policy = jnp.asarray(self.policy_sequence, dtype=jnp.int32)

    def _append_policy_marker(self, obs, state):
        policy_index = self._phase_to_policy[state.layout_index]

        def append(agent_obs):
            marker = jax.nn.one_hot(
                policy_index,
                self.policy_count,
                dtype=agent_obs.dtype,
            )
            marker = jnp.broadcast_to(
                marker,
                (*agent_obs.shape[:-1], self.policy_count),
            )
            return jnp.concatenate((agent_obs, marker), axis=-1)

        return {agent: append(agent_obs) for agent, agent_obs in obs.items()}

    def reset(self, key):
        obs, state = self._env.reset(key)
        return self._append_policy_marker(obs, state), state

    def step(self, key, state, actions):
        obs, state, reward, done, info = self._env.step(key, state, actions)
        return self._append_policy_marker(obs, state), state, reward, done, info

    def observation_space(self, agent_id=""):
        base = self._env.observation_space(agent_id)
        return Box(
            base.low,
            base.high,
            (*base.shape[:-1], base.shape[-1] + self.policy_count),
            dtype=base.dtype,
        )


class PhaseGatedActorCriticCNN(nn.Module):
    """Route each transition through one independent ordinary IPPO CNN."""

    action_dim: int
    config: dict

    @nn.compact
    def __call__(self, hidden, x):
        augmented_obs, dones = x
        policy_count = int(self.config["POLICY_COUNT"])
        obs = augmented_obs[..., :-policy_count]
        marker = augmented_obs[..., -policy_count:]
        marker = marker.reshape((*marker.shape[:2], -1, policy_count))[..., 0, :]

        logits = []
        values = []
        for policy_index in range(policy_count):
            _, distribution, value = _BASE_ACTOR_CRITIC_CNN(
                self.action_dim,
                config=self.config,
                name=f"policy_{policy_index}",
            )(hidden, (obs, dones))
            logits.append(distribution.logits)
            values.append(value)

        selector = jax.nn.one_hot(
            jnp.argmax(marker, axis=-1),
            policy_count,
            dtype=obs.dtype,
        )
        selected_logits = jnp.sum(
            jnp.stack(logits, axis=2) * selector[..., None],
            axis=2,
        )
        selected_value = jnp.sum(
            jnp.stack(values, axis=2) * selector,
            axis=2,
        )
        return hidden, distrax.Categorical(logits=selected_logits), selected_value


@contextmanager
def _phase_gated_ippo(config):
    """Inject PolicySwitch-only env observations and network routing."""
    original_make = ippo.jaxmarl.make
    original_cnn = ippo.ActorCriticCNN
    policy_sequence = tuple(config["POLICY_SEQUENCE"])

    def make_phase_gated_env(name, **kwargs):
        return PhasePolicyObservationWrapper(
            original_make(name, **kwargs),
            policy_sequence,
        )

    ippo.jaxmarl.make = make_phase_gated_env
    ippo.ActorCriticCNN = PhaseGatedActorCriticCNN
    try:
        yield
    finally:
        ippo.ActorCriticCNN = original_cnn
        ippo.jaxmarl.make = original_make


def _train_phase_gated_policies(config):
    if ippo._architecture(config) != "cnn":
        raise ValueError("Phase-gated PolicySwitch training currently requires CNN")
    num_seeds = int(config["NUM_SEEDS"])
    rng = jax.random.PRNGKey(int(config["SEED"]))
    rngs = jax.random.split(rng, num_seeds)
    seed_indices = jnp.arange(num_seeds)
    print(
        f"[{ippo._timestamp()}] Training {config['POLICY_COUNT']} phase-gated "
        f"policies on dynamic layout {config['ENV_KWARGS']['layout']}",
        flush=True,
    )
    with _phase_gated_ippo(config), jax.disable_jit(False):
        train_jit = jax.jit(ippo.make_train(config))
        output = jax.block_until_ready(jax.vmap(train_jit)(rngs, seed_indices))
    return output["runner_state"][0].params


def _split_phase_gated_params(params, policy_count):
    """Convert gated Flax params back to the existing combined format."""
    return tuple(
        {"params": params["params"][f"policy_{policy_index}"]}
        for policy_index in range(policy_count)
    )


def _wandb_metadata(config):
    architecture = ippo._architecture(config)
    layout = config["ENV_KWARGS"]["layout"]
    tags = list(config.get("WANDB_TAGS") or [])
    tags.extend([ALGORITHM_NAME, architecture.upper(), "OvercookedV3", "PolicySwitch"])
    tags = list(dict.fromkeys(tags))
    name = str(
        config.get("RUN_NAME")
        or f"policy_switch_ippo_{architecture}_{layout}_seed{config['SEED']}"
    )
    group = str(config.get("POLICY_SWITCH_WANDB_GROUP") or "policy-switch")
    return name, group, tags


def _log_final_checkpoint_artifact(config, checkpoint_paths, config_path):
    if not config.get("upload_final_checkpoint", False):
        return None
    if wandb.run is None:
        raise RuntimeError("upload_final_checkpoint requires an active W&B run")

    base_layout = config["ENV_KWARGS"]["layout"]
    phases = dynamic_layouts[base_layout].phases
    artifact_name = f"overcooked-v3-policy-switch-{wandb.run.id}-final-checkpoint"
    artifact = wandb.Artifact(
        artifact_name,
        type="checkpoint",
        description=(
            "Phase-gated IPPO policies jointly trained on full dynamic "
            "Overcooked V3 episodes, packed into each safetensors checkpoint."
        ),
        metadata={
            "run_id": wandb.run.id,
            "algorithm": ALGORITHM_NAME,
            "architecture": ippo._architecture(config),
            "layout": base_layout,
            "seed": int(config["SEED"]),
            "num_seeds": int(config["NUM_SEEDS"]),
            "checkpoint_format": "combined_safetensors",
            "training_mode": config["POLICY_SWITCH_TRAINING"],
            "phase_count": len(phases),
            "policy_count": len(config["POLICY_LAYOUTS"]),
            "policy_layouts": list(config["POLICY_LAYOUTS"]),
            "policy_sequence": list(config["POLICY_SEQUENCE"]),
            "phase_recipes": [
                None if phase.recipe is None else list(phase.recipe) for phase in phases
            ],
        },
    )
    for checkpoint_path in checkpoint_paths:
        artifact.add_file(str(checkpoint_path), name=Path(checkpoint_path).name)
    artifact.add_file(str(config_path), name=Path(config_path).name)
    logged_artifact = wandb.run.log_artifact(artifact, aliases=["final"])
    wandb.run.summary["checkpoint/artifact_name"] = artifact_name
    wandb.run.summary["checkpoint/uploaded"] = True
    wandb.run.summary["checkpoint/combined_policy_count"] = len(
        config["POLICY_LAYOUTS"]
    )
    print(
        f"[{ippo._timestamp()}] Queued combined checkpoint artifact: "
        f"{artifact_name}:final",
        flush=True,
    )
    return logged_artifact


def run(config):
    config = OmegaConf.to_container(config, resolve=True)
    base_layout = validate_policy_switch_layout(config["ENV_KWARGS"]["layout"])
    config["ALGORITHM"] = ALGORITHM_NAME
    config["POLICY_SEQUENCE"] = list(phase_policy_sequence(base_layout))
    policy_count = max(config["POLICY_SEQUENCE"]) + 1
    config["POLICY_COUNT"] = policy_count
    config["POLICY_LAYOUTS"] = [
        phase_policy_layout_name(base_layout, policy_index)
        for policy_index in range(policy_count)
    ]
    config["POLICY_SWITCH_TRAINING"] = "phase_gated_dynamic"

    requested_wandb_mode = str(config.get("wandb_mode", "online")).lower()
    config["wandb_mode"] = ippo._resolve_wandb_mode(config)
    if requested_wandb_mode == "online" and config["wandb_mode"] == "offline":
        print(
            f"[{ippo._timestamp()}] WANDB_API_KEY is not set; using offline W&B mode",
            flush=True,
        )

    if not config.get("SAVES_DIR"):
        raise ValueError("SAVES_DIR is required for the combined safetensors output")
    experiment_name, save_dir = _checkpoint_metadata(config)
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_prefix = _checkpoint_prefix(config)
    config_path = save_dir / (
        f"{checkpoint_prefix}_{experiment_name}_seed{config['SEED']}_config.yaml"
    )
    OmegaConf.save(OmegaConf.create(config), config_path)

    wandb_name, wandb_group, wandb_tags = _wandb_metadata(config)
    wandb.init(
        **ippo._wandb_target(config),
        tags=wandb_tags,
        config=config,
        mode=config["wandb_mode"],
        name=wandb_name,
        group=wandb_group,
        job_type="train",
        notes=config.get("NOTES"),
    )
    wandb.define_metric("train/env_step")
    wandb.define_metric("train/*", step_metric="train/env_step")
    wandb.define_metric("debug/*")

    try:
        phase_gated_params = _train_phase_gated_policies(config)

        checkpoint_paths = []
        for vmap_index in range(int(config["NUM_SEEDS"])):
            seed_params = jax.tree.map(
                lambda value: value[vmap_index], phase_gated_params
            )
            policies = _split_phase_gated_params(seed_params, policy_count)
            checkpoint_path = save_dir / (
                f"{checkpoint_prefix}_{experiment_name}_seed{config['SEED']}_"
                f"vmap{vmap_index}.safetensors"
            )
            save_combined_policy_params(policies, checkpoint_path)
            checkpoint_paths.append(checkpoint_path)
            print(
                f"[{ippo._timestamp()}] Saved {policy_count}-policy checkpoint: "
                f"{checkpoint_path}",
                flush=True,
            )

        wandb_enabled = str(config.get("wandb_mode", "disabled")).lower() != "disabled"
        if config.get("upload_final_checkpoint", False) and wandb_enabled:
            _log_final_checkpoint_artifact(config, checkpoint_paths, config_path)
    finally:
        wandb.finish()

    return checkpoint_paths


@hydra.main(
    version_base=None,
    config_path="../../conf",
    config_name="policy_switch_overcooked_v3",
)
def main(config):
    run(config)


def entrypoint():
    if load_project_env():
        print(f"[{ippo._timestamp()}] Loaded project .env")
    main()


if __name__ == "__main__":
    entrypoint()
