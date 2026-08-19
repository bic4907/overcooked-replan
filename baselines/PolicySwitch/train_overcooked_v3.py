"""Train one static IPPO policy per unique Overcooked V3 layout phase."""

import copy
import sys
from contextlib import contextmanager
from pathlib import Path

import hydra
import jax
import wandb
from omegaconf import OmegaConf

from jaxmarl._env import load_project_env
from jaxmarl._experiment import experiment_folder
from jaxmarl.environments.overcooked_v3 import (
    dynamic_layouts,
    phase_policy_layout_name,
    phase_policy_sequence,
)

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


def _checkpoint_prefix(config):
    return f"policy_switch_ippo_{ippo._architecture(config)}"


def _checkpoint_metadata(config):
    layout = config["ENV_KWARGS"]["layout"]
    experiment_name = f"overcooked_v3_{layout}"
    save_dir = Path(config["SAVES_DIR"]) / experiment_folder(config)
    return experiment_name, save_dir


def _policy_training_config(config, policy_index):
    phase_config = copy.deepcopy(config)
    base_layout = config["ENV_KWARGS"]["layout"]
    phase_config["ENV_KWARGS"] = dict(config["ENV_KWARGS"])
    phase_config["ENV_KWARGS"]["layout"] = phase_policy_layout_name(
        base_layout, policy_index
    )
    phase_config["CHECKPOINT_INTERVAL"] = 0
    return phase_config


@contextmanager
def _ippo_metric_namespace(policy_index):
    """Namespace the copied IPPO metrics without modifying the IPPO baseline."""
    original_prefixer = ippo._prefixed_wandb_metrics

    def namespaced_prefixer(metric):
        metrics = original_prefixer(metric)
        result = {}
        for key, value in metrics.items():
            namespace, name = key.split("/", 1)
            result[f"{namespace}/policy_{policy_index}/{name}"] = value
        return result

    ippo._prefixed_wandb_metrics = namespaced_prefixer
    try:
        yield
    finally:
        ippo._prefixed_wandb_metrics = original_prefixer


def _train_policy(config, policy_index):
    phase_config = _policy_training_config(config, policy_index)
    num_seeds = int(config["NUM_SEEDS"])
    rng = jax.random.fold_in(jax.random.PRNGKey(int(config["SEED"])), policy_index)
    rngs = jax.random.split(rng, num_seeds)
    seed_indices = jax.numpy.arange(num_seeds)
    print(
        f"[{ippo._timestamp()}] Training policy {policy_index} on "
        f"{phase_config['ENV_KWARGS']['layout']}",
        flush=True,
    )
    train_jit = jax.jit(ippo.make_train(phase_config))
    with _ippo_metric_namespace(policy_index), jax.disable_jit(False):
        output = jax.block_until_ready(jax.vmap(train_jit)(rngs, seed_indices))
    return output["runner_state"][0].params


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
            "One independently trained IPPO policy per unique Overcooked V3 "
            "phase, packed into each safetensors checkpoint."
        ),
        metadata={
            "run_id": wandb.run.id,
            "algorithm": ALGORITHM_NAME,
            "architecture": ippo._architecture(config),
            "layout": base_layout,
            "seed": int(config["SEED"]),
            "num_seeds": int(config["NUM_SEEDS"]),
            "checkpoint_format": "combined_safetensors",
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
    config["POLICY_LAYOUTS"] = [
        phase_policy_layout_name(base_layout, index) for index in range(policy_count)
    ]

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
    for policy_index in range(policy_count):
        train_namespace = f"train/policy_{policy_index}"
        debug_namespace = f"debug/policy_{policy_index}"
        wandb.define_metric(f"{train_namespace}/env_step")
        wandb.define_metric(
            f"{train_namespace}/*", step_metric=f"{train_namespace}/env_step"
        )
        wandb.define_metric(f"{debug_namespace}/*")

    try:
        policy_params = [
            _train_policy(config, policy_index) for policy_index in range(policy_count)
        ]

        checkpoint_paths = []
        for vmap_index in range(int(config["NUM_SEEDS"])):
            policies = [
                jax.tree.map(lambda value: value[vmap_index], params)
                for params in policy_params
            ]
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
