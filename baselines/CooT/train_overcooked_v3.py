"""Train the Overcooked V3 Coordination Transformer from offline rollouts."""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, cast

import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from flax.training.train_state import TrainState
from omegaconf import DictConfig, OmegaConf

from jaxmarl._experiment import experiment_folder
from jaxmarl._wandb import require_sweep_target
from jaxmarl.wrappers.baselines import save_params

try:
    from .data import CooTShardDataset
    from .model import CooTConfig, CooTTransformer
except ImportError:  # Direct execution: python baselines/CooT/<script>.py
    from data import CooTShardDataset
    from model import CooTConfig, CooTTransformer


def _timestamp() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _resolve_dataset_path(config: dict) -> Path:
    explicit = config.get("DATASET_PATH")
    if explicit:
        return Path(str(explicit)).expanduser().resolve()
    return (
        (Path(str(config["DATASET_ROOT"])) / str(config["ENV_KWARGS"]["layout"]))
        .expanduser()
        .resolve()
    )


def _wandb_mode(config: dict) -> Literal["online", "offline", "disabled"]:
    mode = str(config.get("wandb_mode", "online")).lower()
    if mode not in {"online", "offline", "disabled"}:
        raise ValueError("wandb_mode must be online, offline, or disabled")
    if mode == "online" and not os.environ.get("WANDB_API_KEY", "").strip():
        return "offline"
    return cast(Literal["online", "offline", "disabled"], mode)


def _step_mask_count(config: dict, epoch: int) -> int:
    if not config.get("STEP_MASKING", True):
        return 0
    context_episodes = int(config["CONTEXT_EPISODES"])
    minimum = int(config["MIN_MASK_STEPS_PER_EPISODE"]) * context_episodes
    maximum = int(config["MAX_MASK_STEPS_PER_EPISODE"]) * context_episodes
    if not config.get("CURRICULUM_MASKING", True):
        return maximum
    # The release divides the zero-based epoch by num_epochs, so the final
    # scheduled epoch intentionally stops just short of the configured maximum.
    progress = epoch / max(1, int(config["MAX_EPOCHS"]))
    schedule = str(config.get("MASK_SCHEDULE", "logarithmic"))
    if schedule == "linear":
        rate = progress
    elif schedule == "exponential":
        rate = progress**2
    elif schedule == "logarithmic":
        rate = 1.0 - (1.0 - progress) ** 2
    else:
        raise ValueError(f"Unsupported MASK_SCHEDULE: {schedule}")
    return int(round(minimum + (maximum - minimum) * rate))


def _paper_epoch_lr_schedule(
    learning_rate: float,
    steps_per_epoch: int,
    max_epochs: int,
    warmup_fraction: float,
):
    """Reproduce the release's epoch-stepped warmup and linear decay."""

    warmup_epochs = int(warmup_fraction * max_epochs)

    def schedule(step):
        epoch = step // steps_per_epoch
        if warmup_epochs:
            warmup_rate = epoch / warmup_epochs
        else:
            warmup_rate = jnp.asarray(1.0)
        decay_rate = jnp.maximum(
            0.0,
            (max_epochs - epoch) / max(1, max_epochs - warmup_epochs),
        )
        rate = jnp.where(epoch < warmup_epochs, warmup_rate, decay_rate)
        return learning_rate * rate

    return schedule


def _loss_and_metrics(
    logits, labels, mask, label_smoothing, *, optimization_reduction="mean"
):
    one_hot = jax.nn.one_hot(labels, logits.shape[-1])
    if label_smoothing > 0:
        one_hot = one_hot * (1.0 - label_smoothing) + label_smoothing / logits.shape[-1]
    losses = optax.softmax_cross_entropy(logits, one_hot)
    denominator = jnp.maximum(mask.sum(), 1.0)
    loss_sum = (losses * mask).sum()
    mean_loss = loss_sum / denominator
    if optimization_reduction == "sum":
        optimization_loss = loss_sum
    elif optimization_reduction == "mean":
        optimization_loss = mean_loss
    else:
        raise ValueError(f"Unknown optimization reduction: {optimization_reduction}")
    accuracy = ((jnp.argmax(logits, axis=-1) == labels) * mask).sum() / denominator
    entropy = (
        -(jax.nn.softmax(logits) * jax.nn.log_softmax(logits)).sum(axis=-1) * mask
    ).sum() / denominator
    return optimization_loss, {
        "loss": mean_loss,
        "accuracy": accuracy,
        "entropy": entropy,
    }


def _save_checkpoint(
    path: Path,
    params,
    model_config: CooTConfig,
    train_config: dict,
    dataset_metadata: dict,
    *,
    epoch: int,
    validation_loss: float,
) -> tuple[Path, Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_params(jax.device_get(params), path)
    sidecar = path.with_suffix(".json")
    sidecar.write_text(
        json.dumps(
            {
                "format_version": 1,
                "method": "CooT",
                "source_paper": "arXiv:2506.23549",
                "model_config": asdict(model_config),
                "train_config": train_config,
                "dataset_metadata": dataset_metadata,
                "epoch": epoch,
                "validation_loss": validation_loss,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path, sidecar


@hydra.main(
    version_base=None,
    config_path="../../conf",
    config_name="coot_overcooked_v3",
)
def main(hydra_config: DictConfig) -> None:
    raw_config = OmegaConf.to_container(hydra_config, resolve=True)
    if not isinstance(raw_config, dict):
        raise TypeError("Resolved CooT config must be a mapping")
    config = cast(dict[str, Any], raw_config)
    dataset_path = _resolve_dataset_path(config)
    dataset = CooTShardDataset(dataset_path, cache_size=int(config["SHARD_CACHE_SIZE"]))
    layout = str(config["ENV_KWARGS"]["layout"])
    dataset_layout = str(dataset.metadata.get("layout"))
    if dataset_layout != layout:
        raise ValueError(
            f"Dataset layout {dataset_layout!r} does not match config layout {layout!r}"
        )

    model_config = CooTConfig(
        observation_dim=dataset.observation_dim,
        action_dim=dataset.action_dim,
        episode_horizon=dataset.horizon,
        context_episodes=int(config["CONTEXT_EPISODES"]),
        num_query_states=int(config["NUM_QUERY_STATES"]),
        embedding_dim=int(config["EMBEDDING_DIM"]),
        num_layers=int(config["NUM_LAYERS"]),
        num_heads=int(config["NUM_HEADS"]),
        dropout_rate=float(config["DROPOUT"]),
        attention_dropout_rate=float(config["ATTENTION_DROPOUT"]),
        layer_norm_epsilon=float(config["LAYER_NORM_EPSILON"]),
    )
    if model_config.embedding_dim % model_config.num_heads:
        raise ValueError("EMBEDDING_DIM must be divisible by NUM_HEADS")
    runtime_context_update_steps = int(config["RUNTIME_CONTEXT_UPDATE_STEPS"])
    if not 1 <= runtime_context_update_steps <= model_config.episode_horizon:
        raise ValueError(
            "RUNTIME_CONTEXT_UPDATE_STEPS must fit within one episode horizon"
        )

    batch_size = int(config["BATCH_SIZE"])
    max_epochs = int(config["MAX_EPOCHS"])
    validation_batches = int(config["VALIDATION_BATCHES"])
    if batch_size < 1:
        raise ValueError("BATCH_SIZE must be positive")
    if max_epochs < 1:
        raise ValueError("MAX_EPOCHS must be positive")
    if validation_batches < 1:
        raise ValueError("VALIDATION_BATCHES must be positive")
    if int(config["PATIENCE"]) < 1:
        raise ValueError("PATIENCE must be positive")
    examples_per_epoch = int(
        config.get("EXAMPLES_PER_EPOCH") or dataset.paper_examples_per_epoch
    )
    if examples_per_epoch < 1:
        raise ValueError("EXAMPLES_PER_EPOCH must be positive")
    steps_per_epoch = math.ceil(examples_per_epoch / batch_size)
    schedule = _paper_epoch_lr_schedule(
        float(config["LR"]),
        steps_per_epoch,
        max_epochs,
        float(config["LR_WARMUP_FRACTION"]),
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(float(config["GRAD_CLIP"])),
        optax.adamw(
            learning_rate=schedule,
            weight_decay=float(config["WEIGHT_DECAY"]),
            b1=0.9,
            b2=0.95,
        ),
    )

    model = CooTTransformer(model_config)
    key = jax.random.PRNGKey(int(config["SEED"]))
    key, init_key, dropout_key = jax.random.split(key, 3)
    context_steps = model_config.context_episodes * model_config.episode_horizon
    variables = model.init(
        {"params": init_key, "dropout": dropout_key},
        jnp.zeros((1, context_steps, model_config.observation_dim), dtype=jnp.float32),
        jnp.zeros((1, context_steps, model_config.action_dim), dtype=jnp.float32),
        jnp.zeros((1, context_steps, 1), dtype=jnp.float32),
        jnp.zeros(
            (1, model_config.num_query_states, model_config.observation_dim),
            dtype=jnp.float32,
        ),
        deterministic=False,
    )
    state = TrainState.create(
        apply_fn=model.apply, params=variables["params"], tx=optimizer
    )
    label_smoothing = float(config["LABEL_SMOOTHING"])

    @jax.jit
    def train_step(train_state, batch, dropout_rng):
        def objective(params):
            logits = model.apply(
                {"params": params},
                batch["context_states"],
                batch["context_actions"],
                batch["context_rewards"],
                batch["query_states"],
                deterministic=False,
                rngs={"dropout": dropout_rng},
            )
            return _loss_and_metrics(
                logits,
                batch["target_actions"],
                batch["target_mask"],
                label_smoothing,
                # PyTorch CrossEntropyLoss(reduction="sum") in the release.
                optimization_reduction="sum",
            )

        (_loss, metrics), gradients = jax.value_and_grad(objective, has_aux=True)(
            train_state.params
        )
        train_state = train_state.apply_gradients(grads=gradients)
        metrics = {
            **metrics,
            "gradient_norm": optax.global_norm(gradients),
            "learning_rate": schedule(train_state.step - 1),
        }
        return train_state, metrics

    @jax.jit
    def validation_step(params, batch):
        logits = jnp.asarray(
            model.apply(
                {"params": params},
                batch["context_states"],
                batch["context_actions"],
                batch["context_rewards"],
                batch["query_states"],
                deterministic=True,
            )
        )
        # The supplementary trainer uses all six query positions for training
        # but only the final query position for held-out early stopping.
        last_logits = jax.lax.dynamic_slice_in_dim(
            logits, logits.shape[1] - 1, 1, axis=1
        )
        last_actions = jax.lax.dynamic_slice_in_dim(
            batch["target_actions"], batch["target_actions"].shape[1] - 1, 1, axis=1
        )
        last_mask = jax.lax.dynamic_slice_in_dim(
            batch["target_mask"], batch["target_mask"].shape[1] - 1, 1, axis=1
        )
        return _loss_and_metrics(
            last_logits,
            last_actions,
            last_mask,
            label_smoothing,
        )[1]

    run_name = str(config.get("RUN_NAME") or f"coot-{layout}-seed{config['SEED']}")
    tags = [
        str(tag)
        for tag in dict.fromkeys(
            [*(config.get("WANDB_TAGS") or []), "CooT", "OvercookedV3"]
        )
    ]
    wandb_target = {}
    if not os.environ.get("WANDB_SWEEP_ID"):
        wandb_target = {
            "entity": config.get("ENTITY") or None,
            "project": config.get("PROJECT") or None,
        }
    run = wandb.init(
        **wandb_target,
        config=config,
        name=run_name,
        group=str(config.get("WANDB_GROUP") or f"coot-{layout}"),
        tags=tags,
        job_type="training",
        mode=_wandb_mode(config),
    )
    if run is None:
        raise RuntimeError("W&B did not create a training run")
    require_sweep_target(run, config)

    save_dir = Path(str(config["SAVES_DIR"])) / experiment_folder(config)
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_stem = f"coot_overcooked_v3_{layout}_seed{config['SEED']}"
    best_path = save_dir / f"{checkpoint_stem}_best.safetensors"
    final_path = save_dir / f"{checkpoint_stem}.safetensors"
    rng = np.random.default_rng(int(config["SEED"]))
    best_validation = float("inf")
    best_epoch = -1
    patience = 0

    print(
        f"[{_timestamp()}] CooT training: layout={layout} pairs={len(dataset.pairs)} "
        f"sequence={model_config.sequence_length} examples/epoch={examples_per_epoch}",
        flush=True,
    )
    for epoch in range(max_epochs):
        train_accumulator = []
        mask_count = _step_mask_count(config, epoch)
        for _ in range(steps_per_epoch):
            numpy_batch = dataset.sample_batch(
                rng,
                batch_size,
                split="train",
                context_episodes=model_config.context_episodes,
                num_query_states=model_config.num_query_states,
                rollout_masking=bool(config["ROLLOUT_MASKING"]),
                rollout_mask_exponent=float(config["ROLLOUT_MASK_EXPONENT"]),
                chunk_shuffle=bool(config["CHUNK_SHUFFLE"]),
                chunk_size=int(config["CHUNK_SIZE"]),
                step_mask_count=mask_count,
            )
            batch = {
                key: jnp.asarray(value)
                for key, value in numpy_batch.items()
                if key != "pair_index"
            }
            key, step_key = jax.random.split(key)
            state, metrics = train_step(state, batch, step_key)
            train_accumulator.append(jax.device_get(metrics))

        validation_accumulator = []
        for _ in range(validation_batches):
            numpy_batch = dataset.sample_batch(
                rng,
                batch_size,
                split="validation",
                context_episodes=model_config.context_episodes,
                num_query_states=model_config.num_query_states,
                rollout_masking=False,
                rollout_mask_exponent=float(config["ROLLOUT_MASK_EXPONENT"]),
                chunk_shuffle=False,
                chunk_size=int(config["CHUNK_SIZE"]),
            )
            batch = {
                key: jnp.asarray(value)
                for key, value in numpy_batch.items()
                if key != "pair_index"
            }
            validation_accumulator.append(
                jax.device_get(validation_step(state.params, batch))
            )

        train_metrics = {
            name: float(np.mean([entry[name] for entry in train_accumulator]))
            for name in train_accumulator[0]
        }
        validation_metrics = {
            name: float(np.mean([entry[name] for entry in validation_accumulator]))
            for name in validation_accumulator[0]
        }
        validation_loss = validation_metrics["loss"]
        wandb.log(
            {
                "epoch": epoch + 1,
                "train/loss": train_metrics["loss"],
                "train/action_accuracy": train_metrics["accuracy"],
                "train/action_entropy": train_metrics["entropy"],
                "train/gradient_norm": train_metrics["gradient_norm"],
                "train/learning_rate": train_metrics["learning_rate"],
                "train/masked_context_steps": mask_count,
                "validation/loss": validation_loss,
                "validation/action_accuracy": validation_metrics["accuracy"],
                "validation/action_entropy": validation_metrics["entropy"],
            }
        )
        print(
            f"[{_timestamp()}] epoch={epoch + 1} train={train_metrics['loss']:.5f} "
            f"validation={validation_loss:.5f}",
            flush=True,
        )

        if validation_loss < best_validation:
            best_validation = validation_loss
            best_epoch = epoch + 1
            patience = 0
            _save_checkpoint(
                best_path,
                state.params,
                model_config,
                config,
                dataset.metadata,
                epoch=best_epoch,
                validation_loss=best_validation,
            )
        else:
            patience += 1
            if patience >= int(config["PATIENCE"]):
                print(
                    f"[{_timestamp()}] early stopping at epoch {epoch + 1}; "
                    f"best epoch={best_epoch}",
                    flush=True,
                )
                break

    final_checkpoint, final_sidecar = _save_checkpoint(
        final_path,
        state.params,
        model_config,
        config,
        dataset.metadata,
        epoch=epoch + 1,
        validation_loss=validation_loss,
    )
    run.summary["checkpoint/best_epoch"] = best_epoch
    run.summary["checkpoint/best_validation_loss"] = best_validation
    run.summary["dataset/num_pairs"] = len(dataset.pairs)
    run.summary["dataset/num_train_pairs"] = len(dataset.train_pair_indices)
    run.summary["dataset/num_validation_pairs"] = len(dataset.validation_pair_indices)
    run.summary["dataset/sequence_length"] = model_config.sequence_length
    run.summary["runtime/context_update_steps"] = runtime_context_update_steps

    if config.get("upload_final_checkpoint", True):
        artifact = wandb.Artifact(
            f"coot-overcooked-v3-{run.id}-checkpoint",
            type="checkpoint",
            metadata={
                "algorithm": "CooT",
                "layout": layout,
                "seed": int(config["SEED"]),
                "best_epoch": best_epoch,
            },
        )
        for path in (
            best_path,
            best_path.with_suffix(".json"),
            final_checkpoint,
            final_sidecar,
        ):
            artifact.add_file(str(path), name=path.name)
        run.log_artifact(artifact, aliases=["final", "best"])

    # Persist the fully resolved Hydra config next to checkpoints even offline.
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    ) as config_file:
        config_file.write(OmegaConf.to_yaml(hydra_config, resolve=True))
        resolved_config_path = Path(config_file.name)
    target_config = save_dir / f"{checkpoint_stem}_config.yaml"
    target_config.write_text(
        resolved_config_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    resolved_config_path.unlink(missing_ok=True)
    wandb.finish()
    print(f"[{_timestamp()}] saved final CooT checkpoint: {final_checkpoint}")


if __name__ == "__main__":
    main()
