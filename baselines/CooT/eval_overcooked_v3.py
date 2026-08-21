"""Experimental held-out-partner adaptation evaluation for CooT on V3.

This is intentionally not the primary FCP/Self-play comparison. See TODO.md;
the headline evaluation is the seed-wise SP/XP matrix in
eval_crossplay_overcooked_v3.py.
"""

from __future__ import annotations

import csv
import json
import math
import os
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, cast

import hydra
import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import wandb  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402

import jaxmarl  # noqa: E402
from jaxmarl.wrappers.baselines import load_params  # noqa: E402

try:
    from .metrics import (
        adaptation_summary,
        jensen_shannon_divergence,
        recovery_episodes,
    )
    from .model import CooTConfig, CooTTransformer
    from .runtime import CheckpointPolicy, PolicySpec, load_pair_manifest
except ImportError:  # Direct execution: python baselines/CooT/<script>.py
    from metrics import adaptation_summary, jensen_shannon_divergence, recovery_episodes
    from model import CooTConfig, CooTTransformer
    from runtime import CheckpointPolicy, PolicySpec, load_pair_manifest


def _timestamp() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _br_proximity(episode_return: float, reference_return: float | None) -> float:
    if reference_return is None or not np.isfinite(reference_return):
        return float("nan")
    if abs(reference_return) < 1e-8:
        return float("nan")
    return float(episode_return) / float(reference_return)


def _fold_in_key(base_key: jax.Array, *indices: int) -> jax.Array:
    """Create stable per-condition RNG streams for common-random-number eval."""

    key = base_key
    for index in indices:
        key = jax.random.fold_in(key, int(index))
    return key


def _resolve_path(explicit, root, default_name, pattern) -> Path:
    if explicit:
        path = Path(str(explicit)).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    root = Path(str(root)).expanduser().resolve()
    direct = root / default_name
    if direct.is_file():
        return direct
    candidates = list(root.rglob(pattern))
    if not candidates:
        raise FileNotFoundError(f"No file matching {pattern!r} under {root}")
    return max(candidates, key=lambda path: (path.stat().st_mtime_ns, path.name))


class CooTController:
    """CooT inference state with a V3-specific N-step transition context.

    PORTING NOTE: the released evaluator inserts one complete episode into its
    five-episode FIFO only after that episode ends. Overcooked V3 can change its
    map inside an episode, so this controller commits every
    ``context_update_steps`` completed ``(state, action, reward)`` transitions.
    The Transformer input size and five episode-aligned slots are unchanged;
    the newest slot is a right-aligned partial trajectory for the current
    episode and is refreshed after each completed chunk.
    """

    def __init__(
        self,
        checkpoint: Path,
        *,
        stochastic: bool,
        context_ablation_stride: int,
        context_update_steps: int | None = None,
    ):
        sidecar = checkpoint.with_suffix(".json")
        if not sidecar.is_file():
            raise FileNotFoundError(f"CooT checkpoint sidecar not found: {sidecar}")
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        self.config = CooTConfig.from_dict(metadata["model_config"])
        self.model = CooTTransformer(self.config)
        self.params = load_params(checkpoint)
        self.stochastic = stochastic
        self.context_ablation_stride = max(0, int(context_ablation_stride))
        self.context_capacity = (
            self.config.context_episodes * self.config.episode_horizon
        )
        train_config = metadata.get("train_config", {})
        if not isinstance(train_config, dict):
            train_config = {}
        configured_update_steps = (
            train_config.get("RUNTIME_CONTEXT_UPDATE_STEPS", 20)
            if context_update_steps is None
            else context_update_steps
        )
        self.context_update_steps = int(configured_update_steps)
        if not 1 <= self.context_update_steps <= self.config.episode_horizon:
            raise ValueError(
                "context_update_steps must be in [1, "
                f"{self.config.episode_horizon}], got {self.context_update_steps}"
            )
        self._predict = jax.jit(self._predict_impl)
        self.reset_context()

    def _predict_impl(
        self, params, context_states, context_actions, context_rewards, query_states
    ):
        outputs = jnp.asarray(
            self.model.apply(
                {"params": params},
                context_states[None, ...],
                context_actions[None, ...],
                context_rewards[None, ...],
                query_states[None, ...],
                deterministic=True,
            )
        )
        logits = jax.lax.dynamic_index_in_dim(
            jnp.squeeze(outputs, axis=0), -1, axis=0, keepdims=False
        )
        return logits, jax.nn.softmax(logits)

    def reset_context(self) -> None:
        self.context_states = np.zeros(
            (self.context_capacity, self.config.observation_dim), dtype=np.float32
        )
        self.context_actions = np.zeros(
            (self.context_capacity, self.config.action_dim), dtype=np.float32
        )
        self.context_rewards = np.zeros((self.context_capacity, 1), dtype=np.float32)
        self.context_slot_lengths = [0] * self.config.context_episodes
        self.context_filled_steps = 0
        self.context_updates = 0
        self.query_states = np.zeros(
            (self.config.num_query_states, self.config.observation_dim),
            dtype=np.float32,
        )
        self.pending_states: list[np.ndarray] = []
        self.pending_actions: list[int] = []
        self.pending_rewards: list[float] = []
        self.current_context_slot_open = False
        self.action_count = 0

    def start_episode(self) -> None:
        if self.pending_states or self.pending_actions or self.pending_rewards:
            raise RuntimeError(
                "finish_episode() must flush pending CooT transitions before "
                "start_episode()"
            )
        self.current_context_slot_open = False
        self.query_states.fill(0.0)

    def act(self, observation, key) -> tuple[int, np.ndarray, float]:
        flat_observation = np.asarray(observation, dtype=np.float32).reshape(-1)
        if flat_observation.size != self.config.observation_dim:
            raise ValueError(
                f"Evaluation observation has {flat_observation.size} values; "
                f"checkpoint expects {self.config.observation_dim}"
            )
        self.query_states[:-1] = self.query_states[1:]
        self.query_states[-1] = flat_observation
        logits, probabilities = self._predict(
            self.params,
            jnp.asarray(self.context_states),
            jnp.asarray(self.context_actions),
            jnp.asarray(self.context_rewards),
            jnp.asarray(self.query_states),
        )
        probabilities = np.asarray(jax.device_get(probabilities))
        context_jsd = float("nan")
        if (
            self.context_ablation_stride
            and self.action_count % self.context_ablation_stride == 0
        ):
            _empty_logits, empty_probabilities = self._predict(
                self.params,
                jnp.zeros_like(jnp.asarray(self.context_states)),
                jnp.zeros_like(jnp.asarray(self.context_actions)),
                jnp.zeros_like(jnp.asarray(self.context_rewards)),
                jnp.asarray(self.query_states),
            )
            context_jsd = jensen_shannon_divergence(
                probabilities, np.asarray(jax.device_get(empty_probabilities))
            )
        self.action_count += 1
        action = (
            int(jax.random.categorical(key, logits))
            if self.stochastic
            else int(np.argmax(probabilities))
        )
        self.pending_states.append(flat_observation)
        self.pending_actions.append(action)
        return action, probabilities, context_jsd

    def observe_reward(self, reward: float) -> None:
        if len(self.pending_rewards) >= len(self.pending_actions):
            raise RuntimeError("observe_reward() must follow exactly one CooT action")
        self.pending_rewards.append(float(reward))
        self._commit_ready_context(force=False)

    def _append_context(self, length: int) -> None:
        """Commit completed transitions into this episode's newest slot."""

        states = np.asarray(self.pending_states[:length], dtype=np.float32)
        action_indices = np.asarray(self.pending_actions[:length], dtype=np.int32)
        actions = np.eye(self.config.action_dim, dtype=np.float32)[action_indices]
        rewards = np.asarray(self.pending_rewards[:length], dtype=np.float32).reshape(
            length, 1
        )

        horizon = self.config.episode_horizon
        if not self.current_context_slot_open:
            # Evict one old rollout exactly once per episode. Keeping the four
            # remaining rollout slots aligned preserves the release's learned
            # positional structure more closely than shifting by N tokens.
            self.context_states[:-horizon] = self.context_states[horizon:].copy()
            self.context_actions[:-horizon] = self.context_actions[horizon:].copy()
            self.context_rewards[:-horizon] = self.context_rewards[horizon:].copy()
            self.context_states[-horizon:] = 0.0
            self.context_actions[-horizon:] = 0.0
            self.context_rewards[-horizon:] = 0.0
            self.context_slot_lengths = [*self.context_slot_lengths[1:], 0]
            self.current_context_slot_open = True

        committed_length = self.context_slot_lengths[-1]
        updated_length = committed_length + length
        if updated_length > horizon:
            raise RuntimeError(
                "CooT received more transitions than the checkpoint episode horizon"
            )
        if committed_length:
            states = np.concatenate(
                [self.context_states[-committed_length:].copy(), states], axis=0
            )
            actions = np.concatenate(
                [self.context_actions[-committed_length:].copy(), actions], axis=0
            )
            rewards = np.concatenate(
                [self.context_rewards[-committed_length:].copy(), rewards], axis=0
            )

        # Left-pad the partial rollout so its latest completed transition stays
        # adjacent to the query positions, like a full rollout in the release.
        self.context_states[-horizon:] = 0.0
        self.context_actions[-horizon:] = 0.0
        self.context_rewards[-horizon:] = 0.0
        self.context_states[-updated_length:] = states
        self.context_actions[-updated_length:] = actions
        self.context_rewards[-updated_length:] = rewards
        self.context_slot_lengths[-1] = updated_length
        self.context_filled_steps = sum(self.context_slot_lengths)
        self.context_updates += 1

        del self.pending_states[:length]
        del self.pending_actions[:length]
        del self.pending_rewards[:length]

        # A committed chunk is a pseudo-episode boundary. Clearing the query
        # prevents the same states appearing in both context and query and
        # matches the release's query reset after a completed episode.
        self.query_states.fill(0.0)

    def _commit_ready_context(self, *, force: bool) -> None:
        if not (
            len(self.pending_states)
            == len(self.pending_actions)
            == len(self.pending_rewards)
        ):
            if force:
                raise RuntimeError(
                    "CooT episode ended with incomplete (state, action, reward) data"
                )
            return

        while len(self.pending_rewards) >= self.context_update_steps:
            self._append_context(self.context_update_steps)
        if force and self.pending_rewards:
            self._append_context(len(self.pending_rewards))

    def finish_episode(self) -> None:
        self._commit_ready_context(force=True)


class FixedController:
    """Non-adaptive FCP/SP controller for matched-protocol comparisons."""

    def __init__(self, spec: PolicySpec, action_dim: int):
        self.policy = CheckpointPolicy(spec, action_dim)
        self.context_filled_steps = 0
        self.context_updates = 0

    def reset_context(self) -> None:
        self.context_filled_steps = 0
        self.context_updates = 0

    def start_episode(self) -> None:
        self.policy.reset()

    def act(self, observation, key) -> tuple[int, np.ndarray, float]:
        action, probabilities = self.policy.act(observation, key)
        return action, probabilities, float("nan")

    def observe_reward(self, reward: float) -> None:
        del reward

    def finish_episode(self) -> None:
        pass


def run_episode(env, ego, partner, key, max_steps: int):
    key, reset_key = jax.random.split(key)
    obs, state = env.reset(reset_key)
    ego.start_episode()
    partner.reset()
    env_step = jax.jit(env.step_env)
    episode_return = 0.0
    entropies = []
    context_jsds = []
    action_counts = Counter()
    layout_changes = 0
    recipe_changes = 0

    for step in range(max_steps):
        key, ego_key, partner_key, step_key = jax.random.split(key, 4)
        partner_action, _partner_probabilities = partner.act(
            obs["agent_0"], partner_key
        )
        ego_action, probabilities, context_jsd = ego.act(obs["agent_1"], ego_key)
        obs, state, rewards, dones, info = env_step(
            step_key,
            state,
            {"agent_0": partner_action, "agent_1": ego_action},
        )
        reward = float(rewards["agent_1"])
        ego.observe_reward(reward)
        episode_return += reward
        action_counts[ego_action] += 1
        probabilities = np.clip(probabilities, 1e-8, 1.0)
        entropies.append(float(-np.sum(probabilities * np.log(probabilities))))
        if np.isfinite(context_jsd):
            context_jsds.append(context_jsd)
        if "layout_changed" in info:
            layout_changes += int(np.asarray(info["layout_changed"])[1])
        if "recipe_changed" in info:
            recipe_changes += int(np.asarray(info["recipe_changed"])[1])
        if bool(dones["__all__"]):
            break

    ego.finish_episode()
    distribution = np.array(
        [action_counts[index] for index in range(env.action_space("agent_1").n)],
        dtype=np.float64,
    )
    distribution /= max(1.0, distribution.sum())
    diagnostics = {
        "episode_return": episode_return,
        "episode_length": step + 1,
        "action_entropy": float(np.mean(entropies)),
        "context_action_jsd": (
            float(np.mean(context_jsds)) if context_jsds else float("nan")
        ),
        "layout_changes": layout_changes,
        "recipe_changes": recipe_changes,
        "context_filled_steps": int(ego.context_filled_steps),
        "context_updates": int(ego.context_updates),
        "action_distribution": distribution.tolist(),
    }
    return diagnostics, key


def _log_episode(record: dict[str, Any], global_step: int) -> None:
    wandb.log(
        {
            "eval/global_episode": global_step,
            "eval/episode_return": record["episode_return"],
            "eval/br_proximity": record["br_proximity"],
            "eval/training_seed": record["training_seed"],
            "eval/evaluation_seed": record["evaluation_seed"],
            "eval/episode_length": record["episode_length"],
            "eval/episode_with_partner": record["episode_with_partner"],
            "eval/partner_index": record["partner_index"],
            "eval/is_switch_run": int(record["phase"] == "switch"),
            "debug/action_entropy": record["action_entropy"],
            "debug/context_action_jsd": record["context_action_jsd"],
            "debug/context_filled_steps": record["context_filled_steps"],
            "debug/context_updates": record["context_updates"],
            "debug/layout_changes": record["layout_changes"],
            "debug/recipe_changes": record["recipe_changes"],
        },
        step=global_step,
    )


def _write_records(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    columns = [key for key in records[0] if key != "action_distribution"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record[key] for key in columns})


def _plot_fixed(records, output_dir: Path) -> list[Path]:
    fixed = [record for record in records if record["phase"] == "fixed"]
    if not fixed:
        return []
    method = str(fixed[0]["method"])
    partners = list(dict.fromkeys(record["partner_id"] for record in fixed))
    episode_count = max(record["episode_with_partner"] for record in fixed)
    matrix = np.full((len(partners), episode_count), np.nan)
    entropy = np.full_like(matrix, np.nan)
    jsd = np.full_like(matrix, np.nan)
    proximity = np.full_like(matrix, np.nan)
    partner_index = {partner: index for index, partner in enumerate(partners)}
    for record in fixed:
        row = partner_index[record["partner_id"]]
        column = record["episode_with_partner"] - 1
        matrix[row, column] = record["episode_return"]
        entropy[row, column] = record["action_entropy"]
        jsd[row, column] = record["context_action_jsd"]
        proximity[row, column] = record["br_proximity"]

    paths = []
    episodes = np.arange(1, episode_count + 1)
    mean = np.nanmean(matrix, axis=0)
    stderr = np.nanstd(matrix, axis=0) / math.sqrt(max(1, matrix.shape[0]))
    fig, axis = plt.subplots(figsize=(7, 4))
    axis.plot(episodes, mean, color="#1f77b4", label="mean return")
    axis.fill_between(episodes, mean - stderr, mean + stderr, alpha=0.2)
    axis.set(
        xlabel="Episode with partner",
        ylabel="Sparse return",
        title=f"{method} adaptation",
    )
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    path = output_dir / "adaptation_curve.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    if np.isfinite(proximity).any():
        proximity_mean = np.nanmean(proximity, axis=0)
        proximity_stderr = np.nanstd(proximity, axis=0) / math.sqrt(
            max(1, proximity.shape[0])
        )
        fig, axis = plt.subplots(figsize=(7, 4))
        axis.plot(episodes, proximity_mean, color="#9467bd")
        axis.fill_between(
            episodes,
            proximity_mean - proximity_stderr,
            proximity_mean + proximity_stderr,
            alpha=0.2,
        )
        axis.axhline(1.0, color="black", linestyle="--", linewidth=1)
        axis.set(
            xlabel="Episode with partner",
            ylabel="BR-Proximity",
            title=f"{method} normalized adaptation",
        )
        axis.grid(alpha=0.25)
        fig.tight_layout()
        path = output_dir / "br_proximity_curve.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)

    fig, axis = plt.subplots(
        figsize=(max(7, episode_count * 0.16), max(3, len(partners) * 0.35))
    )
    image = axis.imshow(matrix, aspect="auto", interpolation="nearest", cmap="viridis")
    axis.set(xlabel="Episode", ylabel="Partner", title="Partner-wise episode return")
    axis.set_yticks(np.arange(len(partners)), labels=partners)
    fig.colorbar(image, ax=axis, label="Sparse return")
    fig.tight_layout()
    path = output_dir / "partner_episode_heatmap.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    axes[0].plot(episodes, np.nanmean(entropy, axis=0), color="#ff7f0e")
    axes[0].set(ylabel="Action entropy", title=f"{method} inference diagnostics")
    axes[1].plot(episodes, np.nanmean(jsd, axis=0), color="#2ca02c")
    axes[1].set(xlabel="Episode with partner", ylabel="Context-vs-empty JSD")
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.tight_layout()
    path = output_dir / "context_diagnostics.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)
    return paths


def _plot_switch(records, output_dir: Path, switch_interval: int) -> Path | None:
    switched = [record for record in records if record["phase"] == "switch"]
    if not switched:
        return None
    schedules = defaultdict(list)
    for record in switched:
        schedules[record["schedule_id"]].append(record)
    fig, axis = plt.subplots(figsize=(8, 4.5))
    for schedule_id, schedule_records in schedules.items():
        schedule_records.sort(key=lambda item: item["schedule_episode"])
        axis.plot(
            [record["schedule_episode"] for record in schedule_records],
            [record["episode_return"] for record in schedule_records],
            alpha=0.65,
            label=schedule_id,
        )
    axis.axvline(
        switch_interval + 0.5, color="black", linestyle="--", label="partner switch"
    )
    axis.set(
        xlabel="Episode in switch schedule",
        ylabel="Sparse return",
        title="Sudden partner-switch recovery",
    )
    axis.grid(alpha=0.25)
    if len(schedules) <= 8:
        axis.legend(fontsize=8)
    fig.tight_layout()
    path = output_dir / "partner_switch_recovery.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


@hydra.main(
    version_base=None,
    config_path="../../conf",
    config_name="coot_eval_overcooked_v3",
)
def main(hydra_config: DictConfig) -> None:
    raw_config = OmegaConf.to_container(hydra_config, resolve=True)
    if not isinstance(raw_config, dict):
        raise TypeError("Resolved evaluation config must be a mapping")
    config = cast(dict[str, Any], raw_config)
    layout = str(config["ENV_KWARGS"]["layout"])
    training_seed = int(config.get("TRAINING_SEED", config.get("SEED", 0)))
    evaluation_seed = int(config.get("EVALUATION_SEED", 0))
    manifest_path = _resolve_path(
        config.get("PARTNER_MANIFEST"),
        config["PARTNER_MANIFEST_ROOT"],
        f"{layout}.json",
        f"*{layout}*.json",
    )
    manifest, resolved_manifest_path = load_pair_manifest(manifest_path)
    requested_split = str(config.get("PARTNER_SPLIT", "test")).lower()
    all_pairs = list(manifest["pairs"])
    pairs = [
        pair
        for pair in all_pairs
        if str(pair.get("split", "")).lower() == requested_split
    ]
    if not pairs and bool(config.get("REQUIRE_PARTNER_SPLIT", True)):
        raise ValueError(
            f"Evaluation manifest has no split={requested_split!r} partners; "
            "final adaptation evaluation must not reuse train/validation pairs"
        )
    if not pairs:
        pairs = all_pairs
    partner_limit = config.get("PARTNER_LIMIT")
    if partner_limit is not None:
        pairs = pairs[: int(partner_limit)]
    if not pairs:
        raise ValueError("No evaluation partners remain after PARTNER_LIMIT")
    if bool(config.get("REQUIRE_REFERENCE_RETURN", True)):
        missing_references = [
            str(pair.get("id", index))
            for index, pair in enumerate(pairs)
            if pair.get("reference_return") is None
        ]
        if missing_references:
            raise ValueError(
                "Paper-matched BR-Proximity/recovery needs one method-independent "
                "reference_return per test partner; missing: "
                + ", ".join(missing_references)
            )
    if int(config["EPISODES_PER_PARTNER"]) < 1:
        raise ValueError("EPISODES_PER_PARTNER must be positive")
    if int(config["MAX_STEPS"]) < 1:
        raise ValueError("MAX_STEPS must be positive")
    if int(config["CONTEXT_UPDATE_STEPS"]) < 1:
        raise ValueError("CONTEXT_UPDATE_STEPS must be positive")
    if int(config["SWITCH_INTERVAL"]) < 1:
        raise ValueError("SWITCH_INTERVAL must be positive")

    env = jaxmarl.make(
        "overcooked_v3",
        layout=layout,
        max_steps=int(config["MAX_STEPS"]),
        random_agent_positions=bool(config["ENV_KWARGS"]["random_agent_positions"]),
        include_transition_countdown=bool(
            config["ENV_KWARGS"]["include_transition_countdown"]
        ),
        include_layout_change_mask=bool(
            config["ENV_KWARGS"]["include_layout_change_mask"]
        ),
        transition_warning_steps=int(config["ENV_KWARGS"]["transition_warning_steps"]),
    )
    action_dim = int(env.action_space("agent_1").n)
    ego_kind = str(config["EGO_KIND"]).lower()
    method = str(config.get("METHOD") or ("CooT" if ego_kind == "coot" else "fixed"))
    if ego_kind == "coot":
        checkpoint = _resolve_path(
            config.get("CHECKPOINT"),
            config["CHECKPOINT_ROOT"],
            "__missing__",
            f"coot_overcooked_v3_{layout}_seed{training_seed}_best.safetensors",
        )
        ego = CooTController(
            checkpoint,
            stochastic=bool(config["STOCHASTIC"]),
            context_ablation_stride=int(config["CONTEXT_ABLATION_STRIDE"]),
            context_update_steps=int(config["CONTEXT_UPDATE_STEPS"]),
        )
        if ego.config.episode_horizon != int(config["MAX_STEPS"]):
            raise ValueError(
                f"Checkpoint horizon {ego.config.episode_horizon} does not match "
                f"MAX_STEPS={int(config['MAX_STEPS'])}"
            )
    elif ego_kind == "fixed":
        fixed_checkpoint = config.get("FIXED_EGO_CHECKPOINT")
        if not fixed_checkpoint:
            architecture = str(config["FIXED_EGO_ARCHITECTURE"]).lower()
            vmap_index = int(config["FIXED_EGO_VMAP_INDEX"])
            configured_pattern = config.get("FIXED_EGO_CHECKPOINT_PATTERN")
            if configured_pattern:
                checkpoint_pattern = str(configured_pattern).format(
                    layout=layout,
                    seed=training_seed,
                    architecture=architecture,
                    vmap_index=vmap_index,
                )
            else:
                method_key = method.casefold().replace("-", "").replace("_", "")
                if method_key == "fcp":
                    prefix = "fcp"
                elif method_key in {"selfplay", "sp", "ippo"}:
                    prefix = "ippo"
                else:
                    raise ValueError(
                        "Set FIXED_EGO_CHECKPOINT or "
                        "FIXED_EGO_CHECKPOINT_PATTERN for this fixed method"
                    )
                checkpoint_pattern = (
                    f"{prefix}_{architecture}_overcooked_v3_{layout}_"
                    f"seed{training_seed}_vmap{vmap_index}.safetensors"
                )
            fixed_checkpoint = _resolve_path(
                None,
                config["FIXED_EGO_CHECKPOINT_ROOT"],
                "__missing__",
                checkpoint_pattern,
            )
        fixed_spec = PolicySpec.from_mapping(
            {
                "checkpoint": fixed_checkpoint,
                "architecture": config["FIXED_EGO_ARCHITECTURE"],
                "activation": config["FIXED_EGO_ACTIVATION"],
                "fc_dim_size": config["FIXED_EGO_FC_DIM_SIZE"],
                "gru_hidden_dim": config["FIXED_EGO_GRU_HIDDEN_DIM"],
                "stochastic": config["STOCHASTIC"],
            },
            base_dir=Path.cwd(),
        )
        ego = FixedController(fixed_spec, action_dim)
        checkpoint = fixed_spec.checkpoint
    else:
        raise ValueError("EGO_KIND must be 'coot' or 'fixed'")

    raw_mode = str(config.get("wandb_mode", "online")).lower()
    if raw_mode not in {"online", "offline", "disabled"}:
        raise ValueError("wandb_mode must be online, offline, or disabled")
    mode = cast(Literal["online", "offline", "disabled"], raw_mode)
    if mode == "online" and not os.environ.get("WANDB_API_KEY", "").strip():
        mode = "offline"
    target = {}
    if not os.environ.get("WANDB_SWEEP_ID"):
        target = {
            "entity": config.get("ENTITY") or None,
            "project": config.get("PROJECT") or None,
        }
    run = wandb.init(
        **target,
        config=config,
        name=str(
            config.get("RUN_NAME")
            or f"{method}-{layout}-trainseed{training_seed}-adaptation"
        ),
        group=str(config.get("WANDB_GROUP") or f"adaptation-{layout}"),
        tags=list(
            dict.fromkeys([*(config.get("WANDB_TAGS") or []), method, "AdaptationEval"])
        ),
        job_type="evaluation",
        mode=mode,
    )
    if run is None:
        raise RuntimeError("W&B did not create an evaluation run")
    wandb.define_metric("eval/global_episode")
    wandb.define_metric("eval/*", step_metric="eval/global_episode")
    wandb.define_metric("debug/*", step_metric="eval/global_episode")

    output_dir = (
        Path(str(config["OUTPUT_DIR"]))
        / f"{method}_{layout}_trainseed{training_seed}_{run.id}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    base_key = jax.random.PRNGKey(evaluation_seed)
    records = []
    summaries = {}
    references = {}
    global_step = 0

    for partner_index, pair in enumerate(pairs):
        pair_id = str(pair.get("id", partner_index))
        partner_spec = PolicySpec.from_mapping(
            pair["partner"], base_dir=resolved_manifest_path.parent
        )
        partner = CheckpointPolicy(partner_spec, action_dim)
        ego.reset_context()
        partner_returns = []
        partner_record_start = len(records)
        configured_reference = pair.get("reference_return")
        if configured_reference is not None:
            configured_reference = float(configured_reference)
        for episode in range(int(config["EPISODES_PER_PARTNER"])):
            episode_key = _fold_in_key(base_key, 0, partner_index, episode)
            diagnostics, _unused_key = run_episode(
                env, ego, partner, episode_key, int(config["MAX_STEPS"])
            )
            global_step += 1
            record = {
                **diagnostics,
                "method": method,
                "layout": layout,
                "training_seed": training_seed,
                "evaluation_seed": evaluation_seed,
                "phase": "fixed",
                "br_proximity": _br_proximity(
                    diagnostics["episode_return"], configured_reference
                ),
                "partner_id": pair_id,
                "partner_index": partner_index,
                "episode_with_partner": episode + 1,
                "schedule_id": "",
                "schedule_episode": 0,
            }
            records.append(record)
            partner_returns.append(diagnostics["episode_return"])
            _log_episode(record, global_step)
        summary = adaptation_summary(
            partner_returns,
            early_window=int(config["EARLY_WINDOW"]),
            late_window=int(config["LATE_WINDOW"]),
            slope_episodes=int(config["SLOPE_EPISODES"]),
            rolling_window=int(config["TARGET_ROLLING_WINDOW"]),
            target_fraction=float(config["ADAPTATION_TARGET_FRACTION"]),
        )
        summaries[pair_id] = summary
        references[pair_id] = (
            float(configured_reference)
            if configured_reference is not None
            else summary["final_return"]
        )
        for record in records[partner_record_start:]:
            record["br_proximity"] = _br_proximity(
                record["episode_return"], references[pair_id]
            )
        summary["initial_br_proximity"] = _br_proximity(
            summary["initial_return"], references[pair_id]
        )
        summary["final_br_proximity"] = _br_proximity(
            summary["final_return"], references[pair_id]
        )
        for metric, value in summary.items():
            if np.isfinite(value):
                run.summary[f"partner/{pair_id}/{metric}"] = value

    recovery = {}
    switch_interval = int(config["SWITCH_INTERVAL"])
    if bool(config["SWITCH_EVAL"]) and len(pairs) >= 2:
        pairs_by_id = {
            str(pair.get("id", index)): pair for index, pair in enumerate(pairs)
        }
        configured_pairs = manifest.get("switch_pairs") or []
        if configured_pairs:
            switch_pairs = [(str(left), str(right)) for left, right in configured_pairs]
        else:
            ids = list(pairs_by_id)
            switch_pairs = []
            for index in range(len(ids) - 1):
                switch_pairs.extend(
                    [(ids[index], ids[index + 1]), (ids[index + 1], ids[index])]
                )
        switch_pairs = switch_pairs[: int(config["MAX_SWITCH_PAIRS"])]

        for schedule_index, (left_id, right_id) in enumerate(switch_pairs):
            if left_id not in pairs_by_id or right_id not in pairs_by_id:
                raise ValueError(f"Unknown switch pair: {left_id!r}, {right_id!r}")
            schedule_id = f"{left_id}->{right_id}"
            ego.reset_context()
            schedule_records = []
            for block_index, partner_id in enumerate((left_id, right_id)):
                pair = pairs_by_id[partner_id]
                partner = CheckpointPolicy(
                    PolicySpec.from_mapping(
                        pair["partner"], base_dir=resolved_manifest_path.parent
                    ),
                    action_dim,
                )
                for episode in range(switch_interval):
                    episode_key = _fold_in_key(
                        base_key, 1, schedule_index, block_index, episode
                    )
                    diagnostics, _unused_key = run_episode(
                        env, ego, partner, episode_key, int(config["MAX_STEPS"])
                    )
                    global_step += 1
                    record = {
                        **diagnostics,
                        "method": method,
                        "layout": layout,
                        "training_seed": training_seed,
                        "evaluation_seed": evaluation_seed,
                        "phase": "switch",
                        "br_proximity": _br_proximity(
                            diagnostics["episode_return"], references[partner_id]
                        ),
                        "partner_id": partner_id,
                        "partner_index": list(pairs_by_id).index(partner_id),
                        "episode_with_partner": episode + 1,
                        "schedule_id": schedule_id,
                        "schedule_episode": block_index * switch_interval + episode + 1,
                    }
                    records.append(record)
                    schedule_records.append(record)
                    _log_episode(record, global_step)
            post_switch = [
                record["episode_return"]
                for record in schedule_records[switch_interval:]
            ]
            recovery_value = recovery_episodes(
                post_switch,
                references[right_id],
                target_fraction=float(config["RECOVERY_TARGET_FRACTION"]),
                rolling_window=int(config["RECOVERY_ROLLING_WINDOW"]),
            )
            recovery[schedule_id] = recovery_value
            if np.isfinite(recovery_value):
                run.summary[f"switch/{schedule_id}/recovery_episodes"] = recovery_value

    finite_recovery = [value for value in recovery.values() if np.isfinite(value)]
    finite_targets = [
        item["episodes_to_target"]
        for item in summaries.values()
        if np.isfinite(item["episodes_to_target"])
    ]
    final_proximities = [
        item["final_br_proximity"]
        for item in summaries.values()
        if np.isfinite(item["final_br_proximity"])
    ]
    aggregate = {
        "mean_initial_return": float(
            np.mean([item["initial_return"] for item in summaries.values()])
        ),
        "mean_final_return": float(
            np.mean([item["final_return"] for item in summaries.values()])
        ),
        "mean_adaptation_gain": float(
            np.mean([item["absolute_gain"] for item in summaries.values()])
        ),
        "mean_relative_gain": float(
            np.mean([item["relative_gain"] for item in summaries.values()])
        ),
        "mean_early_slope": float(
            np.mean([item["early_slope"] for item in summaries.values()])
        ),
        "mean_return_auc": float(
            np.mean([item["return_auc"] for item in summaries.values()])
        ),
        "mean_final_br_proximity": (
            float(np.mean(final_proximities)) if final_proximities else float("nan")
        ),
        "mean_episodes_to_target": (
            float(np.mean(finite_targets)) if finite_targets else float("nan")
        ),
        "mean_recovery_episodes": float(np.mean(finite_recovery))
        if finite_recovery
        else float("nan"),
    }
    for name, value in aggregate.items():
        if np.isfinite(value):
            run.summary[f"adaptation/{name}"] = value
    run.summary["checkpoint"] = str(checkpoint)
    run.summary["partner_manifest"] = str(resolved_manifest_path)
    run.summary["training_seed"] = training_seed
    run.summary["evaluation_seed"] = evaluation_seed
    run.summary["protocol/context_update_steps"] = int(config["CONTEXT_UPDATE_STEPS"])

    _write_records(output_dir / "episodes.csv", records)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "method": method,
                "layout": layout,
                "training_seed": training_seed,
                "evaluation_seed": evaluation_seed,
                "context_update_steps": int(config["CONTEXT_UPDATE_STEPS"]),
                "checkpoint": str(checkpoint),
                "partner_summaries": summaries,
                "reference_returns": references,
                "switch_recovery": recovery,
                "aggregate": aggregate,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    plot_paths = _plot_fixed(records, output_dir)
    switch_plot = _plot_switch(records, output_dir, switch_interval)
    if switch_plot is not None:
        plot_paths.append(switch_plot)
    wandb.log(
        {f"plots/{path.stem}": wandb.Image(str(path)) for path in plot_paths},
        step=global_step,
    )
    table_columns = [
        "phase",
        "training_seed",
        "evaluation_seed",
        "partner_id",
        "episode_with_partner",
        "schedule_id",
        "schedule_episode",
        "episode_return",
        "br_proximity",
        "action_entropy",
        "context_action_jsd",
        "context_filled_steps",
        "context_updates",
    ]
    wandb.log(
        {
            "tables/episode_adaptation": wandb.Table(
                columns=cast(Any, table_columns),
                data=[
                    [record[column] for column in table_columns] for record in records
                ],
            )
        },
        step=global_step,
    )
    artifact = wandb.Artifact(
        f"{method.lower()}-{layout}-{run.id}-adaptation-results",
        type="evaluation",
    )
    artifact.add_dir(str(output_dir))
    run.log_artifact(artifact, aliases=["latest"])
    wandb.finish()
    print(f"[{_timestamp()}] saved adaptation evaluation: {output_dir}")


if __name__ == "__main__":
    main()
