"""Sharded rollout storage and paper-faithful CooT batch sampling."""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Literal

import numpy as np


DATASET_FORMAT_VERSION = 1


def write_metadata(dataset_dir: Path, metadata: dict[str, Any]) -> Path:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    output = dataset_dir / "metadata.json"
    payload = {"format_version": DATASET_FORMAT_VERSION, **metadata}
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return output


def write_pair_shard(
    dataset_dir: Path,
    pair_index: int,
    observations: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    lengths: np.ndarray,
) -> Path:
    """Write one policy pair at a time to keep V3 datasets manageable."""

    dataset_dir.mkdir(parents=True, exist_ok=True)
    output = dataset_dir / f"pair_{pair_index:03d}.npz"
    np.savez_compressed(
        output,
        observations=np.asarray(observations, dtype=np.float16),
        actions=np.asarray(actions, dtype=np.uint8),
        rewards=np.asarray(rewards, dtype=np.float32),
        lengths=np.asarray(lengths, dtype=np.int32),
    )
    return output


class CooTShardDataset:
    """Lazy sampler over per-partner rollout shards.

    PORTING NOTE: the released implementation materializes every context 70 times. V3 grids
    and 450-step episodes make that prohibitively large, so this port stores raw
    trajectories once and samples the same ``M x K x L`` distribution online.
    This is a storage-only change; the online sampler preserves context/query
    independence, targets, and the marginal masking distribution.
    """

    def __init__(self, dataset_dir: str | Path, cache_size: int = 2):
        self.dataset_dir = Path(dataset_dir).expanduser().resolve()
        metadata_path = self.dataset_dir / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"CooT metadata not found: {metadata_path}")
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if self.metadata.get("format_version") != DATASET_FORMAT_VERSION:
            raise ValueError(
                "Unsupported CooT dataset format: "
                f"{self.metadata.get('format_version')}"
            )
        self.pairs = list(self.metadata.get("pairs") or [])
        if not self.pairs:
            raise ValueError("CooT dataset contains no policy pairs")
        self.train_pair_indices = [
            index
            for index, pair in enumerate(self.pairs)
            if str(pair.get("split", "train")).lower() != "validation"
        ]
        self.validation_pair_indices = [
            index
            for index, pair in enumerate(self.pairs)
            if str(pair.get("split", "train")).lower() == "validation"
        ]
        if not self.train_pair_indices:
            raise ValueError("CooT dataset contains no training policy pairs")
        self.action_dim = int(self.metadata["action_dim"])
        self.horizon = int(self.metadata["episode_horizon"])
        self.observation_shape = tuple(self.metadata["observation_shape"])
        self.observation_dim = int(np.prod(self.observation_shape))
        self.validation_fraction = float(self.metadata.get("validation_fraction", 0.1))
        self.cache_size = max(1, int(cache_size))
        self._cache: OrderedDict[int, dict[str, np.ndarray]] = OrderedDict()

    def _load_pair(self, pair_index: int) -> dict[str, np.ndarray]:
        cached = self._cache.pop(pair_index, None)
        if cached is not None:
            self._cache[pair_index] = cached
            return cached
        path = self.dataset_dir / self.pairs[pair_index]["shard"]
        if not path.is_file():
            raise FileNotFoundError(f"CooT pair shard not found: {path}")
        with np.load(path, allow_pickle=False) as archive:
            shard = {key: archive[key] for key in archive.files}
        expected = {"observations", "actions", "rewards", "lengths"}
        missing = expected - shard.keys()
        if missing:
            raise ValueError(f"{path} is missing arrays: {sorted(missing)}")
        self._cache[pair_index] = shard
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return shard

    def _candidate_pair_indices(
        self, split: Literal["train", "validation"]
    ) -> list[int]:
        if split == "train":
            return self.train_pair_indices
        # PORTING NOTE: the release validates on five held-out partner/BR pairs.
        # When a manifest provides them, use every trajectory from those pairs.
        # Small local datasets may omit them; only then do we use a trajectory
        # holdout from the training pairs so the baseline remains runnable.
        return self.validation_pair_indices or self.train_pair_indices

    def _split_indices(
        self,
        pair_index: int,
        shard: dict[str, np.ndarray],
        split: Literal["train", "validation"],
    ) -> np.ndarray:
        count = int(shard["observations"].shape[0])
        if self.validation_pair_indices:
            pair_split = str(self.pairs[pair_index].get("split", "train")).lower()
            expected_split = "validation" if split == "validation" else "train"
            if pair_split != expected_split:
                raise ValueError(
                    f"Pair {pair_index} belongs to {pair_split}, not {expected_split}"
                )
            return np.arange(count)
        validation_count = max(1, int(round(count * self.validation_fraction)))
        if count - validation_count < 2:
            raise ValueError(
                "Each CooT pair needs at least three trajectories after collection"
            )
        if split == "train":
            return np.arange(0, count - validation_count)
        return np.arange(count - validation_count, count)

    @staticmethod
    def _shuffle_episode_chunks(
        rng: np.random.Generator,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        chunk_size: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Match the supplementary 20-step chunk shuffle.

        The first chunk remains fixed and the remaining chunks are permuted.
        """

        horizon = states.shape[0]
        if chunk_size <= 1 or horizon <= chunk_size:
            return states, actions, rewards
        chunks = [
            np.arange(start, min(start + chunk_size, horizon))
            for start in range(0, horizon, chunk_size)
        ]
        ordering = [
            chunks[0],
            *[chunks[index] for index in rng.permutation(np.arange(1, len(chunks)))],
        ]
        indices = np.concatenate(ordering)
        return states[indices], actions[indices], rewards[indices]

    @staticmethod
    def _rollout_mask_count(
        rng: np.random.Generator, context_episodes: int, exponent: float
    ) -> int:
        counts = np.arange(context_episodes + 1)
        probabilities = 1.0 / np.power(counts + 1.0, exponent)
        probabilities /= probabilities.sum()
        return int(rng.choice(counts, p=probabilities))

    def sample_batch(
        self,
        rng: np.random.Generator,
        batch_size: int,
        *,
        split: Literal["train", "validation"],
        context_episodes: int,
        num_query_states: int,
        rollout_masking: bool,
        rollout_mask_exponent: float,
        chunk_shuffle: bool,
        chunk_size: int,
        step_mask_count: int = 0,
    ) -> dict[str, np.ndarray]:
        """Sample a batch from one uniformly selected partner pair."""

        pair_candidates = self._candidate_pair_indices(split)
        pair_index = int(rng.choice(pair_candidates))
        shard = self._load_pair(pair_index)
        candidates = self._split_indices(pair_index, shard, split)
        if len(candidates) <= context_episodes:
            raise ValueError(
                f"Pair {pair_index} has {len(candidates)} {split} trajectories, "
                f"but CooT needs more than {context_episodes}"
            )

        sequence_steps = context_episodes * self.horizon
        context_states = np.zeros(
            (batch_size, sequence_steps, self.observation_dim), dtype=np.float32
        )
        context_actions = np.zeros(
            (batch_size, sequence_steps, self.action_dim), dtype=np.float32
        )
        context_rewards = np.zeros((batch_size, sequence_steps, 1), dtype=np.float32)
        query_states = np.zeros(
            (batch_size, num_query_states, self.observation_dim), dtype=np.float32
        )
        target_actions = np.zeros((batch_size, num_query_states), dtype=np.int32)
        target_mask = np.zeros((batch_size, num_query_states), dtype=np.float32)

        for batch_index in range(batch_size):
            # The released collector samples context rollout indices with
            # replacement. Keeping that behavior also allows repeated episodes
            # inside a context, as in the supplementary implementation.
            selected_context = rng.choice(
                candidates, size=context_episodes, replace=True
            )
            remaining = np.setdiff1d(candidates, selected_context, assume_unique=False)
            query_trajectory = int(rng.choice(remaining))
            masked_rollouts = (
                self._rollout_mask_count(rng, context_episodes, rollout_mask_exponent)
                if rollout_masking and split == "train"
                else 0
            )

            for context_index, trajectory_index in enumerate(selected_context):
                start = context_index * self.horizon
                stop = start + self.horizon
                if context_index < masked_rollouts:
                    continue
                states = np.asarray(
                    shard["observations"][trajectory_index], dtype=np.float32
                ).reshape(self.horizon, -1)
                actions = np.eye(self.action_dim, dtype=np.float32)[
                    shard["actions"][trajectory_index].astype(np.int32)
                ]
                rewards = np.asarray(
                    shard["rewards"][trajectory_index], dtype=np.float32
                ).reshape(self.horizon, 1)
                if chunk_shuffle and split == "train":
                    states, actions, rewards = self._shuffle_episode_chunks(
                        rng, states, actions, rewards, chunk_size
                    )
                context_states[batch_index, start:stop] = states
                context_actions[batch_index, start:stop] = actions
                context_rewards[batch_index, start:stop] = rewards

            query_length = int(shard["lengths"][query_trajectory])
            if query_length < 1:
                raise ValueError(f"Pair {pair_index} contains an empty trajectory")
            # The release left-pads short prefixes to form a six-state query.
            endpoint = int(rng.integers(query_length))
            source_start = max(0, endpoint - num_query_states + 1)
            source_stop = endpoint + 1
            destination_start = num_query_states - (source_stop - source_start)
            query_states[batch_index, destination_start:] = np.asarray(
                shard["observations"][query_trajectory, source_start:source_stop],
                dtype=np.float32,
            ).reshape(source_stop - source_start, -1)
            target_actions[batch_index, destination_start:] = shard["actions"][
                query_trajectory, source_start:source_stop
            ]
            # The supplementary one-hot targets are all-zero for left padding,
            # which contributes zero cross-entropy. Preserve that behavior with
            # an explicit loss mask instead of treating padding as action zero.
            target_mask[batch_index, destination_start:] = 1.0

        if split == "train" and step_mask_count > 0:
            count = min(int(step_mask_count), sequence_steps)
            for batch_index in range(batch_size):
                indices = rng.choice(sequence_steps, size=count, replace=False)
                # Supplementary behavior: mask action and reward, never state.
                context_actions[batch_index, indices] = 0.0
                context_rewards[batch_index, indices] = 0.0

        return {
            "context_states": context_states,
            "context_actions": context_actions,
            "context_rewards": context_rewards,
            "query_states": query_states,
            "target_actions": target_actions,
            "target_mask": target_mask,
            "pair_index": np.full((batch_size,), pair_index, dtype=np.int32),
        }

    @property
    def paper_examples_per_epoch(self) -> int:
        contexts = int(self.metadata.get("contexts_per_pair", 125))
        queries = int(self.metadata.get("queries_per_context", 70))
        return len(self.train_pair_indices) * contexts * queries
