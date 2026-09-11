"""Pickle-free human trajectories, independent of the display/input backend."""

import dataclasses
import hashlib
import json
import os
import tempfile
import uuid
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

import jax
import numpy as np

from .common import OvercookedActionsEnum

SCHEMA_VERSION = 1
ACTION_NAMES = [action.name for action in OvercookedActionsEnum]


def atomic_save(path, arrays):
    """A crash during a checkpoint leaves the previous complete archive intact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.stem}-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_episode(path):
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    metadata = json.loads(str(arrays.pop("metadata")))
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported trajectory schema: {path}")
    return arrays, metadata


def state_arrays(value, prefix="state"):
    """Flatten all array leaves, including nested Flax/Chex agent dataclasses."""
    if dataclasses.is_dataclass(value):
        value = {
            field.name: getattr(value, field.name)
            for field in dataclasses.fields(value)
        }
    if isinstance(value, Mapping):
        result = {}
        for key, child in value.items():
            result.update(state_arrays(child, f"{prefix}/{key}"))
        return result
    if value is None:
        return {}
    return {prefix: np.array(value, copy=True)}


def environment_signature(env_kwargs):
    # Includes uncommitted changes: a git commit alone cannot identify this env.
    sources = {}
    directory = Path(__file__).parent
    for path in sorted(directory.glob("*.py")):
        if path.name != "human_data.py":
            sources[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    for path in (
        directory.parent / "multi_agent_env.py",
        directory.parent / "spaces.py",
    ):
        sources[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    definition = {
        "env_kwargs": env_kwargs,
        "source_sha256": sources,
        "actions": ACTION_NAMES,
    }
    fingerprint = hashlib.sha256(
        json.dumps(definition, sort_keys=True).encode()
    ).hexdigest()
    return fingerprint, sources


class HumanEpisode:
    def __init__(self, env, env_kwargs, output, seed, mode, tick_hz, players):
        self.env = env
        self.key = jax.random.PRNGKey(seed)
        self.key, reset_key = jax.random.split(self.key)
        self.obs, self.state = env.reset(reset_key)
        self.step_fn = jax.jit(
            env.step_env
        )  # Preserve the actual terminal observation.
        self.observations = [self._obs_array(self.obs)]
        self.states = [state_arrays(self.state)]
        self.actions = []
        self.human_mask = []
        self.rewards = []
        self.shaped_rewards = []
        self.dones = []
        self.step_keys = []
        self.elapsed = []
        self.closed = False
        episode_id = uuid.uuid4().hex
        self.path = Path(output) / env_kwargs["layout"] / f"{episode_id}.npz"
        fingerprint, sources = environment_signature(env_kwargs)
        phase_order = (
            np.asarray(env.phase_orders[int(self.state.phase_order_index)]).tolist()
            if hasattr(env, "phase_orders")
            else list(range(len(env.dynamic_layout.phases)))
        )
        self.metadata = {
            "schema_version": SCHEMA_VERSION,
            "episode_id": episode_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "environment": "overcooked_v3",
            "env_kwargs": env_kwargs,
            "env_fingerprint": fingerprint,
            "source_sha256": sources,
            "jax_version": jax.__version__,
            "numpy_version": np.__version__,
            "seed": seed,
            "mode": mode,
            "tick_hz": tick_hz if mode == "realtime" else None,
            "players": players,
            "agents": list(env.agents),
            "action_names": ACTION_NAMES,
            "observation_shape": list(self.observations[0].shape[1:]),
            "observation_axes": "time, agent, height, width, channel",
            "phase_order": phase_order,
            "status": "draft",
        }

    def _obs_array(self, obs):
        return np.stack([np.asarray(obs[agent]) for agent in self.env.agents])

    @property
    def length(self):
        return len(self.actions)

    @property
    def score(self):
        # Sparse reward is shared; summing over agents would double the score.
        return sum(float(reward[0]) for reward in self.rewards)

    def step(self, actions, human_mask, elapsed):
        if self.closed:
            raise ValueError("Start a new episode before recording more actions")
        action_array = np.asarray(actions, dtype=np.int32)
        mask = np.asarray(human_mask, dtype=bool)
        if (
            action_array.shape != (self.env.num_agents,)
            or mask.shape != action_array.shape
        ):
            raise ValueError("Expected one action and one human label mask per agent")
        if np.any((action_array < 0) | (action_array >= len(ACTION_NAMES))):
            raise ValueError("Invalid action index")
        self.key, step_key = jax.random.split(self.key)
        obs, state, rewards, dones, info = self.step_fn(
            step_key,
            self.state,
            {agent: action_array[i] for i, agent in enumerate(self.env.agents)},
        )
        # Materialize the entire transition before appending to the trajectory.
        next_obs = self._obs_array(obs)
        next_state = state_arrays(state)
        reward = np.asarray([rewards[a] for a in self.env.agents], dtype=np.float32)
        shaped = np.asarray(
            [info["shaped_reward"][a] for a in self.env.agents], dtype=np.float32
        )
        done = bool(dones["__all__"])
        self.observations.append(next_obs)
        self.states.append(next_state)
        self.actions.append(action_array)
        self.human_mask.append(mask)
        self.rewards.append(reward)
        self.shaped_rewards.append(shaped)
        self.dones.append(done)
        self.step_keys.append(np.asarray(step_key))
        self.elapsed.append(float(elapsed))
        self.obs, self.state = obs, state
        if done:
            self.finish("pending")
        elif self.length % 25 == 0:
            self.save()

    def save(self):
        if not self.length:
            return None
        self.metadata.update(
            length=self.length,
            episode_return=self.score,
            complete=bool(self.dones[-1]),
            saved_at=datetime.now(timezone.utc).isoformat(),
        )
        arrays = {
            "metadata": np.asarray(json.dumps(self.metadata, sort_keys=True)),
            "observations": np.stack(self.observations),
            "actions": np.stack(self.actions),
            "human_mask": np.stack(self.human_mask),
            "rewards": np.stack(self.rewards),
            "shaped_rewards": np.stack(self.shaped_rewards),
            "dones": np.asarray(self.dones, dtype=bool),
            "step_keys": np.stack(self.step_keys),
            "elapsed_seconds": np.asarray(self.elapsed, dtype=np.float64),
        }
        for key in self.states[0]:
            arrays[key] = np.stack([state[key] for state in self.states])
        atomic_save(self.path, arrays)
        return self.path

    def finish(self, status="pending"):
        if status not in ("pending", "accepted", "rejected"):
            raise ValueError(f"Invalid review status: {status}")
        self.closed = True
        self.metadata["status"] = status
        return self.save()
