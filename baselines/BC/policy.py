"""Shared feed-forward BC policy and portable inference loader."""

import json
from pathlib import Path

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization


class BCPolicy(nn.Module):
    hidden_size: int = 256
    num_actions: int = 6

    @nn.compact
    def __call__(self, observations):
        x = observations.reshape((observations.shape[0], -1))
        x = nn.relu(nn.Dense(self.hidden_size)(x))
        x = nn.relu(nn.Dense(self.hidden_size)(x))
        return nn.Dense(self.num_actions)(x)


def load_policy(directory):
    """Return (logits_fn, config). Input: batch of raw HWC observations.

    Use argmax for deterministic actions, or jax.random.categorical(key, logits)
    for stochastic actions. Observation channels must match config/env_kwargs.
    """
    directory = Path(directory)
    config = json.loads((directory / "config.json").read_text())
    model = BCPolicy(config["hidden_size"], len(config["action_names"]))
    shape = tuple(config["observation_shape"])
    template = model.init(jax.random.PRNGKey(0), jnp.zeros((1, *shape)))["params"]
    params = serialization.from_bytes(
        template, (directory / "policy.msgpack").read_bytes()
    )
    with np.load(directory / "normalization.npz", allow_pickle=False) as archive:
        mean, scale = jnp.asarray(archive["mean"]), jnp.asarray(archive["scale"])

    @jax.jit
    def logits(observations):
        return model.apply({"params": params}, (observations - mean) / scale)

    return logits, config
