"""Behavior cloning on top of an optimal behavior prior.

The method's whole claim rests on one choice: start the human model from a
self-play policy trained across the layout distribution instead of from random
weights. Training then spends its data on where a person differs from a
competent player, not on learning the game.

So the human model is the *same network* as the prior -- ActorCriticCNN, no
recurrence, exactly as the paper describes -- and its parameters are loaded
straight from the prior's checkpoint. Nothing is reshaped or renamed, which is
what makes the transfer trustworthy rather than approximate.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, NamedTuple, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.traverse_util import flatten_dict, unflatten_dict

from baselines.IPPO.ippo_overcooked_v3 import ActorCriticCNN
from jaxmarl.wrappers.baselines import load_params, save_params

#: Config the policy network needs. Must match the prior's, or its weights do
#: not describe the same network.
DEFAULT_POLICY_CONFIG = {
    "ACTIVATION": "relu",
    "FC_DIM_SIZE": 128,
    "GRU_HIDDEN_DIM": 128,
}

NUM_ACTIONS = 6


class TrainingReport(NamedTuple):
    params: dict
    history: list[dict]
    frozen: tuple[str, ...]


def build_policy(num_actions: int = NUM_ACTIONS, config: dict | None = None):
    """The prior's network, and a function from parameters to action logits."""
    config = {**DEFAULT_POLICY_CONFIG, **(config or {})}
    network = ActorCriticCNN(num_actions, config=config)

    def logits(params, observations):
        # The policy expects a leading time axis and a done flag per row; a
        # batch of independent states is one step of many parallel actors.
        batch = observations.shape[0]
        inputs = (observations[None, :], jnp.zeros((1, batch), dtype=jnp.bool_))
        _, pi, _ = network.apply(params, None, inputs)
        return pi.logits[0]

    return network, logits


def initial_params(
    network: ActorCriticCNN,
    observation_shape: Sequence[int],
    key: jax.Array,
    prior_checkpoint: str | Path | None = None,
) -> dict:
    """Randomly initialise, then overwrite with the prior when one is given."""
    batch = jnp.zeros((1, 1, *observation_shape), dtype=jnp.float32)
    params = network.init(key, None, (batch, jnp.zeros((1, 1), dtype=jnp.bool_)))
    if prior_checkpoint is None:
        return params

    prior = load_params(str(prior_checkpoint))
    ours = flatten_dict(params, sep="/")
    theirs = flatten_dict(prior, sep="/")
    missing = sorted(set(ours) - set(theirs))
    extra = sorted(set(theirs) - set(ours))
    if missing or extra:
        raise ValueError(
            "The prior checkpoint does not describe this network. Missing "
            f"{missing[:4]}, unexpected {extra[:4]}. The prior has to be trained "
            "with the same architecture, FC_DIM_SIZE and observation shape."
        )
    mismatched = [key for key in ours if jnp.shape(ours[key]) != jnp.shape(theirs[key])]
    if mismatched:
        raise ValueError(
            f"The prior's parameters have different shapes at {mismatched[:4]}; "
            "the observation canvas or layer widths must differ."
        )
    return unflatten_dict({key: jnp.asarray(theirs[key]) for key in ours}, sep="/")


def freeze_mask(params: dict, prefixes: Sequence[str]) -> dict:
    """Per-parameter multipliers that hold the named subtrees fixed.

    Freezing early layers is the paper's extra regulariser: the prior already
    knows how to see a kitchen, so leaving that part alone spends the human
    data on the part that differs.
    """
    flat = flatten_dict(params, sep="/")
    mask = {
        key: jnp.float32(0.0 if any(key.startswith(p) for p in prefixes) else 1.0)
        for key in flat
    }
    return unflatten_dict(mask, sep="/")


def frozen_names(params: dict, prefixes: Sequence[str]) -> tuple[str, ...]:
    if not prefixes:
        return ()
    flat = flatten_dict(params, sep="/")
    return tuple(k for k in sorted(flat) if any(k.startswith(p) for p in prefixes))


def cross_entropy(logits_fn: Callable, params, observations, actions):
    logits = logits_fn(params, observations)
    labels = jax.nn.one_hot(actions, logits.shape[-1])
    return -jnp.mean(jnp.sum(labels * jax.nn.log_softmax(logits), axis=-1))


def accuracy(logits_fn: Callable, params, observations, actions):
    return jnp.mean(jnp.argmax(logits_fn(params, observations), axis=-1) == actions)


def train(
    observations: jax.Array,
    actions: jax.Array,
    validation_observations: jax.Array | None = None,
    validation_actions: jax.Array | None = None,
    prior_checkpoint: str | Path | None = None,
    freeze_prefixes: Sequence[str] = (),
    epochs: int = 20,
    batch_size: int = 256,
    learning_rate: float = 3e-4,
    max_grad_norm: float = 0.5,
    seed: int = 0,
    policy_config: dict | None = None,
    log: Callable[[dict], None] | None = None,
) -> TrainingReport:
    """Fit action logits to human actions, starting from the prior."""
    if observations.shape[0] != actions.shape[0]:
        raise ValueError("observations and actions must have the same length")
    if observations.shape[0] < batch_size:
        raise ValueError(
            f"{observations.shape[0]} rows is fewer than one batch of {batch_size}"
        )

    key = jax.random.PRNGKey(seed)
    key, init_key = jax.random.split(key)
    network, logits_fn = build_policy(NUM_ACTIONS, policy_config)
    params = initial_params(network, observations.shape[1:], init_key, prior_checkpoint)
    frozen = frozen_names(params, freeze_prefixes)
    mask = freeze_mask(params, freeze_prefixes)

    optimiser = optax.chain(
        optax.clip_by_global_norm(max_grad_norm), optax.adam(learning_rate)
    )
    optimiser_state = optimiser.init(params)

    @jax.jit
    def step(params, optimiser_state, batch_observations, batch_actions):
        loss, grads = jax.value_and_grad(cross_entropy, argnums=1)(
            logits_fn, params, batch_observations, batch_actions
        )
        # Freezing is applied to the gradient rather than the update so that
        # neither momentum nor clipping can move a held parameter.
        grads = jax.tree.map(lambda g, m: g * m, grads, mask)
        updates, optimiser_state = optimiser.update(grads, optimiser_state, params)
        return optax.apply_updates(params, updates), optimiser_state, loss

    @jax.jit
    def evaluate_batch(params, batch_observations, batch_actions):
        return (
            cross_entropy(logits_fn, params, batch_observations, batch_actions),
            accuracy(logits_fn, params, batch_observations, batch_actions),
        )

    def evaluate(params, all_observations, all_actions, chunk=4096):
        """Score the whole set in chunks.

        One convolution over tens of thousands of rows asks for more device
        memory than the activations are worth, and the answer is a mean either
        way -- so it is taken chunk by chunk, weighted by chunk size.
        """
        rows = all_observations.shape[0]
        loss_total = accuracy_total = 0.0
        for start in range(0, rows, chunk):
            stop = min(start + chunk, rows)
            loss, correct = evaluate_batch(
                params, all_observations[start:stop], all_actions[start:stop]
            )
            weight = stop - start
            loss_total += float(loss) * weight
            accuracy_total += float(correct) * weight
        return loss_total / rows, accuracy_total / rows

    rows = observations.shape[0]
    batches = rows // batch_size
    history = []
    for epoch in range(epochs):
        key, shuffle_key = jax.random.split(key)
        order = jax.random.permutation(shuffle_key, rows)
        total = 0.0
        for index in range(batches):
            rows_in_batch = order[index * batch_size : (index + 1) * batch_size]
            params, optimiser_state, loss = step(
                params,
                optimiser_state,
                observations[rows_in_batch],
                actions[rows_in_batch],
            )
            total += float(loss)
        record = {"epoch": epoch + 1, "batch_loss": total / max(batches, 1)}
        train_loss, train_accuracy = evaluate(params, observations, actions)
        record["train_loss"] = train_loss
        record["train_accuracy"] = train_accuracy
        if validation_observations is not None and validation_actions is not None:
            validation_loss, validation_accuracy = evaluate(
                params, validation_observations, validation_actions
            )
            record["validation_loss"] = validation_loss
            record["validation_accuracy"] = validation_accuracy
        history.append(record)
        if log is not None:
            log(record)

    return TrainingReport(params=params, history=history, frozen=frozen)


def save(
    directory: str | Path,
    params: dict,
    observation_shape: Sequence[int],
    metadata: dict | None = None,
) -> Path:
    """Write the model next to what is needed to rebuild the network."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    save_params(params, str(directory / "policy.safetensors"))
    (directory / "config.json").write_text(
        json.dumps(
            {
                "network": "ActorCriticCNN",
                "policy_config": DEFAULT_POLICY_CONFIG,
                "observation_shape": list(observation_shape),
                "num_actions": NUM_ACTIONS,
                **(metadata or {}),
            },
            indent=2,
        )
    )
    return directory


def load(directory: str | Path):
    """Return (logits_fn, params, config) for a saved human model."""
    directory = Path(directory)
    config = json.loads((directory / "config.json").read_text())
    _, logits_fn = build_policy(config["num_actions"], config["policy_config"])
    params = load_params(str(directory / "policy.safetensors"))
    params = jax.tree.map(jnp.asarray, params)
    return logits_fn, params, config


def action_distribution(logits_fn: Callable, params, observations) -> np.ndarray:
    """Share of rows the model assigns to each action, for a quick sanity read."""
    predicted = np.asarray(jnp.argmax(logits_fn(params, observations), axis=-1))
    return np.bincount(predicted, minlength=NUM_ACTIONS) / predicted.size
