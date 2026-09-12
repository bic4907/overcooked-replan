import json
import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from baselines.OBP import bc
from jaxmarl.wrappers.baselines import save_params

OBSERVATION_SHAPE = (7, 13, 31)
ROWS = 256


@pytest.fixture(scope="module")
def toy_dataset():
    """Rows whose action is decided by one channel, so fitting is learnable."""
    key = jax.random.PRNGKey(0)
    observations = jax.random.normal(key, (ROWS, *OBSERVATION_SHAPE))
    actions = (jnp.sum(observations[..., 0], axis=(1, 2)) > 0).astype(jnp.int32)
    return observations, actions


def test_logits_have_one_row_per_observation(toy_dataset):
    observations, _ = toy_dataset
    network, logits_fn = bc.build_policy()
    params = bc.initial_params(network, OBSERVATION_SHAPE, jax.random.PRNGKey(0))
    logits = logits_fn(params, observations[:8])
    assert logits.shape == (8, bc.NUM_ACTIONS)


def test_training_reduces_the_loss(toy_dataset):
    observations, actions = toy_dataset
    report = bc.train(observations, actions, epochs=6, batch_size=64, seed=0)
    assert len(report.history) == 6
    assert report.history[-1]["train_loss"] < report.history[0]["train_loss"]
    assert report.frozen == ()


def test_prior_checkpoint_is_loaded_verbatim(toy_dataset):
    """The point of the method is that the prior's weights transfer unchanged."""
    observations, _ = toy_dataset
    network, logits_fn = bc.build_policy()
    prior = bc.initial_params(network, OBSERVATION_SHAPE, jax.random.PRNGKey(7))

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "prior.safetensors"
        save_params(prior, str(path))
        loaded = bc.initial_params(
            network, OBSERVATION_SHAPE, jax.random.PRNGKey(0), prior_checkpoint=path
        )

    for ours, theirs in zip(
        jax.tree.leaves(loaded), jax.tree.leaves(jax.tree.map(jnp.asarray, prior))
    ):
        assert jnp.array_equal(ours, theirs)
    assert jnp.allclose(
        logits_fn(loaded, observations[:4]), logits_fn(prior, observations[:4])
    )


def test_a_prior_for_another_network_is_rejected():
    network, _ = bc.build_policy()
    prior = bc.initial_params(network, (5, 7, 31), jax.random.PRNGKey(0))
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "prior.safetensors"
        save_params(prior, str(path))
        with pytest.raises(ValueError, match="different shapes"):
            bc.initial_params(
                network,
                OBSERVATION_SHAPE,
                jax.random.PRNGKey(0),
                prior_checkpoint=path,
            )


def test_frozen_parameters_do_not_move(toy_dataset):
    observations, actions = toy_dataset
    report = bc.train(
        observations,
        actions,
        epochs=3,
        batch_size=64,
        seed=0,
        freeze_prefixes=("params/CNN_0",),
    )
    network, _ = bc.build_policy()
    # train() splits its seed before initialising, so the reference has to be
    # built from the same derived key or it is a different network entirely.
    _, init_key = jax.random.split(jax.random.PRNGKey(0))
    start = bc.initial_params(network, OBSERVATION_SHAPE, init_key)

    from flax.traverse_util import flatten_dict

    before = flatten_dict(start, sep="/")
    after = flatten_dict(report.params, sep="/")
    assert report.frozen, "expected the CNN trunk to be reported as frozen"
    for name in report.frozen:
        assert jnp.array_equal(before[name], after[name]), name
    moved = [
        name
        for name in after
        if name not in report.frozen and not jnp.array_equal(before[name], after[name])
    ]
    assert moved, "the unfrozen parameters should still train"


def test_save_and_load_round_trip(toy_dataset):
    observations, actions = toy_dataset
    report = bc.train(observations, actions, epochs=1, batch_size=64, seed=0)
    _, logits_fn = bc.build_policy()

    with tempfile.TemporaryDirectory() as directory:
        bc.save(directory, report.params, OBSERVATION_SHAPE, metadata={"note": "test"})
        config = json.loads((Path(directory) / "config.json").read_text())
        loaded_fn, loaded_params, loaded_config = bc.load(directory)

    assert config["observation_shape"] == list(OBSERVATION_SHAPE)
    assert loaded_config["note"] == "test"
    assert jnp.allclose(
        loaded_fn(loaded_params, observations[:16]),
        logits_fn(report.params, observations[:16]),
    )


def test_training_rejects_mismatched_lengths(toy_dataset):
    observations, actions = toy_dataset
    with pytest.raises(ValueError, match="same length"):
        bc.train(observations, actions[:-1], batch_size=64)


def test_training_rejects_a_dataset_smaller_than_one_batch(toy_dataset):
    observations, actions = toy_dataset
    with pytest.raises(ValueError, match="fewer than one batch"):
        bc.train(observations[:10], actions[:10], batch_size=64)


def test_action_distribution_sums_to_one(toy_dataset):
    observations, actions = toy_dataset
    report = bc.train(observations, actions, epochs=1, batch_size=64, seed=0)
    _, logits_fn = bc.build_policy()
    shares = bc.action_distribution(logits_fn, report.params, observations)
    assert shares.shape == (bc.NUM_ACTIONS,)
    assert np.isclose(shares.sum(), 1.0)
