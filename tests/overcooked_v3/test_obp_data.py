from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxmarl
from baselines.OBP.data import (
    build_dataset,
    canvas_offset,
    find_episodes,
    load_episode,
    pad_state,
    split_by_episode,
)
from jaxmarl.environments.overcooked_v3.multi_layout import (
    OBP_CANVAS_HEIGHT,
    OBP_CANVAS_WIDTH,
    OBP_LAYOUT_NAMES,
    padded_dynamic_layout,
)

DEMONSTRATIONS = Path("data/human")
needs_demonstrations = pytest.mark.skipif(
    not DEMONSTRATIONS.is_dir(), reason="no human demonstrations checked out"
)


@pytest.mark.parametrize("name", OBP_LAYOUT_NAMES)
def test_canvas_offset_matches_where_the_padded_layout_puts_the_agents(name):
    """The offset used for states has to be the one used for grids."""
    raw = jaxmarl.make("overcooked_v3", layout=name)
    padded = jaxmarl.make("overcooked_v3", layout=padded_dynamic_layout(name))
    _, raw_state = raw.reset(jax.random.PRNGKey(0))
    _, padded_state = padded.reset(jax.random.PRNGKey(0))

    top, left = canvas_offset(name)
    assert int(padded_state.agents.pos.x[0] - raw_state.agents.pos.x[0]) == left
    assert int(padded_state.agents.pos.y[0] - raw_state.agents.pos.y[0]) == top


def test_canvas_offset_rejects_an_oversized_layout():
    with pytest.raises(ValueError, match="larger than"):
        canvas_offset("split_0", width=5, height=5)


@needs_demonstrations
def test_every_demonstration_rebuilds_and_pads():
    """Loading verifies against the stored observations and re-renders padded."""
    paths = find_episodes(DEMONSTRATIONS)
    assert paths, "expected at least one demonstration"
    for path in paths:
        episode = load_episode(path)
        steps, agents = episode.actions.shape
        assert episode.observations.shape == (
            steps,
            agents,
            OBP_CANVAS_HEIGHT,
            OBP_CANVAS_WIDTH,
            31,
        )
        assert episode.human_mask.shape[0] >= steps


@needs_demonstrations
def test_padded_observations_contain_the_original_ones():
    """Outside the kitchen is wall; inside it must be untouched."""
    path = find_episodes(DEMONSTRATIONS)[0]
    episode = load_episode(path)
    top, left = canvas_offset(episode.layout)

    with np.load(path, allow_pickle=False) as archive:
        stored = jnp.asarray(archive["observations"])
    height, width = stored.shape[2:4]
    window = episode.observations[:, :, top : top + height, left : left + width, :]
    assert jnp.allclose(window, stored[: episode.actions.shape[0]], atol=1e-6)


@needs_demonstrations
def test_dataset_keeps_one_row_per_human_action():
    episodes = [load_episode(path) for path in find_episodes(DEMONSTRATIONS)[:4]]
    dataset = build_dataset(episodes)
    expected = sum(
        int(jnp.sum(episode.human_mask[: episode.actions.shape[0]]))
        for episode in episodes
    )
    assert dataset.observations.shape[0] == expected
    assert dataset.actions.shape == (expected,)
    assert dataset.episode_index.shape == (expected,)
    assert int(jnp.max(dataset.actions)) <= 5
    assert int(jnp.min(dataset.actions)) >= 0


@needs_demonstrations
def test_split_never_puts_one_episode_on_both_sides():
    episodes = [load_episode(path) for path in find_episodes(DEMONSTRATIONS)[:8]]
    dataset = build_dataset(episodes)
    train, validation = split_by_episode(dataset, 0.25, seed=0)

    assert not bool(jnp.any(train & validation))
    assert bool(jnp.all(train | validation))
    index = np.asarray(dataset.episode_index)
    assert not set(index[np.asarray(train)]) & set(index[np.asarray(validation)])


@needs_demonstrations
def test_split_can_hold_nothing_out():
    episodes = [load_episode(path) for path in find_episodes(DEMONSTRATIONS)[:2]]
    dataset = build_dataset(episodes)
    train, validation = split_by_episode(dataset, 0.0)
    assert bool(jnp.all(train))
    assert not bool(jnp.any(validation))


@needs_demonstrations
def test_pad_state_rejects_a_grid_that_does_not_fit():
    path = find_episodes(DEMONSTRATIONS)[0]
    from baselines.OBP.data import _read_archive, _state_from_arrays

    _, arrays = _read_archive(path)
    state = _state_from_arrays(arrays)
    with pytest.raises(ValueError, match="does not fit"):
        pad_state(state, top=0, left=0, height=3, width=3)


def test_find_episodes_can_filter_by_layout():
    if not DEMONSTRATIONS.is_dir():
        pytest.skip("no human demonstrations checked out")
    everything = find_episodes(DEMONSTRATIONS)
    filtered = find_episodes(DEMONSTRATIONS, layouts=["split_0"])
    assert filtered
    assert set(filtered) <= set(everything)
    assert all(path.parent.name == "split_0" for path in filtered)
    assert find_episodes(DEMONSTRATIONS, layouts=["no_such_layout"]) == []


def test_find_episodes_reports_a_missing_directory():
    with pytest.raises(FileNotFoundError):
        find_episodes("data/definitely-not-here")
