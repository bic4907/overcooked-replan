from typing import Any

import jax
import jax.numpy as jnp
import pytest

from jaxmarl import make
from jaxmarl.environments.overcooked_v3.multi_layout import (
    OBP_CANVAS_HEIGHT,
    OBP_CANVAS_WIDTH,
    OBP_LAYOUT_NAMES,
    MultiLayoutOvercookedV3,
    OBP_LAYOUT_SETS,
    pad_grid,
    padded_dynamic_layout,
)

ENV_KWARGS: dict[str, Any] = dict(
    max_steps=450,
    include_transition_countdown=True,
    include_layout_change_mask=True,
    transition_observer="both",
    random_agent_positions=False,
)
STEPS = 320  # crosses the phase boundary at 150


def _rollout(env, state, key, steps=STEPS):
    """Random-action rollout from a given state, as one scan."""

    def body(carry, _):
        state, key = carry
        key, action_key, step_key = jax.random.split(key, 3)
        actions = {
            agent: jax.random.randint(jax.random.fold_in(action_key, i), (), 0, 6)
            for i, agent in enumerate(env.agents)
        }
        obs, state, reward, _, info = env.step_env(step_key, state, actions)
        return (state, key), (
            obs["agent_0"],
            reward["agent_0"],
            info["shaped_reward"]["agent_0"],
            state.grid,
            state.agents.pos.x,
            state.agents.pos.y,
            state.agents.inventory,
            state.layout_index,
        )

    return jax.lax.scan(body, (state, key), jnp.arange(steps))[1]


def test_pad_grid_centres_and_fills_with_wall():
    padded = pad_grid("\nWWW\nWAW\nWWW\n", 5, 5).strip("\n").splitlines()
    assert padded == ["WWWWW", "WWWWW", "WWAWW", "WWWWW", "WWWWW"]


def test_pad_grid_rejects_oversized_layout():
    with pytest.raises(ValueError, match="larger than"):
        pad_grid("\nWWWWW\nWWWWW\n", 3, 3)


@pytest.mark.parametrize("name", OBP_LAYOUT_NAMES)
def test_padding_preserves_the_game(name):
    """Padding may move the kitchen but must not change how it plays."""
    raw = make("overcooked_v3", layout=name, **ENV_KWARGS)
    padded = make("overcooked_v3", layout=padded_dynamic_layout(name), **ENV_KWARGS)
    key = jax.random.PRNGKey(3)

    _, raw_state = raw.reset(key)
    _, padded_state = padded.reset(key)
    dx = padded_state.agents.pos.x[0] - raw_state.agents.pos.x[0]
    dy = padded_state.agents.pos.y[0] - raw_state.agents.pos.y[0]

    _, r_reward, r_shaped, _, r_x, r_y, r_inv, r_phase = _rollout(raw, raw_state, key)
    _, p_reward, p_shaped, _, p_x, p_y, p_inv, p_phase = _rollout(
        padded, padded_state, key
    )

    assert jnp.array_equal(r_reward, p_reward)
    assert jnp.array_equal(r_shaped, p_shaped)
    assert jnp.array_equal(r_inv, p_inv)
    assert jnp.array_equal(r_phase, p_phase)
    assert jnp.array_equal(r_x + dx, p_x)
    assert jnp.array_equal(r_y + dy, p_y)


@pytest.mark.parametrize("name", OBP_LAYOUT_NAMES)
def test_every_layout_shares_the_canvas(name):
    env = make("overcooked_v3", layout=padded_dynamic_layout(name), **ENV_KWARGS)
    assert env.observation_space("agent_0").shape == (
        OBP_CANVAS_HEIGHT,
        OBP_CANVAS_WIDTH,
        31,
    )


def test_reset_installs_the_drawn_layout():
    env = MultiLayoutOvercookedV3(**ENV_KWARGS)
    drawn = set()
    for seed in range(60):
        obs, state = env.reset(jax.random.PRNGKey(seed))
        layout_id = int(env.layout_id(state))
        drawn.add(layout_id)
        base = layout_id * env.phases_per_layout

        assert obs["agent_0"].shape == (OBP_CANVAS_HEIGHT, OBP_CANVAS_WIDTH, 31)
        assert int(state.layout_base) == base
        assert int(state.layout_index) == base
        assert jnp.array_equal(state.grid[:, :, 0], env.phase_static_objects[base])
        assert int(state.recipe) == int(env.layout_recipes[layout_id])
        spawns = env.phase_agent_positions[base]
        assert jnp.array_equal(state.agents.pos.x, spawns[:, 0])
        assert jnp.array_equal(state.agents.pos.y, spawns[:, 1])
    assert len(drawn) == env.num_layouts, "every layout should be drawn eventually"


def test_episode_matches_the_dedicated_single_layout_env():
    """A drawn layout must play exactly as that layout's own environment does."""
    env = MultiLayoutOvercookedV3(**ENV_KWARGS)
    for seed in range(8):
        key = jax.random.PRNGKey(seed)
        _, state = env.reset(key)
        layout_id = int(env.layout_id(state))
        name = OBP_LAYOUT_NAMES[layout_id]
        single = make("overcooked_v3", layout=padded_dynamic_layout(name), **ENV_KWARGS)

        multi_trace = _rollout(env, state, key)
        single_trace = _rollout(
            single,
            state.replace(
                layout_index=jnp.array(0, dtype=jnp.int32),
                layout_base=jnp.array(0, dtype=jnp.int32),
            ),
            key,
        )
        for multi, single_value in zip(multi_trace[:-1], single_trace[:-1]):
            assert jnp.array_equal(multi, single_value), name
        # The multi-layout phase index is offset by the layout's own block.
        assert jnp.array_equal(
            multi_trace[-1] - layout_id * env.phases_per_layout, single_trace[-1]
        ), name


def test_recipe_follows_the_drawn_layout():
    """outage kitchens cook two onions, the others three."""
    env = MultiLayoutOvercookedV3(**ENV_KWARGS)
    for layout_id, name in enumerate(OBP_LAYOUT_NAMES):
        expected = 8 if name.startswith("outage") else 12
        assert int(env.layout_recipes[layout_id]) == expected, name


def test_rejects_duplicate_layouts():
    with pytest.raises(ValueError, match="duplicates"):
        MultiLayoutOvercookedV3(layout=("split_0", "split_0"), **ENV_KWARGS)


def test_rejects_unknown_layout():
    with pytest.raises(ValueError, match="Unknown dynamic layout"):
        MultiLayoutOvercookedV3(layout=("no_such_layout",), **ENV_KWARGS)


def test_single_layout_environment_is_unaffected_by_the_base_offset():
    """layout_base defaults to zero, so one-layout environments are unchanged."""
    env = make("overcooked_v3", layout="split_0", **ENV_KWARGS)
    _, state = env.reset(jax.random.PRNGKey(0))
    assert int(state.layout_base) == 0
    for step in (0, 149, 150, 299, 300, 449):
        step_array = jnp.array(step, dtype=jnp.int32)
        assert int(env._phase_index(step_array, 0)) == int(
            env.get_layout_index(step_array)
        )


def test_layout_set_name_resolves_to_its_layouts():
    """A set name has to work wherever a single layout name used to go."""
    env = MultiLayoutOvercookedV3(layout="obp10", **ENV_KWARGS)
    assert env.layout_names == OBP_LAYOUT_SETS["obp10"]
    assert env.layout_names == OBP_LAYOUT_NAMES


def test_rejects_unknown_layout_set():
    with pytest.raises(ValueError, match="Unknown layout set"):
        MultiLayoutOvercookedV3(layout="no_such_set", **ENV_KWARGS)


def test_registry_builds_the_layout_distribution():
    env = make("overcooked_v3_multilayout", layout="obp10", **ENV_KWARGS)
    assert isinstance(env, MultiLayoutOvercookedV3)
    assert env.observation_space("agent_0").shape == (
        OBP_CANVAS_HEIGHT,
        OBP_CANVAS_WIDTH,
        31,
    )
