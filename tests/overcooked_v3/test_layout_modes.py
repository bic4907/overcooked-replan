"""Coverage for the V3 ``layout_mode`` / ``reset_on_layout_change`` conditions.

These are the three switching conditions of the map-switch experiment:
``single_map`` uses a static ``*_policy_N`` layout, ``multimap`` uses
``episode_random``, and ``switchmap`` uses ``cyclic`` with a per-boundary reset.
"""

from collections import Counter

import jax
import jax.numpy as jnp
import pytest

from jaxmarl.environments.overcooked_v3 import dynamic_layouts
from jaxmarl.environments.overcooked_v3.common import DynamicObject, StaticObject
from jaxmarl.environments.overcooked_v3.dynamic_overcooked import OvercookedV3

SCENARIOS = (
    "split_0",
    "split_1",
    "outage_0",
    "outage_1",
    "recipe_switch_0",
    "recipe_switch_1",
    "distance_switch_0",
    "distance_switch_1",
)


def _env(layout, **kwargs):
    return OvercookedV3(
        layout=layout,
        max_steps=450,
        include_transition_countdown=False,
        include_layout_change_mask=False,
        **kwargs,
    )


def test_layout_mode_rejects_unknown_values():
    with pytest.raises(ValueError, match="layout_mode"):
        _env("split_0", layout_mode="random")
    with pytest.raises(ValueError, match="reset_on_layout_change"):
        _env("split_0", reset_on_layout_change="yes")


@pytest.mark.parametrize("layout_name", SCENARIOS)
def test_episode_random_draws_each_distinct_kitchen_evenly(layout_name):
    """A -> B -> A cycles repeat a phase; sampling must not favour A."""
    env = _env(layout_name, layout_mode="episode_random")
    _, states = jax.vmap(env.reset)(jax.random.split(jax.random.PRNGKey(0), 800))
    counts = Counter(int(index) for index in states.layout_index)

    assert set(counts) == {0, 1}
    assert min(counts.values()) / sum(counts.values()) > 0.4


@pytest.mark.parametrize("layout_name", SCENARIOS)
def test_episode_random_holds_one_kitchen_for_the_whole_episode(layout_name):
    env = _env(layout_name, layout_mode="episode_random")
    keys = jax.random.split(jax.random.PRNGKey(1), 16)
    _, states = jax.vmap(env.reset)(keys)
    sampled = states.layout_index

    actions = {agent: jnp.zeros((16,), dtype=jnp.int32) for agent in env.agents}
    step = jax.jit(jax.vmap(env.step_env))
    for offset in range(220):
        _, states, _, _, info = step(
            jax.random.split(jax.random.PRNGKey(offset), 16), states, actions
        )
        assert not jnp.any(info["layout_changed"])
    assert jnp.array_equal(states.layout_index, sampled)


def test_episode_random_uses_the_sampled_phase_recipe():
    """The reset recipe must follow the installed phase, not always phase 0."""
    env = _env("recipe_switch_0", layout_mode="episode_random")
    phase_recipes = env.phase_recipes[env.unique_phase_indices]
    assert len(set(int(recipe) for recipe in phase_recipes)) == 2

    _, states = jax.vmap(env.reset)(jax.random.split(jax.random.PRNGKey(2), 64))
    expected = env.phase_recipes[states.layout_index]
    assert jnp.array_equal(states.recipe, expected)
    assert jnp.array_equal(states.previous_recipe, expected)


@pytest.mark.parametrize("layout_name", SCENARIOS)
def test_reset_on_layout_change_starts_a_fresh_kitchen(layout_name):
    env = _env(layout_name, reset_on_layout_change=True)
    boundary = int(env.phase_ends[0])
    _, state = env.reset(jax.random.PRNGKey(0))

    # Dirty every transient slot so a surviving one would be visible.
    grid = state.grid.at[:, :, 1].set(DynamicObject.PLATE)
    state = state.replace(
        step=jnp.array(boundary - 1),
        grid=grid,
        agents=state.agents.replace(
            inventory=jnp.full((env.num_agents,), DynamicObject.PLATE)
        ),
    )

    actions = {agent: jnp.array(4, dtype=jnp.int32) for agent in env.agents}
    _, state, _, _, info = jax.jit(env.step_env)(
        jax.random.PRNGKey(1), state, actions
    )

    assert jnp.all(info["layout_changed"])
    assert int(state.layout_index) == 1
    assert jnp.all(state.agents.inventory == DynamicObject.EMPTY)
    assert jnp.all(state.grid[:, :, 1] == DynamicObject.EMPTY)
    assert jnp.array_equal(state.grid[:, :, 0], env.phase_static_objects[1])

    spawns = env.phase_agent_positions[1]
    assert jnp.array_equal(state.agents.pos.x, spawns[:, 0])
    assert jnp.array_equal(state.agents.pos.y, spawns[:, 1])
    # Only the episode clock survives the transition.
    assert int(state.step) == boundary


@pytest.mark.parametrize("layout_name", SCENARIOS)
def test_cyclic_without_reset_keeps_untouched_cells(layout_name):
    """The default transition edits only the cells the two phases disagree on."""
    env = _env(layout_name, reset_on_layout_change=False)
    boundary = int(env.phase_ends[0])
    _, state = env.reset(jax.random.PRNGKey(0))

    empty_in_both = (env.phase_static_objects[0] == StaticObject.EMPTY) & (
        env.phase_static_objects[1] == StaticObject.EMPTY
    )
    grid = state.grid.at[:, :, 1].set(
        jnp.where(empty_in_both, DynamicObject.PLATE, state.grid[:, :, 1])
    )
    state = state.replace(step=jnp.array(boundary - 1), grid=grid)

    actions = {agent: jnp.array(4, dtype=jnp.int32) for agent in env.agents}
    _, state, _, _, info = jax.jit(env.step_env)(
        jax.random.PRNGKey(1), state, actions
    )

    assert jnp.all(info["layout_changed"])
    assert jnp.all(state.grid[:, :, 1][empty_in_both] == DynamicObject.PLATE)


@pytest.mark.parametrize("layout_name", SCENARIOS)
def test_static_phase_policy_layouts_never_switch(layout_name):
    """``single_map`` runs one specialist kitchen for the whole episode."""
    for policy_index in (0, 1):
        specialist = f"{layout_name}_policy_{policy_index}"
        assert len(dynamic_layouts[specialist].phases) == 1
        env = _env(specialist)
        _, state = env.reset(jax.random.PRNGKey(0))

        actions = {agent: jnp.array(4, dtype=jnp.int32) for agent in env.agents}
        step = jax.jit(env.step_env)
        for offset in (0, 150, 300):
            _, _, _, _, info = step(
                jax.random.PRNGKey(offset),
                state.replace(step=jnp.array(offset)),
                actions,
            )
            assert not jnp.any(info["layout_changed"])
            assert jnp.all(info["layout_index"] == 0)


@pytest.mark.parametrize("layout_name", SCENARIOS)
def test_phase_steps_forces_one_switching_cadence(layout_name):
    """Every scenario switches on the same clock once ``phase_steps`` is set."""
    default_env = _env(layout_name)
    forced = _env(layout_name, phase_steps=150)

    assert set(int(step) for step in forced.phase_durations) == {150}
    assert forced.cycle_steps == 150 * len(forced.phase_durations)
    # The authored schedule is only replaced when the override is passed.
    assert forced.phase_durations.shape == default_env.phase_durations.shape

    boundaries = []
    actions = {agent: jnp.array(4, dtype=jnp.int32) for agent in forced.agents}
    step = jax.jit(forced.step_env)
    _, state = forced.reset(jax.random.PRNGKey(0))
    for _ in range(450):
        _, state, _, done, info = step(jax.random.PRNGKey(0), state, actions)
        if bool(jnp.all(info["layout_changed"])):
            boundaries.append(int(state.step))
        if bool(done["__all__"]):
            break
    assert boundaries[:2] == [150, 300]


@pytest.mark.parametrize("layout_name", SCENARIOS)
def test_phase_steps_leaves_specialists_static(layout_name):
    """A single-phase layout still never switches under the shared cadence."""
    for policy_index in (0, 1):
        env = _env(f"{layout_name}_policy_{policy_index}", phase_steps=150)
        actions = {agent: jnp.array(4, dtype=jnp.int32) for agent in env.agents}
        step = jax.jit(env.step_env)
        _, state = env.reset(jax.random.PRNGKey(0))
        for _ in range(320):
            _, state, _, _, info = step(jax.random.PRNGKey(0), state, actions)
            assert not jnp.any(info["layout_changed"])


def test_phase_steps_rejects_non_positive_values():
    with pytest.raises(ValueError, match="phase_steps"):
        _env("split_0", phase_steps=0)
    with pytest.raises(ValueError, match="phase_steps"):
        _env("split_0", phase_steps=True)
