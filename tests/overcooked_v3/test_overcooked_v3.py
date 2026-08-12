import jax
import jax.numpy as jnp
import pytest

from jaxmarl import make
from jaxmarl.environments.overcooked_v3.common import (
    Direction,
    DynamicObject,
    Position,
    StaticObject,
)
from jaxmarl.environments.overcooked_v3.dynamic_layouts import (
    DynamicLayout,
    dynamic_layouts,
)
from jaxmarl.environments.overcooked_v3.dynamic_overcooked import OvercookedV3
from jaxmarl.environments.overcooked_v3.overcooked import (
    ObservationType,
    OvercookedV3Base,
)

BASE = """
WWWWW
W A W
W A W
WOPXW
WWBWW
"""


def _env(second, first_steps=1, second_steps=1, **kwargs):
    layout = DynamicLayout.from_data([[BASE, first_steps], [second, second_steps]])
    return OvercookedV3(layout=layout, max_steps=20, **kwargs)


def _actions(env, first=4, second=4):
    return {
        env.agents[0]: jnp.array(first, dtype=jnp.uint32),
        env.agents[1]: jnp.array(second, dtype=jnp.uint32),
    }


def _step(env, state, actions=None, seed=1):
    if actions is None:
        actions = _actions(env)
    return jax.jit(env.step_env)(jax.random.PRNGKey(seed), state, actions)


def _positions(state):
    return list(
        zip(
            map(int, state.agents.pos.x),
            map(int, state.agents.pos.y),
        )
    )


def test_overcooked_v3_is_registered_and_based_on_v2():
    env = make("overcooked_v3", layout="dynamic_00", max_steps=20)
    assert isinstance(env, OvercookedV3)
    assert isinstance(env, OvercookedV3Base)


def test_default_observation_adds_countdown_and_layout_change_mask():
    env = OvercookedV3(layout="dynamic_00", max_steps=20)
    obs, state = env.reset(jax.random.PRNGKey(0))

    assert state.grid.shape == (env.height, env.width, 3)
    assert obs["agent_0"].shape == (env.height, env.width, 32)
    assert state.layout_index.item() == 0
    assert state.steps_until_layout_change.item() == 100
    assert jnp.all(obs["agent_0"][..., -2] == 0.0)
    assert jnp.all(obs["agent_0"][..., -1] == 0.0)
    assert jnp.any(state.layout_change_mask)

    warning_state = state.replace(step=jnp.array(80))
    warning_obs = env.get_obs(warning_state)
    assert jnp.all(warning_obs["agent_0"][..., -2] == 1.0)
    assert jnp.array_equal(
        warning_obs["agent_0"][..., -1].astype(jnp.bool_),
        state.layout_change_mask,
    )


def test_transition_countdown_decreases_and_resets_after_layout_change():
    env = _env(
        BASE.replace("W A W", "W AWW", 1),
        first_steps=4,
        second_steps=6,
        transition_warning_steps=2,
    )
    obs, state = env.reset(jax.random.PRNGKey(0))

    assert state.steps_until_layout_change.item() == 4
    assert jnp.all(obs["agent_0"][..., -2] == 0.0)
    assert jnp.all(obs["agent_0"][..., -1] == 0.0)

    state = state.replace(step=jnp.array(2))
    obs = env.get_obs(state)
    assert jnp.all(obs["agent_0"][..., -2] == 1.0)
    assert jnp.any(obs["agent_0"][..., -1])

    state = state.replace(step=jnp.array(3))
    obs = env.get_obs(state)
    assert jnp.allclose(obs["agent_0"][..., -2], 0.5)

    obs, state, _, _, info = _step(env, state)
    assert state.layout_index.item() == 1
    assert state.steps_until_layout_change.item() == 6
    assert jnp.all(obs["agent_0"][..., -2] == 0.0)
    assert jnp.all(obs["agent_0"][..., -1] == 0.0)
    assert jnp.all(info["steps_until_layout_change"] == 6)
    assert jnp.all(info["transition_countdown"] == 0.0)


@pytest.mark.parametrize("warning_steps", (0, -1, 1.5, True))
def test_transition_warning_steps_must_be_a_positive_integer(warning_steps):
    with pytest.raises(ValueError, match="positive integer"):
        OvercookedV3(
            layout="dynamic_00",
            max_steps=20,
            transition_warning_steps=warning_steps,
        )


def test_transition_countdown_can_be_disabled_for_old_checkpoints():
    env = OvercookedV3(
        layout="dynamic_00",
        max_steps=20,
        include_transition_countdown=False,
    )
    obs, _ = env.reset(jax.random.PRNGKey(0))

    assert env.obs_shape == (env.height, env.width, 30)
    assert obs["agent_0"].shape == env.obs_shape


def test_layout_change_mask_can_be_disabled_independently():
    env = OvercookedV3(
        layout="dynamic_00",
        max_steps=20,
        include_transition_countdown=True,
        include_layout_change_mask=False,
    )
    _, state = env.reset(jax.random.PRNGKey(0))
    obs = env.get_obs(state.replace(step=jnp.array(80)))

    assert env.obs_shape == (env.height, env.width, 31)
    assert jnp.all(obs["agent_0"][..., -1] == 1.0)


def test_featurized_observation_appends_countdown_and_flat_change_mask():
    env = OvercookedV3(
        layout="dynamic_00",
        max_steps=20,
        observation_type=ObservationType.FEATURIZED,
    )
    _, state = env.reset(jax.random.PRNGKey(0))
    state = state.replace(step=jnp.array(80))
    obs = env.get_obs(state)
    agent_obs = obs["agent_0"]
    mask_size = env.height * env.width

    assert agent_obs.shape == env.obs_shape
    assert agent_obs[-(mask_size + 1)] == 1.0
    assert jnp.array_equal(
        agent_obs[-mask_size:].astype(jnp.bool_),
        state.layout_change_mask.flatten(),
    )


@pytest.mark.parametrize("layout_name", sorted(dynamic_layouts))
def test_registered_layouts_reset_and_reach_second_phase(layout_name):
    env = OvercookedV3(layout=layout_name, max_steps=500)
    _, state = env.reset(jax.random.PRNGKey(0))
    state = state.replace(step=env.phase_durations[0] - 1)

    _, state, _, _, info = _step(env, state)
    assert jnp.all(info["layout_changed"])
    assert jnp.all(info["layout_index"] == 1)
    assert state.layout_index.item() == 1


def test_dynamic_layout_validates_phase_shapes_and_durations():
    other_size = """
WWWWWW
W AA W
WOPXBW
"""
    with pytest.raises(ValueError, match="same size"):
        DynamicLayout.from_data([[BASE, 1], [other_size, 1]])
    with pytest.raises(ValueError, match="greater than zero"):
        DynamicLayout.from_data([[BASE, 0]])


def test_zero_and_o_are_loaded_as_v2_ingredient_zero_piles():
    layout = DynamicLayout.from_data([[BASE.replace("O", "0"), 7]])
    static_objects = layout.phases[0].layout.static_objects

    assert layout.cycle_steps == 7
    assert jnp.sum(static_objects == StaticObject.ingredient_pile(0)) == 1


def test_layout_cycles_at_phase_boundaries():
    blocked = BASE.replace("W A W", "W AWW", 1)
    env = _env(blocked, first_steps=2, second_steps=1)
    _, state = env.reset(jax.random.PRNGKey(0))

    _, state, _, _, info = _step(env, state, seed=1)
    assert not jnp.any(info["layout_changed"])

    _, state, _, _, info = _step(env, state, seed=2)
    assert jnp.all(info["layout_changed"])
    assert state.layout_index.item() == 1

    _, state, _, _, info = _step(env, state, seed=3)
    assert jnp.all(info["layout_changed"])
    assert state.layout_index.item() == 0


def test_changed_static_cell_clears_dynamic_object_and_extra_info():
    changed = """
WWWWW
WPA W
W A W
WOWXW
WWBWW
"""
    env = _env(changed)
    _, state = env.reset(jax.random.PRNGKey(0))
    state = state.replace(
        grid=state.grid.at[1, 1, 1].set(DynamicObject.ingredient(0)).at[1, 1, 2].set(9)
    )

    _, state, _, _, _ = _step(env, state)
    assert state.grid[1, 1, 0].item() == StaticObject.POT
    assert state.grid[1, 1, 1].item() == DynamicObject.EMPTY
    assert state.grid[1, 1, 2].item() == 0


def test_unchanged_cell_preserves_dynamic_object():
    changed = BASE.replace("W A W", "W AWW", 1)
    env = _env(changed)
    _, state = env.reset(jax.random.PRNGKey(0))
    onion = DynamicObject.ingredient(0)
    state = state.replace(grid=state.grid.at[1, 1, 1].set(onion))

    _, state, _, _, _ = _step(env, state)
    assert state.grid[1, 1, 1].item() == onion


def test_boundary_move_requires_cell_empty_before_and_after_change():
    blocked = BASE.replace("W A W\nW A W", "W A W\nW AWW")
    env = _env(blocked)
    _, state = env.reset(jax.random.PRNGKey(0))
    state = state.replace(
        agents=state.agents.replace(dir=state.agents.dir.at[1].set(Direction.RIGHT))
    )

    _, state, _, _, info = _step(env, state, _actions(env, second=0))
    assert jnp.all(info["layout_changed"])
    assert _positions(state)[1] == (2, 2)
    assert state.agents.dir[1].item() == Direction.RIGHT


def test_new_block_relocates_agent_and_preserves_direction_and_inventory():
    blocked = """
WWWWW
W A W
W WAW
WOPXW
WWBWW
"""
    env = _env(blocked)
    _, state = env.reset(jax.random.PRNGKey(0))
    onion = DynamicObject.ingredient(0)
    state = state.replace(
        agents=state.agents.replace(
            dir=state.agents.dir.at[1].set(Direction.DOWN),
            inventory=state.agents.inventory.at[1].set(onion),
        )
    )

    _, state, _, _, _ = _step(env, state)
    assert _positions(state)[1] == (3, 2)
    assert state.agents.dir[1].item() == Direction.DOWN
    assert state.agents.inventory[1].item() == onion


def test_all_adjacent_positions_blocked_uses_new_phase_agent_starts():
    blocked = """
WWWWW
WAWWW
WWWWW
WWWAX
WOPBW
"""
    env = _env(blocked)
    _, state = env.reset(jax.random.PRNGKey(0))

    _, state, _, _, _ = _step(env, state)
    assert _positions(state) == [(1, 1), (3, 3)]


def test_relocated_agents_never_overlap():
    blocked = """
WWWWW
WAWWW
WWWWW
WWWAX
WOPBW
"""
    env = _env(blocked)
    _, state = env.reset(jax.random.PRNGKey(0))

    _, state, _, _, _ = _step(env, state)
    positions = _positions(state)
    assert len(set(positions)) == env.num_agents
    for x, y in positions:
        assert state.grid[y, x, 0].item() == StaticObject.EMPTY


def test_v2_configuration_flags_remain_available():
    env = OvercookedV3(
        layout="dynamic_00",
        negative_rewards=True,
        sample_recipe_on_delivery=True,
        start_cooking_interaction=True,
        agent_view_size=2,
    )
    obs, _ = env.reset(jax.random.PRNGKey(0))

    assert env.negative_rewards is True
    assert env.obs_shape == (5, 5, 32)
    assert obs["agent_0"].shape == env.obs_shape


def test_jitted_step_reports_v2_reward_and_dynamic_layout_info():
    env = OvercookedV3(layout="dynamic_00", max_steps=20)
    _, state = env.reset(jax.random.PRNGKey(0))

    _, _, rewards, dones, infos = _step(env, state)
    assert set(rewards) == set(env.agents)
    assert "__all__" in dones
    assert set(infos) == {
        "shaped_reward",
        "layout_index",
        "layout_changed",
        "steps_until_layout_change",
        "transition_countdown",
        "layout_change_tile_count",
        "wall_tile_count",
        "ingredient_pile_count",
        "signal_tile_count",
        "left_workload_tile_count",
        "right_workload_tile_count",
        "left_ingredient_pile_count",
        "right_ingredient_pile_count",
    }
