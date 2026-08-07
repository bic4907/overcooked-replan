import jax
import jax.numpy as jnp
import pytest

from jaxmarl.environments.overcooked.common import (
    DIR_TO_VEC,
    OBJECT_INDEX_TO_VEC,
    OBJECT_TO_INDEX,
)
from jaxmarl.environments.overcooked.dynamic_layouts import (
    DynamicLayout,
    dynamic_layouts,
)
from jaxmarl.environments.overcooked.dynamic_overcooked import DynamicOvercooked
from jaxmarl.environments.overcooked.overcooked import OvercookedActions

BASE = """
WWWWW
W A W
W A W
WOPXW
WWBWW
"""


def _env(second, first_steps=1, second_steps=1):
    layout = DynamicLayout.from_data(
        [[BASE, first_steps], [second, second_steps]]
    )
    return DynamicOvercooked(layout=layout, max_steps=20)


def _actions(env, first=OvercookedActions.stay, second=OvercookedActions.stay):
    return {
        env.agents[0]: jnp.array(first, dtype=jnp.uint32),
        env.agents[1]: jnp.array(second, dtype=jnp.uint32),
    }


def _step(env, state, actions=None, seed=1):
    if actions is None:
        actions = _actions(env)
    return jax.jit(env.step_env)(jax.random.PRNGKey(seed), state, actions)


@pytest.mark.parametrize("layout_name", sorted(dynamic_layouts))
def test_registered_layouts_reset_and_reach_second_phase(layout_name):
    env = DynamicOvercooked(layout=layout_name, max_steps=500)
    _, state = env.reset(jax.random.PRNGKey(0))
    state = state.replace(step=env.phase_durations[0] - 1)

    _, _, _, _, info = _step(env, state)
    assert jnp.all(info["layout_changed"])
    assert jnp.all(info["layout_index"] == 1)


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


def test_zero_character_is_loaded_as_v1_onion_pile():
    with_zero = BASE.replace("O", "0")
    layout = DynamicLayout.from_data([[with_zero, 7]])
    assert layout.cycle_steps == 7
    assert len(layout.phases[0].layout["onion_pile_idx"]) == 1


def test_layout_cycles_at_phase_boundaries():
    blocked = BASE.replace("W A W", "W AWW", 1)
    env = _env(blocked, first_steps=2, second_steps=1)
    _, state = env.reset(jax.random.PRNGKey(0))

    _, state, _, _, info = _step(env, state, seed=1)
    assert not jnp.any(info["layout_changed"])

    _, state, _, _, info = _step(env, state, seed=2)
    assert jnp.all(info["layout_changed"])
    assert jnp.all(info["layout_index"] == 1)

    _, state, _, _, info = _step(env, state, seed=3)
    assert jnp.all(info["layout_changed"])
    assert jnp.all(info["layout_index"] == 0)


def test_changed_static_cell_clears_loose_object():
    changed = """
WWWWW
WPA W
W A W
WOWXW
WWBWW
"""
    env = _env(changed)
    _, state = env.reset(jax.random.PRNGKey(0))
    padding = env.agent_view_size - 1
    state = state.replace(
        maze_map=state.maze_map.at[padding + 1, padding + 1].set(
            OBJECT_INDEX_TO_VEC[OBJECT_TO_INDEX["onion"]]
        )
    )

    _, state, _, _, _ = _step(env, state)
    assert (
        state.maze_map[padding + 1, padding + 1, 0].item()
        == OBJECT_TO_INDEX["pot"]
    )


def test_boundary_move_requires_cell_empty_before_and_after_change():
    blocked = BASE.replace("W A W\nW A W", "W A W\nW AWW")
    env = _env(blocked)
    _, state = env.reset(jax.random.PRNGKey(0))
    state = state.replace(
        agent_dir_idx=state.agent_dir_idx.at[1].set(OvercookedActions.right),
        agent_dir=state.agent_dir.at[1].set(DIR_TO_VEC[OvercookedActions.right]),
    )

    _, state, _, _, info = _step(
        env,
        state,
        _actions(env, second=OvercookedActions.right),
    )
    assert jnp.all(info["layout_changed"])
    assert tuple(map(int, state.agent_pos[1])) == (2, 2)
    assert state.agent_dir_idx[1].item() == OvercookedActions.right


def test_new_block_pushes_agent_opposite_facing_without_changing_inventory():
    blocked = """
WWWWW
WA  W
W W W
WOPXW
WWABW
"""
    env = _env(blocked)
    _, state = env.reset(jax.random.PRNGKey(0))
    onion = OBJECT_TO_INDEX["onion"]
    state = state.replace(
        agent_pos=state.agent_pos.at[0].set(jnp.array([1, 1], dtype=jnp.uint32)),
        agent_dir_idx=state.agent_dir_idx.at[1].set(OvercookedActions.down),
        agent_dir=state.agent_dir.at[1].set(DIR_TO_VEC[OvercookedActions.down]),
        agent_inv=state.agent_inv.at[1].set(onion),
    )

    _, state, _, _, _ = _step(env, state)
    assert tuple(map(int, state.agent_pos[1])) == (2, 1)
    assert state.agent_dir_idx[1].item() == OvercookedActions.down
    assert state.agent_inv[1].item() == onion


def test_blocked_opposite_position_searches_clockwise():
    blocked = """
WWWWW
W WWW
W W W
WAOPX
WWABW
"""
    env = _env(blocked)
    _, state = env.reset(jax.random.PRNGKey(0))
    state = state.replace(
        agent_pos=state.agent_pos.at[0].set(jnp.array([1, 3], dtype=jnp.uint32)),
        agent_dir_idx=state.agent_dir_idx.at[1].set(OvercookedActions.down),
        agent_dir=state.agent_dir.at[1].set(DIR_TO_VEC[OvercookedActions.down]),
    )

    _, state, _, _, _ = _step(env, state)
    assert tuple(map(int, state.agent_pos[1])) == (3, 2)
    assert state.agent_dir_idx[1].item() == OvercookedActions.down


def test_all_adjacent_positions_blocked_uses_new_phase_agent_start():
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

    assert tuple(map(int, state.agent_pos[0])) == (1, 1)
    assert tuple(map(int, state.agent_pos[1])) == (3, 3)
