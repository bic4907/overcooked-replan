import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxmarl.environments.overcooked_v3.common import (
    Direction,
    DynamicObject,
    OvercookedActionsEnum,
    Position,
    StaticObject,
)
from jaxmarl.environments.overcooked_v3.dynamic_layouts import dynamic_layouts
from jaxmarl.environments.overcooked_v3.dynamic_overcooked import OvercookedV3
from jaxmarl.environments.overcooked_v3.settings import INDICATOR_ACTIVATION_COST
from jaxmarl.viz.overcooked_v3_visualizer import OvercookedV3Visualizer

ROLE_SCENARIOS = (
    "split_no_sig",
    "split_sig",
    "outage_no_sig",
    "outage_sig",
)


def _reachable_floor(static_objects, start):
    width = static_objects.shape[1]
    height = static_objects.shape[0]
    frontier = [start]
    reached = {start}
    while frontier:
        x, y = frontier.pop()
        for next_x, next_y in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            next_pos = (next_x, next_y)
            if (
                0 <= next_x < width
                and 0 <= next_y < height
                and static_objects[next_y, next_x] == StaticObject.EMPTY
                and next_pos not in reached
            ):
                reached.add(next_pos)
                frontier.append(next_pos)
    return reached


@pytest.mark.parametrize(
    ("layout_name", "grid_shape"),
    (
        ("split_no_sig", (7, 13, 3)),
        ("split_sig", (7, 13, 3)),
        ("outage_no_sig", (7, 11, 3)),
        ("outage_sig", (7, 11, 3)),
    ),
)
def test_role_scenario_is_registered_and_resettable(layout_name, grid_shape):
    env = OvercookedV3(layout=layout_name, max_steps=220)
    obs, state = env.reset(jax.random.PRNGKey(0))

    assert state.grid.shape == grid_shape
    assert obs["agent_0"].shape == (*grid_shape[:2], 33)
    assert state.layout_index.item() == 0


@pytest.mark.parametrize(
    ("no_sig_name", "sig_name", "signal_position"),
    (
        ("split_no_sig", "split_sig", (6, 3)),
        ("outage_no_sig", "outage_sig", (5, 4)),
    ),
)
def test_sig_pair_only_replaces_one_counter_with_button(
    no_sig_name, sig_name, signal_position
):
    no_sig = dynamic_layouts[no_sig_name]
    sig = dynamic_layouts[sig_name]
    signal_x, signal_y = signal_position

    for no_sig_phase, sig_phase in zip(no_sig.phases, sig.phases):
        no_sig_static = no_sig_phase.layout.static_objects
        sig_static = sig_phase.layout.static_objects
        differences = np.argwhere(no_sig_static != sig_static)

        assert differences.tolist() == [[signal_y, signal_x]]
        assert no_sig_static[signal_y, signal_x] == StaticObject.RECIPE_INDICATOR
        assert sig_static[signal_y, signal_x] == StaticObject.BUTTON_RECIPE_INDICATOR


def test_kitchen_split_moves_the_complete_workload_between_connected_bays():
    layout = dynamic_layouts["split_no_sig"]
    left_active = layout.phases[0].layout.static_objects
    right_active = layout.phases[1].layout.static_objects
    workload_objects = (StaticObject.POT, StaticObject.PLATE_PILE, StaticObject.GOAL)

    def workload_count(static_objects, columns):
        return sum(
            np.count_nonzero(static_objects[:, columns] == object_type)
            for object_type in workload_objects
        )

    assert workload_count(left_active, slice(1, 6)) == 4
    assert workload_count(left_active, slice(7, 12)) == 0
    assert workload_count(right_active, slice(1, 6)) == 0
    assert workload_count(right_active, slice(7, 12)) == 4

    onion_pile = StaticObject.ingredient_pile(0)
    for static_objects in (left_active, right_active):
        assert static_objects[1, 1] == onion_pile
        assert static_objects[1, 11] == onion_pile
        assert static_objects[2, 6] == StaticObject.EMPTY
        assert static_objects[4, 6] == StaticObject.EMPTY

    support_positions = ((2, 2), (10, 2))
    for phase, support_position in zip(layout.phases, support_positions):
        first_start, second_start = phase.agent_positions
        first_reachable = _reachable_floor(phase.layout.static_objects, first_start)
        assert second_start in first_reachable
        assert support_position in first_reachable


@pytest.mark.parametrize("layout_name", ("outage_no_sig", "outage_sig"))
def test_resource_outage_separates_agents_but_keeps_a_shared_handoff(layout_name):
    layout = dynamic_layouts[layout_name]
    onion_pile = StaticObject.ingredient_pile(0)

    for phase in layout.phases:
        static_objects = phase.layout.static_objects
        left_start, right_start = phase.agent_positions
        assert right_start not in _reachable_floor(static_objects, left_start)
        assert static_objects[2, 5] == StaticObject.WALL
        assert static_objects[2, 4] == StaticObject.EMPTY
        assert static_objects[2, 6] == StaticObject.EMPTY

    normal_phase = layout.phases[0].layout.static_objects
    outage_phase = layout.phases[1].layout.static_objects
    assert normal_phase[1, 1] == onion_pile
    assert normal_phase[1, 9] == onion_pile
    assert outage_phase[1, 1] == onion_pile
    assert outage_phase[1, 9] == StaticObject.WALL

    for object_type in (StaticObject.POT, StaticObject.PLATE_PILE, StaticObject.GOAL):
        positions = np.argwhere(outage_phase == object_type)
        assert positions.shape == (2, 2)
        assert jnp.sum(positions[:, 1] < 5) == 1
        assert jnp.sum(positions[:, 1] > 5) == 1


@pytest.mark.parametrize(
    ("layout_name", "change_position", "change_count"),
    (
        ("split_no_sig", (1, 2), 8),
        ("split_sig", (1, 2), 8),
        ("outage_no_sig", (9, 1), 1),
        ("outage_sig", (9, 1), 1),
    ),
)
def test_change_mask_marks_only_the_next_static_tile_change(
    layout_name, change_position, change_count
):
    env = OvercookedV3(layout=layout_name, max_steps=220)
    obs, state = env.reset(jax.random.PRNGKey(0))
    change_x, change_y = change_position

    assert state.layout_change_mask[change_y, change_x]
    assert jnp.all(obs["agent_0"][..., -1] == 0.0)

    warning_state = state.replace(step=jnp.array(80))
    warning_obs = env.get_obs(warning_state)
    change_mask = warning_obs["agent_0"][..., -1]
    assert change_mask[change_y, change_x] == 1.0
    assert jnp.sum(change_mask) == change_count


def test_visualizer_formats_transition_countdown_in_seconds():
    env = OvercookedV3(layout="split_no_sig", max_steps=220)
    _, state = env.reset(jax.random.PRNGKey(0))
    visualizer = OvercookedV3Visualizer(seconds_per_step=0.2)

    assert visualizer.caption_with_countdown(state, "step=0") == "step=0"
    assert (
        visualizer.caption_with_countdown(
            state.replace(step=jnp.array(79), steps_until_layout_change=21),
            "step=79",
        )
        == "step=79"
    )
    assert (
        visualizer.caption_with_countdown(
            state.replace(step=jnp.array(80), steps_until_layout_change=20),
            "step=80",
        )
        == "step=80 | layout change in 20 steps (4.0s)"
    )


def test_visualizer_blinks_and_draws_count_on_each_changing_tile():
    env = OvercookedV3(layout="split_no_sig", max_steps=220)
    _, state = env.reset(jax.random.PRNGKey(0))
    visualizer = OvercookedV3Visualizer(tile_size=24)
    before_warning = state.replace(step=jnp.array(79), steps_until_layout_change=21)
    warning_on = state.replace(step=jnp.array(80), steps_until_layout_change=20)
    warning_off = state.replace(step=jnp.array(81), steps_until_layout_change=19)

    raw_before = np.asarray(visualizer._render_state(before_warning))
    frame_before = visualizer._render_frame(before_warning)
    raw_warning_on = np.asarray(visualizer._render_state(warning_on))
    raw_warning_off = np.asarray(visualizer._render_state(warning_off))
    frame_warning = visualizer._render_frame(warning_on)

    assert np.array_equal(frame_before, raw_before)
    assert not np.array_equal(raw_warning_on, raw_warning_off)

    changed_pixels = np.any(frame_warning != raw_warning_on, axis=-1)
    changed_tiles = np.zeros_like(state.layout_change_mask, dtype=bool)
    for tile_y, tile_x in np.argwhere(changed_pixels):
        changed_tiles[tile_y // visualizer.tile_size, tile_x // visualizer.tile_size] = True
    assert np.array_equal(changed_tiles, np.asarray(state.layout_change_mask))


def test_visualizer_saves_low_resolution_mp4(tmp_path):
    env = OvercookedV3(layout="split_no_sig", max_steps=2)
    _, state = env.reset(jax.random.PRNGKey(0))
    actions = {agent: jnp.array(0) for agent in env.agents}
    _, next_state, _, _, _ = env.step_env(jax.random.PRNGKey(1), state, actions)
    video_path = tmp_path / "episode.mp4"

    OvercookedV3Visualizer(tile_size=8).save_video(
        [state, next_state],
        video_path,
        captions=["step=0", "step=1"],
        fps=2,
        quality=3,
    )

    assert video_path.is_file()
    assert video_path.stat().st_size > 0
    assert video_path.read_bytes()[4:8] == b"ftyp"


def test_outage_agents_can_pass_an_onion_across_the_shared_counter():
    env = OvercookedV3(layout="outage_no_sig", max_steps=20)
    _, state = env.reset(jax.random.PRNGKey(0))
    onion = DynamicObject.ingredient(0)
    state = state.replace(
        agents=state.agents.replace(
            pos=Position(
                x=state.agents.pos.x.at[0].set(4),
                y=state.agents.pos.y.at[0].set(2),
            ),
            dir=state.agents.dir.at[0].set(Direction.RIGHT),
            inventory=state.agents.inventory.at[0].set(onion),
        )
    )
    actions = {
        env.agents[0]: jnp.array(OvercookedActionsEnum.interact),
        env.agents[1]: jnp.array(OvercookedActionsEnum.stay),
    }
    _, state, _, _, _ = env.step_env(jax.random.PRNGKey(1), state, actions)

    assert state.agents.inventory[0] == DynamicObject.EMPTY
    assert state.grid[2, 5, 1] == onion

    state = state.replace(
        agents=state.agents.replace(
            pos=Position(
                x=state.agents.pos.x.at[1].set(6),
                y=state.agents.pos.y.at[1].set(2),
            ),
            dir=state.agents.dir.at[1].set(Direction.LEFT),
        )
    )
    actions = {
        env.agents[0]: jnp.array(OvercookedActionsEnum.stay),
        env.agents[1]: jnp.array(OvercookedActionsEnum.interact),
    }
    _, state, _, _, _ = env.step_env(jax.random.PRNGKey(2), state, actions)

    assert state.grid[2, 5, 1] == DynamicObject.EMPTY
    assert state.agents.inventory[1] == onion


def test_split_runtime_moves_workload_at_step_100():
    env = OvercookedV3(layout="split_no_sig", max_steps=220)
    _, state = env.reset(jax.random.PRNGKey(0))
    state = state.replace(step=jnp.array(99))
    actions = {agent: jnp.array(OvercookedActionsEnum.stay) for agent in env.agents}

    _, state, _, _, infos = jax.jit(env.step_env)(jax.random.PRNGKey(1), state, actions)

    assert state.step.item() == 100
    assert state.layout_index.item() == 1
    assert state.grid[2, 1, 0].item() == StaticObject.EMPTY
    assert state.grid[2, 11, 0].item() == StaticObject.POT
    assert jnp.all(infos["left_workload_tile_count"] == 0)
    assert jnp.all(infos["right_workload_tile_count"] == 4)
    assert jnp.all(infos["left_ingredient_pile_count"] == 1)
    assert jnp.all(infos["right_ingredient_pile_count"] == 1)
    assert jnp.all(infos["layout_changed"])


def test_outage_runtime_removes_only_the_right_kitchen_onion_at_step_100():
    env = OvercookedV3(layout="outage_no_sig", max_steps=220)
    _, state = env.reset(jax.random.PRNGKey(0))
    state = state.replace(step=jnp.array(99))
    actions = {agent: jnp.array(OvercookedActionsEnum.stay) for agent in env.agents}

    _, state, _, _, infos = jax.jit(env.step_env)(jax.random.PRNGKey(1), state, actions)

    assert state.step.item() == 100
    assert state.layout_index.item() == 1
    assert state.grid[1, 1, 0].item() == StaticObject.ingredient_pile(0)
    assert state.grid[1, 9, 0].item() == StaticObject.WALL
    assert jnp.all(infos["left_workload_tile_count"] == 3)
    assert jnp.all(infos["right_workload_tile_count"] == 3)
    assert jnp.all(infos["left_ingredient_pile_count"] == 1)
    assert jnp.all(infos["right_ingredient_pile_count"] == 0)
    assert jnp.all(infos["layout_changed"])


@pytest.mark.parametrize(
    ("layout_name", "signal_position", "standing_position"),
    (
        ("split_sig", (6, 3), (5, 3)),
        ("outage_sig", (5, 4), (4, 4)),
    ),
)
def test_sig_button_produces_public_timed_signal(
    layout_name, signal_position, standing_position
):
    env = OvercookedV3(layout=layout_name, max_steps=20)
    _, state = env.reset(jax.random.PRNGKey(0))
    signal_x, signal_y = signal_position
    standing_x, standing_y = standing_position
    state = state.replace(
        agents=state.agents.replace(
            pos=Position(
                x=state.agents.pos.x.at[0].set(standing_x),
                y=state.agents.pos.y.at[0].set(standing_y),
            ),
            dir=state.agents.dir.at[0].set(Direction.RIGHT),
        )
    )
    actions = {
        env.agents[0]: jnp.array(OvercookedActionsEnum.interact),
        env.agents[1]: jnp.array(OvercookedActionsEnum.stay),
    }

    obs, state, rewards, _, _ = jax.jit(env.step_env)(
        jax.random.PRNGKey(1), state, actions
    )

    assert state.grid[signal_y, signal_x, 2].item() == env.signal_activation_time
    assert obs[env.agents[0]][signal_y, signal_x, -3].item() == 1.0
    assert obs[env.agents[1]][signal_y, signal_x, -3].item() == 1.0
    assert rewards[env.agents[0]].item() == pytest.approx(
        -INDICATOR_ACTIVATION_COST
    )
    assert rewards[env.agents[1]].item() == pytest.approx(
        -INDICATOR_ACTIVATION_COST
    )

    stay_actions = {
        env.agents[0]: jnp.array(OvercookedActionsEnum.stay),
        env.agents[1]: jnp.array(OvercookedActionsEnum.stay),
    }
    for expected_steps in range(env.signal_activation_time - 1, 0, -1):
        obs, state, _, _, _ = jax.jit(env.step_env)(
            jax.random.PRNGKey(expected_steps + 10), state, stay_actions
        )
        assert state.grid[signal_y, signal_x, 2].item() == expected_steps
        assert obs[env.agents[1]][signal_y, signal_x, -3].item() == pytest.approx(
            expected_steps / env.signal_activation_time
        )

    obs, state, _, _, _ = jax.jit(env.step_env)(
        jax.random.PRNGKey(30), state, stay_actions
    )
    assert state.grid[signal_y, signal_x, 2].item() == 0
    assert obs[env.agents[1]][signal_y, signal_x, -3].item() == 0.0


def test_active_signal_is_labeled_in_rendered_button_tile():
    env = OvercookedV3(layout="split_sig", max_steps=20)
    _, state = env.reset(jax.random.PRNGKey(0))
    signal_x, signal_y = 6, 3
    state = state.replace(
        grid=state.grid.at[signal_y, signal_x, 2].set(env.signal_activation_time)
    )
    visualizer = OvercookedV3Visualizer(tile_size=32)

    raw_frame = np.asarray(visualizer._render_state(state))
    labeled_frame = visualizer._render_frame(state)
    changed_pixels = np.any(raw_frame != labeled_frame, axis=-1)
    changed_tiles = np.zeros_like(state.layout_change_mask, dtype=bool)
    for pixel_y, pixel_x in np.argwhere(changed_pixels):
        changed_tiles[pixel_y // visualizer.tile_size, pixel_x // visualizer.tile_size] = (
            True
        )

    expected_tiles = np.zeros_like(changed_tiles)
    expected_tiles[signal_y, signal_x] = True
    assert np.array_equal(changed_tiles, expected_tiles)


def test_no_sig_placeholder_does_not_activate_or_store_objects():
    env = OvercookedV3(layout="split_no_sig", max_steps=20)
    _, state = env.reset(jax.random.PRNGKey(0))
    state = state.replace(
        agents=state.agents.replace(
            pos=Position(
                x=state.agents.pos.x.at[0].set(5),
                y=state.agents.pos.y.at[0].set(3),
            ),
            dir=state.agents.dir.at[0].set(Direction.RIGHT),
            inventory=state.agents.inventory.at[0].set(DynamicObject.ingredient(0)),
        )
    )
    actions = {
        env.agents[0]: jnp.array(OvercookedActionsEnum.interact),
        env.agents[1]: jnp.array(OvercookedActionsEnum.stay),
    }

    _, state, rewards, _, _ = jax.jit(env.step_env)(
        jax.random.PRNGKey(1), state, actions
    )

    assert state.grid[3, 6, 0].item() == StaticObject.RECIPE_INDICATOR
    assert state.grid[3, 6, 1].item() == DynamicObject.EMPTY
    assert state.grid[3, 6, 2].item() == 0
    assert state.agents.inventory[0] == DynamicObject.ingredient(0)
    assert rewards[env.agents[0]].item() == 0
