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
from jaxmarl.environments.overcooked_v3.dynamic_layouts import (
    ROLE_SCENARIO_LAYOUTS,
    dynamic_layouts,
)
from jaxmarl.environments.overcooked_v3.dynamic_overcooked import OvercookedV3
from jaxmarl.environments.overcooked_v3.settings import INDICATOR_ACTIVATION_COST
from jaxmarl.viz.overcooked_v3_visualizer import OvercookedV3Visualizer

ROLE_SCENARIOS = (
    "split_no_sig",
    "split_sig",
    "outage_no_sig",
    "outage_sig",
)

CANONICAL_ROLE_SCENARIOS = tuple(
    name for names in ROLE_SCENARIO_LAYOUTS.values() for name in names
)
SPLIT_RESOURCE_COUNTS = (1, 2, 1, 1)
OUTAGE_RESOURCE_COUNTS = (1, 1, 1, 1)


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


def _can_interact_from(static_objects, object_position, reachable_floor):
    x, y = object_position
    return any(
        neighbor in reachable_floor
        for neighbor in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1))
    )


def _shortest_floor_distance(static_objects, starts, goals):
    goals = set(goals)
    frontier = [(position, 0) for position in starts]
    reached = set(starts)
    while frontier:
        (x, y), distance = frontier.pop(0)
        if (x, y) in goals:
            return distance
        for next_position in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            next_x, next_y = next_position
            if (
                0 <= next_y < static_objects.shape[0]
                and 0 <= next_x < static_objects.shape[1]
                and static_objects[next_y, next_x] == StaticObject.EMPTY
                and next_position not in reached
            ):
                reached.add(next_position)
                frontier.append((next_position, distance + 1))
    return None


def test_each_role_scenario_family_has_one_selected_layout():
    assert set(ROLE_SCENARIO_LAYOUTS) == {
        "splitnosig",
        "splitsig",
        "outagenosig",
        "outagesig",
    }
    for family, names in ROLE_SCENARIO_LAYOUTS.items():
        assert names == (f"{family}_0",)
        signatures = {
            tuple(
                phase.layout.static_objects.tobytes()
                for phase in dynamic_layouts[name].phases
            )
            for name in names
        }
        assert len(signatures) == 1


@pytest.mark.parametrize("layout_name", CANONICAL_ROLE_SCENARIOS)
def test_all_role_scenario_variants_are_resettable(layout_name):
    env = OvercookedV3(layout=layout_name, max_steps=220)
    obs, state = env.reset(jax.random.PRNGKey(0))

    expected_shape = (7, 11, 3) if layout_name.startswith("split") else (6, 9, 3)
    assert state.grid.shape == expected_shape
    assert obs["agent_0"].shape == (*expected_shape[:2], 33)


@pytest.mark.parametrize("variant", range(1))
@pytest.mark.parametrize(
    ("no_sig_family", "sig_family"),
    (("splitnosig", "splitsig"), ("outagenosig", "outagesig")),
)
def test_each_sig_variant_only_replaces_one_indicator(
    variant, no_sig_family, sig_family
):
    no_sig = dynamic_layouts[f"{no_sig_family}_{variant}"]
    sig = dynamic_layouts[f"{sig_family}_{variant}"]

    for no_sig_phase, sig_phase in zip(no_sig.phases, sig.phases):
        no_sig_static = no_sig_phase.layout.static_objects
        sig_static = sig_phase.layout.static_objects
        differences = np.argwhere(no_sig_static != sig_static)

        assert differences.shape == (1, 2)
        signal_y, signal_x = differences[0]
        assert no_sig_static[signal_y, signal_x] == StaticObject.RECIPE_INDICATOR
        assert sig_static[signal_y, signal_x] == StaticObject.BUTTON_RECIPE_INDICATOR


@pytest.mark.parametrize("variant", range(1))
def test_outage_makes_cross_kitchen_supply_a_short_route(variant):
    layout = dynamic_layouts[f"outagenosig_{variant}"]
    assert tuple(phase.steps for phase in layout.phases) == (40, 160)
    normal_phase = layout.phases[0].layout.static_objects
    outage_phase = layout.phases[1].layout.static_objects
    left_start, right_start = layout.phases[0].agent_positions
    onion = StaticObject.ingredient_pile(0)

    assert normal_phase.shape == (6, 9)
    assert right_start not in _reachable_floor(normal_phase, left_start)
    assert right_start not in _reachable_floor(outage_phase, left_start)
    assert _shortest_floor_distance(outage_phase, {left_start}, {right_start}) is None
    assert np.sum(outage_phase[:, 4] == StaticObject.EMPTY) == 0
    assert np.sum(outage_phase[:, 4] == StaticObject.RECIPE_INDICATOR) == 1
    differences = np.argwhere(normal_phase != outage_phase)
    expected_onions, expected_pots, expected_plates, expected_goals = (
        OUTAGE_RESOURCE_COUNTS
    )
    assert differences.shape == (expected_onions, 2)
    for outage_y, outage_x in differences:
        assert outage_x > 4
        assert normal_phase[outage_y, outage_x] == onion
        assert outage_phase[outage_y, outage_x] == StaticObject.WALL

    expected = {
        onion: expected_onions,
        StaticObject.POT: expected_pots,
        StaticObject.PLATE_PILE: expected_plates,
        StaticObject.GOAL: expected_goals,
    }
    for object_type, count in expected.items():
        assert np.sum(normal_phase[:, :4] == object_type) == count
        assert np.sum(normal_phase[:, 5:] == object_type) == count
    assert np.sum(outage_phase[:, :4] == onion) == expected_onions
    assert np.sum(outage_phase[:, 5:] == onion) == 0

    left_reachable = _reachable_floor(outage_phase, left_start)
    right_reachable = _reachable_floor(outage_phase, right_start)
    for object_type in (
        onion,
        StaticObject.POT,
        StaticObject.PLATE_PILE,
        StaticObject.GOAL,
    ):
        for y, x in np.argwhere(outage_phase == object_type):
            assert _can_interact_from(
                outage_phase,
                (x, y),
                left_reachable if x < 4 else right_reachable,
            )
    handoffs = [
        ((3, y), (5, y))
        for y in range(1, outage_phase.shape[0] - 1)
        if outage_phase[y, 4] == StaticObject.WALL
        and (3, y) in left_reachable
        and (5, y) in right_reachable
    ]
    assert handoffs

    left_onion_y, left_onion_x = np.argwhere(outage_phase[:, :4] == onion)[0]
    onion_interaction_floors = {
        position
        for position in (
            (left_onion_x - 1, left_onion_y),
            (left_onion_x + 1, left_onion_y),
            (left_onion_x, left_onion_y - 1),
            (left_onion_x, left_onion_y + 1),
        )
        if position in left_reachable
    }
    right_pot_y, relative_pot_x = np.argwhere(outage_phase[:, 5:] == StaticObject.POT)[
        0
    ]
    right_pot_x = relative_pot_x + 5
    pot_interaction_floors = {
        position
        for position in (
            (right_pot_x - 1, right_pot_y),
            (right_pot_x + 1, right_pot_y),
            (right_pot_x, right_pot_y - 1),
            (right_pot_x, right_pot_y + 1),
        )
        if position in right_reachable
    }
    left_handoff_floors = {left for left, _right in handoffs}
    right_handoff_floors = {right for _left, right in handoffs}
    assert (
        _shortest_floor_distance(
            outage_phase, onion_interaction_floors, left_handoff_floors
        )
        <= 2
    )
    assert (
        _shortest_floor_distance(
            outage_phase, right_handoff_floors, pot_interaction_floors
        )
        <= 2
    )


@pytest.mark.parametrize("variant", range(1))
def test_split_variants_change_resource_capacity(variant):
    layout = dynamic_layouts[f"splitnosig_{variant}"]
    assert tuple(phase.steps for phase in layout.phases) == (40, 160)
    open_phase = layout.phases[0].layout.static_objects
    closed_phase = layout.phases[1].layout.static_objects
    left_start, right_start = layout.phases[0].agent_positions
    expected_onions, expected_pots, expected_plates, expected_goals = (
        SPLIT_RESOURCE_COUNTS
    )
    onion = StaticObject.ingredient_pile(0)

    differences = np.argwhere(open_phase != closed_phase)
    assert differences.shape == (1, 2)
    door_y, door_x = differences[0]
    assert door_x == 5
    assert open_phase[door_y, door_x] == StaticObject.EMPTY
    assert closed_phase[door_y, door_x] == StaticObject.WALL
    assert right_start in _reachable_floor(open_phase, left_start)
    assert right_start not in _reachable_floor(closed_phase, left_start)

    expected_left = {onion: expected_onions, StaticObject.POT: expected_pots}
    expected_right = {
        StaticObject.PLATE_PILE: expected_plates,
        StaticObject.GOAL: expected_goals,
    }
    for phase in (open_phase, closed_phase):
        for object_type, count in expected_left.items():
            assert np.sum(phase[:, :5] == object_type) == count
            assert np.sum(phase[:, 6:] == object_type) == 0
        for object_type, count in expected_right.items():
            assert np.sum(phase[:, :5] == object_type) == 0
            assert np.sum(phase[:, 6:] == object_type) == count

    left_reachable = _reachable_floor(closed_phase, left_start)
    right_reachable = _reachable_floor(closed_phase, right_start)
    for object_type in expected_left:
        for y, x in np.argwhere(closed_phase == object_type):
            assert _can_interact_from(closed_phase, (x, y), left_reachable)
    for object_type in expected_right:
        for y, x in np.argwhere(closed_phase == object_type):
            assert _can_interact_from(closed_phase, (x, y), right_reachable)


@pytest.mark.parametrize(
    ("layout_name", "grid_shape"),
    (
        ("split_no_sig", (7, 11, 3)),
        ("split_sig", (7, 11, 3)),
        ("outage_no_sig", (6, 9, 3)),
        ("outage_sig", (6, 9, 3)),
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
        ("split_no_sig", "split_sig", (5, 4)),
        ("outage_no_sig", "outage_sig", (4, 3)),
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


def test_kitchen_split_closes_doorway_between_complementary_bays():
    layout = dynamic_layouts["split_no_sig"]
    open_phase = layout.phases[0].layout.static_objects
    split_phase = layout.phases[1].layout.static_objects
    left_start, right_start = layout.phases[0].agent_positions

    onion_pile = StaticObject.ingredient_pile(0)
    for static_objects in (open_phase, split_phase):
        assert static_objects[0, 1] == onion_pile
        assert jnp.sum(static_objects[:, :5] == StaticObject.POT) == 2
        assert jnp.sum(static_objects[:, 6:] == StaticObject.POT) == 0
        assert jnp.sum(static_objects[:, :5] == StaticObject.PLATE_PILE) == 0
        assert jnp.sum(static_objects[:, 6:] == StaticObject.PLATE_PILE) == 1
        assert jnp.sum(static_objects[:, :5] == StaticObject.GOAL) == 0
        assert jnp.sum(static_objects[:, 6:] == StaticObject.GOAL) == 1

    assert open_phase[2, 5] == StaticObject.EMPTY
    assert right_start in _reachable_floor(open_phase, left_start)
    assert split_phase[2, 5] == StaticObject.WALL
    assert right_start not in _reachable_floor(split_phase, left_start)
    assert split_phase[2, 4] == StaticObject.EMPTY
    assert split_phase[2, 6] == StaticObject.EMPTY
    assert np.sum(open_phase != split_phase) == 1


@pytest.mark.parametrize("layout_name", ("outage_no_sig", "outage_sig"))
def test_resource_outage_separates_agents_but_keeps_a_shared_handoff(layout_name):
    layout = dynamic_layouts[layout_name]
    onion_pile = StaticObject.ingredient_pile(0)

    for phase in layout.phases:
        static_objects = phase.layout.static_objects
        left_start, right_start = phase.agent_positions
        assert right_start not in _reachable_floor(static_objects, left_start)
        assert static_objects[2, 4] == StaticObject.WALL
        assert static_objects[2, 3] == StaticObject.EMPTY
        assert static_objects[2, 5] == StaticObject.EMPTY

    normal_phase = layout.phases[0].layout.static_objects
    outage_phase = layout.phases[1].layout.static_objects
    assert normal_phase[0, 3] == onion_pile
    assert normal_phase[0, 5] == onion_pile
    assert outage_phase[0, 3] == onion_pile
    assert outage_phase[0, 5] == StaticObject.WALL

    for object_type in (StaticObject.POT, StaticObject.PLATE_PILE, StaticObject.GOAL):
        positions = np.argwhere(outage_phase == object_type)
        assert positions.shape == (2, 2)
        assert jnp.sum(positions[:, 1] < 4) == 1
        assert jnp.sum(positions[:, 1] > 4) == 1


@pytest.mark.parametrize(
    ("layout_name", "change_position", "change_count"),
    (
        ("split_no_sig", (5, 2), 1),
        ("split_sig", (5, 2), 1),
        ("outage_no_sig", (5, 0), 1),
        ("outage_sig", (5, 0), 1),
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

    warning_step = int(env.phase_durations[0]) - env.transition_warning_steps
    warning_state = state.replace(step=jnp.array(warning_step))
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
            state.replace(step=jnp.array(19), steps_until_layout_change=21),
            "step=19",
        )
        == "step=19"
    )
    assert (
        visualizer.caption_with_countdown(
            state.replace(step=jnp.array(20), steps_until_layout_change=20),
            "step=20",
        )
        == "step=20 | layout change in 20 steps (4.0s)"
    )


def test_visualizer_blinks_and_draws_count_on_each_changing_tile():
    env = OvercookedV3(layout="split_no_sig", max_steps=220)
    _, state = env.reset(jax.random.PRNGKey(0))
    visualizer = OvercookedV3Visualizer(tile_size=24)
    before_warning = state.replace(step=jnp.array(19), steps_until_layout_change=21)
    warning_on = state.replace(step=jnp.array(20), steps_until_layout_change=20)
    warning_off = state.replace(step=jnp.array(21), steps_until_layout_change=19)

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
        changed_tiles[
            tile_y // visualizer.tile_size, tile_x // visualizer.tile_size
        ] = True
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
                x=state.agents.pos.x.at[0].set(3),
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
    assert state.grid[2, 4, 1] == onion

    state = state.replace(
        agents=state.agents.replace(
            pos=Position(
                x=state.agents.pos.x.at[1].set(5),
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

    assert state.grid[2, 4, 1] == DynamicObject.EMPTY
    assert state.agents.inventory[1] == onion


def test_split_runtime_closes_handoff_wall_at_step_40():
    env = OvercookedV3(layout="split_no_sig", max_steps=220)
    _, state = env.reset(jax.random.PRNGKey(0))
    state = state.replace(step=jnp.array(39))
    actions = {agent: jnp.array(OvercookedActionsEnum.stay) for agent in env.agents}

    _, state, _, _, infos = jax.jit(env.step_env)(jax.random.PRNGKey(1), state, actions)

    assert state.step.item() == 40
    assert state.layout_index.item() == 1
    assert state.grid[2, 5, 0].item() == StaticObject.WALL
    assert jnp.all(infos["left_workload_tile_count"] == 2)
    assert jnp.all(infos["right_workload_tile_count"] == 2)
    assert jnp.all(infos["left_ingredient_pile_count"] == 1)
    assert jnp.all(infos["right_ingredient_pile_count"] == 0)
    assert jnp.all(infos["layout_changed"])


def test_outage_runtime_removes_only_the_right_kitchen_onion_at_step_40():
    env = OvercookedV3(layout="outage_no_sig", max_steps=220)
    _, state = env.reset(jax.random.PRNGKey(0))
    state = state.replace(step=jnp.array(39))
    actions = {agent: jnp.array(OvercookedActionsEnum.stay) for agent in env.agents}

    _, state, _, _, infos = jax.jit(env.step_env)(jax.random.PRNGKey(1), state, actions)

    assert state.step.item() == 40
    assert state.layout_index.item() == 1
    assert state.grid[0, 3, 0].item() == StaticObject.ingredient_pile(0)
    assert state.grid[0, 5, 0].item() == StaticObject.WALL
    assert jnp.all(infos["left_workload_tile_count"] == 3)
    assert jnp.all(infos["right_workload_tile_count"] == 3)
    assert jnp.all(infos["left_ingredient_pile_count"] == 1)
    assert jnp.all(infos["right_ingredient_pile_count"] == 0)
    assert jnp.all(infos["layout_changed"])


@pytest.mark.parametrize(
    ("layout_name", "signal_position", "standing_position"),
    (
        ("split_sig", (5, 4), (4, 4)),
        ("outage_sig", (4, 3), (3, 3)),
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
    assert rewards[env.agents[0]].item() == pytest.approx(-INDICATOR_ACTIVATION_COST)
    assert rewards[env.agents[1]].item() == pytest.approx(-INDICATOR_ACTIVATION_COST)

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
    signal_x, signal_y = 5, 4
    state = state.replace(
        grid=state.grid.at[signal_y, signal_x, 2].set(env.signal_activation_time)
    )
    visualizer = OvercookedV3Visualizer(tile_size=32)

    raw_frame = np.asarray(visualizer._render_state(state))
    labeled_frame = visualizer._render_frame(state)
    changed_pixels = np.any(raw_frame != labeled_frame, axis=-1)
    changed_tiles = np.zeros_like(state.layout_change_mask, dtype=bool)
    for pixel_y, pixel_x in np.argwhere(changed_pixels):
        changed_tiles[
            pixel_y // visualizer.tile_size, pixel_x // visualizer.tile_size
        ] = True

    expected_tiles = np.zeros_like(changed_tiles)
    expected_tiles[signal_y, signal_x] = True
    assert np.array_equal(changed_tiles, expected_tiles)


def test_no_sig_placeholder_does_not_activate_or_store_objects():
    env = OvercookedV3(layout="split_no_sig", max_steps=20)
    _, state = env.reset(jax.random.PRNGKey(0))
    state = state.replace(
        agents=state.agents.replace(
            pos=Position(
                x=state.agents.pos.x.at[0].set(4),
                y=state.agents.pos.y.at[0].set(4),
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

    assert state.grid[4, 5, 0].item() == StaticObject.RECIPE_INDICATOR
    assert state.grid[4, 5, 1].item() == DynamicObject.EMPTY
    assert state.grid[4, 5, 2].item() == 0
    assert state.agents.inventory[0] == DynamicObject.ingredient(0)
    assert rewards[env.agents[0]].item() == 0
