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
from jaxmarl.environments.overcooked_v3.layouts import overcooked_v3_base_layouts
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


def test_each_role_scenario_family_has_expected_unique_layouts():
    expected_counts = {
        "splitnosig": 3,
        "splitsig": 3,
        "outagenosig": 3,
        "outagesig": 3,
        "recipe_switch": 3,
        "distance_switch": 3,
    }
    assert set(ROLE_SCENARIO_LAYOUTS) == set(expected_counts)
    for family, names in ROLE_SCENARIO_LAYOUTS.items():
        count = expected_counts[family]
        assert names == tuple(f"{family}_{variant}" for variant in range(count))
        signatures = {
            tuple(
                phase.layout.static_objects.tobytes()
                for phase in dynamic_layouts[name].phases
            )
            for name in names
        }
        assert len(signatures) == count


@pytest.mark.parametrize("layout_name", CANONICAL_ROLE_SCENARIOS)
def test_all_role_scenario_variants_are_resettable(layout_name):
    env = OvercookedV3(layout=layout_name, max_steps=220)
    obs, state = env.reset(jax.random.PRNGKey(0))

    layout_shape = dynamic_layouts[layout_name].initial_layout.static_objects.shape
    expected_shape = (*layout_shape, 3)
    expected_channels = 35 if layout_name.startswith("recipe_switch") else 33
    assert state.grid.shape == expected_shape
    assert obs["agent_0"].shape == (*expected_shape[:2], expected_channels)


@pytest.mark.parametrize("layout_name", ROLE_SCENARIO_LAYOUTS["distance_switch"])
def test_distance_switch_keeps_local_access_and_reverses_role_costs(
    layout_name,
):
    layout = dynamic_layouts[layout_name]
    assert tuple(phase.steps for phase in layout.phases) == (150, 150, 1000)
    assert all(phase.recipe is None for phase in layout.phases)

    phase_a, phase_b, phase_a_return = (
        phase.layout.static_objects for phase in layout.phases
    )
    starts = layout.phases[0].agent_positions
    agent_0_start, agent_1_start = starts
    assert all(phase.agent_positions == starts for phase in layout.phases)
    assert np.array_equal(phase_a, phase_a_return)
    assert np.array_equal(
        phase_a == StaticObject.EMPTY,
        phase_b == StaticObject.EMPTY,
    )
    reachable_0 = _reachable_floor(phase_a, agent_0_start)
    reachable_1 = _reachable_floor(phase_a, agent_1_start)
    assert agent_1_start not in reachable_0
    assert agent_0_start not in reachable_1
    assert np.sum(phase_a != phase_b) == 4

    onion = StaticObject.ingredient_pile(0)
    station_types = (
        onion,
        StaticObject.POT,
        StaticObject.PLATE_PILE,
        StaticObject.GOAL,
    )
    for phase in (phase_a, phase_b):
        for station_type in station_types:
            positions = np.argwhere(phase == station_type)
            assert positions.shape == (2, 2)
            for reachable in (reachable_0, reachable_1):
                assert any(
                    _can_interact_from(phase, (x, y), reachable) for y, x in positions
                )

    assert np.array_equal(
        np.argwhere(phase_a == StaticObject.POT),
        np.argwhere(phase_b == StaticObject.POT),
    )
    assert np.array_equal(
        np.argwhere(phase_a == StaticObject.PLATE_PILE),
        np.argwhere(phase_b == StaticObject.PLATE_PILE),
    )
    assert {tuple(position) for position in np.argwhere(phase_a == onion)} == {
        tuple(position) for position in np.argwhere(phase_b == StaticObject.GOAL)
    }
    assert {
        tuple(position) for position in np.argwhere(phase_a == StaticObject.GOAL)
    } == {tuple(position) for position in np.argwhere(phase_b == onion)}


def test_distance_switch_zero_starts_from_canonical_asymmetric_advantages():
    distance_switch = dynamic_layouts["distance_switch_0"].phases[0]
    canonical = overcooked_v3_base_layouts["asymm_advantages"]
    assert np.array_equal(
        distance_switch.layout.static_objects, canonical.static_objects
    )
    assert distance_switch.agent_positions == tuple(canonical.agent_positions)


def test_outage_uses_two_onion_recipe_while_split_keeps_three_onions():
    onion = DynamicObject.ingredient(0)

    for layout_name in (
        *ROLE_SCENARIO_LAYOUTS["splitnosig"],
        *ROLE_SCENARIO_LAYOUTS["splitsig"],
    ):
        env = OvercookedV3(layout=layout_name, max_steps=20)
        _, state = env.reset(jax.random.PRNGKey(0))
        assert env.recipe_size == 3
        assert state.recipe == 3 * onion

    for layout_name in (
        *ROLE_SCENARIO_LAYOUTS["outagenosig"],
        *ROLE_SCENARIO_LAYOUTS["outagesig"],
    ):
        env = OvercookedV3(layout=layout_name, max_steps=20)
        _, state = env.reset(jax.random.PRNGKey(0))
        assert env.recipe_size == 2
        assert state.recipe == 2 * onion


def test_outage_pot_starts_cooking_after_second_onion():
    env = OvercookedV3(layout="outage_no_sig", max_steps=20)
    _, state = env.reset(jax.random.PRNGKey(0))
    onion = DynamicObject.ingredient(0)
    state = state.replace(
        grid=state.grid.at[0, 2, 1].set(onion),
        agents=state.agents.replace(
            pos=Position(
                x=state.agents.pos.x.at[0].set(2),
                y=state.agents.pos.y.at[0].set(1),
            ),
            dir=state.agents.dir.at[0].set(Direction.UP),
            inventory=state.agents.inventory.at[0].set(onion),
        ),
    )
    actions = {
        env.agents[0]: jnp.array(OvercookedActionsEnum.interact),
        env.agents[1]: jnp.array(OvercookedActionsEnum.stay),
    }

    _, state, _, _, _ = env.step_env(jax.random.PRNGKey(1), state, actions)

    assert DynamicObject.ingredient_count(state.grid[0, 2, 1]) == 2
    assert state.grid[0, 2, 2] == 19


@pytest.mark.parametrize("variant", range(3))
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
        assert no_sig_static[signal_y, signal_x] == StaticObject.INERT_SIGNAL_INDICATOR
        assert sig_static[signal_y, signal_x] == StaticObject.BUTTON_RECIPE_INDICATOR


@pytest.mark.parametrize("variant", range(3))
def test_outage_makes_cross_kitchen_supply_a_short_route(variant):
    layout = dynamic_layouts[f"outagenosig_{variant}"]
    assert tuple(phase.steps for phase in layout.phases) == (40, 160)
    normal_phase = layout.phases[0].layout.static_objects
    outage_phase = layout.phases[1].layout.static_objects
    left_start, right_start = layout.phases[0].agent_positions
    onion = StaticObject.ingredient_pile(0)

    assert normal_phase.shape == (5, 7)
    assert right_start not in _reachable_floor(normal_phase, left_start)
    assert right_start not in _reachable_floor(outage_phase, left_start)
    assert _shortest_floor_distance(outage_phase, {left_start}, {right_start}) is None
    assert np.sum(outage_phase[:, 3] == StaticObject.EMPTY) == 0
    assert np.sum(outage_phase[:, 3] == StaticObject.RECIPE_INDICATOR) == 1
    differences = np.argwhere(normal_phase != outage_phase)
    expected_onions = np.sum(normal_phase[:, :3] == onion)
    expected_pots = np.sum(normal_phase[:, :3] == StaticObject.POT)
    expected_plates = np.sum(normal_phase[:, :3] == StaticObject.PLATE_PILE)
    expected_goals = np.sum(normal_phase[:, :3] == StaticObject.GOAL)
    assert min(expected_onions, expected_pots, expected_plates, expected_goals) >= 1
    assert differences.shape == (expected_onions, 2)
    for outage_y, outage_x in differences:
        assert outage_x > 3
        assert normal_phase[outage_y, outage_x] == onion
        assert outage_phase[outage_y, outage_x] == StaticObject.WALL

    expected = {
        onion: expected_onions,
        StaticObject.POT: expected_pots,
        StaticObject.PLATE_PILE: expected_plates,
        StaticObject.GOAL: expected_goals,
    }
    for object_type, count in expected.items():
        assert np.sum(normal_phase[:, :3] == object_type) == count
        assert np.sum(normal_phase[:, 4:] == object_type) == count
    assert np.sum(outage_phase[:, :3] == onion) == expected_onions
    assert np.sum(outage_phase[:, 4:] == onion) == 0

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
                left_reachable if x < 3 else right_reachable,
            )
    handoffs = [
        ((2, y), (4, y))
        for y in range(1, outage_phase.shape[0] - 1)
        if outage_phase[y, 3] == StaticObject.WALL
        and (2, y) in left_reachable
        and (4, y) in right_reachable
    ]
    assert handoffs

    onion_interaction_floors = set()
    for left_onion_y, left_onion_x in np.argwhere(outage_phase[:, :3] == onion):
        onion_interaction_floors.update(
            position
            for position in (
                (left_onion_x - 1, left_onion_y),
                (left_onion_x + 1, left_onion_y),
                (left_onion_x, left_onion_y - 1),
                (left_onion_x, left_onion_y + 1),
            )
            if position in left_reachable
        )
    pot_interaction_floors = set()
    for right_pot_y, relative_pot_x in np.argwhere(
        outage_phase[:, 4:] == StaticObject.POT
    ):
        right_pot_x = relative_pot_x + 4
        pot_interaction_floors.update(
            position
            for position in (
                (right_pot_x - 1, right_pot_y),
                (right_pot_x + 1, right_pot_y),
                (right_pot_x, right_pot_y - 1),
                (right_pot_x, right_pot_y + 1),
            )
            if position in right_reachable
        )
    left_handoff_floors = {left for left, _right in handoffs}
    right_handoff_floors = {right for _left, right in handoffs}
    assert (
        _shortest_floor_distance(
            outage_phase, onion_interaction_floors, left_handoff_floors
        )
        <= 1
    )
    assert (
        _shortest_floor_distance(
            outage_phase, right_handoff_floors, pot_interaction_floors
        )
        <= 1
    )


@pytest.mark.parametrize("variant", range(3))
def test_split_variants_keep_complementary_resources_in_separate_bays(variant):
    layout = dynamic_layouts[f"splitnosig_{variant}"]
    assert tuple(phase.steps for phase in layout.phases) == (40, 160)
    open_phase = layout.phases[0].layout.static_objects
    closed_phase = layout.phases[1].layout.static_objects
    left_start, right_start = layout.phases[0].agent_positions
    onion = StaticObject.ingredient_pile(0)
    expected_onions = np.sum(open_phase[:, :4] == onion)
    expected_pots = np.sum(open_phase[:, :4] == StaticObject.POT)
    expected_plates = np.sum(open_phase[:, 5:] == StaticObject.PLATE_PILE)
    expected_goals = np.sum(open_phase[:, 5:] == StaticObject.GOAL)
    assert min(expected_onions, expected_pots, expected_plates, expected_goals) >= 1

    differences = np.argwhere(open_phase != closed_phase)
    assert differences.shape == (1, 2)
    door_y, door_x = differences[0]
    assert door_x == 4
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
            assert np.sum(phase[:, :4] == object_type) == count
            assert np.sum(phase[:, 5:] == object_type) == 0
        for object_type, count in expected_right.items():
            assert np.sum(phase[:, :4] == object_type) == 0
            assert np.sum(phase[:, 5:] == object_type) == count

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
        ("split_no_sig", (7, 9, 3)),
        ("split_sig", (7, 9, 3)),
        ("outage_no_sig", (5, 7, 3)),
        ("outage_sig", (5, 7, 3)),
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
        ("split_no_sig", "split_sig", (4, 2)),
        ("outage_no_sig", "outage_sig", (3, 3)),
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
        assert no_sig_static[signal_y, signal_x] == StaticObject.INERT_SIGNAL_INDICATOR
        assert sig_static[signal_y, signal_x] == StaticObject.BUTTON_RECIPE_INDICATOR


@pytest.mark.parametrize(
    ("no_sig_name", "sig_name", "recipe_position"),
    (
        ("split_no_sig", "split_sig", (4, 0)),
        ("outage_no_sig", "outage_sig", (3, 0)),
    ),
)
def test_sig_pair_keeps_recipe_indicator_at_a_separate_fixed_tile(
    no_sig_name, sig_name, recipe_position
):
    recipe_x, recipe_y = recipe_position
    for layout_name in (no_sig_name, sig_name):
        for phase in dynamic_layouts[layout_name].phases:
            assert (
                phase.layout.static_objects[recipe_y, recipe_x]
                == StaticObject.RECIPE_INDICATOR
            )


def test_kitchen_split_closes_doorway_between_complementary_bays():
    layout = dynamic_layouts["split_no_sig"]
    open_phase = layout.phases[0].layout.static_objects
    split_phase = layout.phases[1].layout.static_objects
    left_start, right_start = layout.phases[0].agent_positions

    onion_pile = StaticObject.ingredient_pile(0)
    for static_objects in (open_phase, split_phase):
        assert jnp.sum(static_objects[:, :4] == onion_pile) == 1
        assert jnp.sum(static_objects[:, :4] == StaticObject.POT) == 2
        assert jnp.sum(static_objects[:, 5:] == StaticObject.POT) == 0
        assert jnp.sum(static_objects[:, :4] == StaticObject.PLATE_PILE) == 0
        assert jnp.sum(static_objects[:, 5:] == StaticObject.PLATE_PILE) == 2
        assert jnp.sum(static_objects[:, :4] == StaticObject.GOAL) == 0
        assert jnp.sum(static_objects[:, 5:] == StaticObject.GOAL) == 1

    assert open_phase[5, 4] == StaticObject.EMPTY
    assert right_start in _reachable_floor(open_phase, left_start)
    assert split_phase[5, 4] == StaticObject.WALL
    assert right_start not in _reachable_floor(split_phase, left_start)
    assert split_phase[5, 3] == StaticObject.EMPTY
    assert split_phase[5, 5] == StaticObject.EMPTY
    assert np.sum(open_phase != split_phase) == 1


@pytest.mark.parametrize("layout_name", ("outage_no_sig", "outage_sig"))
def test_resource_outage_separates_agents_but_keeps_a_shared_handoff(layout_name):
    layout = dynamic_layouts[layout_name]
    onion_pile = StaticObject.ingredient_pile(0)

    for phase in layout.phases:
        static_objects = phase.layout.static_objects
        left_start, right_start = phase.agent_positions
        assert right_start not in _reachable_floor(static_objects, left_start)
        assert static_objects[1, 3] == StaticObject.WALL
        assert static_objects[2, 3] == StaticObject.WALL
        assert static_objects[1, 2] == StaticObject.EMPTY
        assert static_objects[1, 4] == StaticObject.EMPTY

    normal_phase = layout.phases[0].layout.static_objects
    outage_phase = layout.phases[1].layout.static_objects
    assert normal_phase[0, 1] == onion_pile
    assert normal_phase[0, 5] == onion_pile
    assert outage_phase[0, 1] == onion_pile
    assert outage_phase[0, 5] == StaticObject.WALL

    expected_per_bay = {
        StaticObject.POT: 1,
        StaticObject.PLATE_PILE: 2,
        StaticObject.GOAL: 1,
    }
    for object_type, count in expected_per_bay.items():
        positions = np.argwhere(outage_phase == object_type)
        assert positions.shape == (2 * count, 2)
        assert jnp.sum(positions[:, 1] < 3) == count
        assert jnp.sum(positions[:, 1] > 3) == count


@pytest.mark.parametrize(
    ("layout_name", "change_position", "change_count"),
    (
        ("split_no_sig", (4, 5), 1),
        ("split_sig", (4, 5), 1),
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


def test_outage_left_cook_can_preload_two_onions_and_pass_one():
    env = OvercookedV3(layout="outage_no_sig", max_steps=20)
    _, state = env.reset(jax.random.PRNGKey(0))
    onion = DynamicObject.ingredient(0)
    state = state.replace(
        agents=state.agents.replace(
            pos=Position(
                x=state.agents.pos.x.at[0].set(2),
                y=state.agents.pos.y.at[0].set(1),
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
    assert state.grid[1, 3, 1] == onion

    state = state.replace(
        agents=state.agents.replace(
            pos=Position(
                x=state.agents.pos.x.at[0].set(2),
                y=state.agents.pos.y.at[0].set(2),
            ),
            dir=state.agents.dir.at[0].set(Direction.RIGHT),
            inventory=state.agents.inventory.at[0].set(onion),
        )
    )
    _, state, _, _, _ = env.step_env(jax.random.PRNGKey(2), state, actions)

    assert state.agents.inventory[0] == DynamicObject.EMPTY
    assert state.grid[1, 3, 1] == onion
    assert state.grid[2, 3, 1] == onion

    state = state.replace(
        agents=state.agents.replace(
            pos=Position(
                x=state.agents.pos.x.at[1].set(4),
                y=state.agents.pos.y.at[1].set(1),
            ),
            dir=state.agents.dir.at[1].set(Direction.LEFT),
        )
    )
    actions = {
        env.agents[0]: jnp.array(OvercookedActionsEnum.stay),
        env.agents[1]: jnp.array(OvercookedActionsEnum.interact),
    }
    _, state, _, _, _ = env.step_env(jax.random.PRNGKey(3), state, actions)

    assert state.grid[1, 3, 1] == DynamicObject.EMPTY
    assert state.grid[2, 3, 1] == onion
    assert state.agents.inventory[1] == onion


def test_split_runtime_closes_handoff_wall_at_step_40():
    env = OvercookedV3(layout="split_no_sig", max_steps=220)
    _, state = env.reset(jax.random.PRNGKey(0))
    state = state.replace(step=jnp.array(39))
    actions = {agent: jnp.array(OvercookedActionsEnum.stay) for agent in env.agents}

    _, state, _, _, infos = jax.jit(env.step_env)(jax.random.PRNGKey(1), state, actions)

    assert state.step.item() == 40
    assert state.layout_index.item() == 1
    assert state.grid[5, 4, 0].item() == StaticObject.WALL
    assert jnp.all(infos["left_workload_tile_count"] == 2)
    assert jnp.all(infos["right_workload_tile_count"] == 3)
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
    assert state.grid[0, 1, 0].item() == StaticObject.ingredient_pile(0)
    assert state.grid[0, 5, 0].item() == StaticObject.WALL
    assert jnp.all(infos["left_workload_tile_count"] == 4)
    assert jnp.all(infos["right_workload_tile_count"] == 4)
    assert jnp.all(infos["left_ingredient_pile_count"] == 1)
    assert jnp.all(infos["right_ingredient_pile_count"] == 0)
    assert jnp.all(infos["layout_changed"])


@pytest.mark.parametrize(
    ("layout_name", "signal_position", "standing_position"),
    (
        ("split_sig", (4, 2), (3, 2)),
        ("outage_sig", (3, 3), (2, 3)),
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
    assert INDICATOR_ACTIVATION_COST == 0.0
    assert rewards[env.agents[0]].item() == 0.0
    assert rewards[env.agents[1]].item() == 0.0

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
    signal_x, signal_y = 4, 2
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


def test_no_sig_blank_signal_cell_does_not_activate_or_store_objects():
    env = OvercookedV3(layout="split_no_sig", max_steps=20)
    _, state = env.reset(jax.random.PRNGKey(0))
    state = state.replace(
        agents=state.agents.replace(
            pos=Position(
                x=state.agents.pos.x.at[0].set(3),
                y=state.agents.pos.y.at[0].set(2),
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

    assert state.grid[2, 4, 0].item() == StaticObject.INERT_SIGNAL_INDICATOR
    assert state.grid[2, 4, 1].item() == DynamicObject.EMPTY
    assert state.grid[2, 4, 2].item() == 0
    assert state.agents.inventory[0] == DynamicObject.ingredient(0)
    assert rewards[env.agents[0]].item() == 0
