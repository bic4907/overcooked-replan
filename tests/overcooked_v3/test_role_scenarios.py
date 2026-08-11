import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxmarl.environments.overcooked_v3.common import (
    Direction,
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


@pytest.mark.parametrize("layout_name", ROLE_SCENARIOS)
def test_role_scenario_is_registered_and_resettable(layout_name):
    env = OvercookedV3(layout=layout_name, max_steps=220)
    obs, state = env.reset(jax.random.PRNGKey(0))

    assert state.grid.shape == (7, 11, 3)
    assert obs["agent_0"].shape == (7, 11, 32)
    assert state.layout_index.item() == 0


@pytest.mark.parametrize(
    ("no_sig_name", "sig_name", "signal_position"),
    (
        ("split_no_sig", "split_sig", (5, 2)),
        ("outage_no_sig", "outage_sig", (5, 3)),
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
        assert no_sig_static[signal_y, signal_x] == StaticObject.WALL
        assert sig_static[signal_y, signal_x] == StaticObject.BUTTON_RECIPE_INDICATOR


def test_kitchen_split_closes_center_and_distributes_resources():
    layout = dynamic_layouts["split_no_sig"]
    choose_phase = layout.phases[0].layout.static_objects
    split_phase = layout.phases[1].layout.static_objects

    assert choose_phase[3, 5] == StaticObject.EMPTY
    assert split_phase[3, 5] == StaticObject.WALL
    assert choose_phase[1, 1] == StaticObject.ingredient_pile(0)
    assert choose_phase[5, 1] == StaticObject.POT
    assert choose_phase[1, 8] == StaticObject.PLATE_PILE
    assert choose_phase[5, 8] == StaticObject.GOAL


@pytest.mark.parametrize(
    ("layout_name", "change_position"),
    (
        ("split_no_sig", (5, 3)),
        ("split_sig", (5, 3)),
        ("outage_no_sig", (9, 1)),
        ("outage_sig", (9, 1)),
    ),
)
def test_change_mask_marks_only_the_next_static_tile_change(
    layout_name, change_position
):
    env = OvercookedV3(layout=layout_name, max_steps=220)
    obs, state = env.reset(jax.random.PRNGKey(0))
    change_x, change_y = change_position
    change_mask = obs["agent_0"][..., -1]

    assert state.layout_change_mask[change_y, change_x]
    assert change_mask[change_y, change_x] == 1.0
    assert jnp.sum(change_mask) == 1


def test_visualizer_formats_transition_countdown_in_seconds():
    env = OvercookedV3(layout="split_no_sig", max_steps=220)
    _, state = env.reset(jax.random.PRNGKey(0))
    visualizer = OvercookedV3Visualizer(seconds_per_step=0.2)

    assert (
        visualizer.caption_with_countdown(state, "step=0")
        == "step=0 | next layout change in 8.0s"
    )


@pytest.mark.parametrize("layout_name", ("outage_no_sig", "outage_sig"))
def test_resource_outage_removes_right_onion_station(layout_name):
    layout = dynamic_layouts[layout_name]
    normal_phase = layout.phases[0].layout.static_objects
    outage_phase = layout.phases[1].layout.static_objects
    onion_pile = StaticObject.ingredient_pile(0)

    assert np.count_nonzero(normal_phase == onion_pile) == 2
    assert np.count_nonzero(outage_phase == onion_pile) == 1
    assert normal_phase[1, 9] == onion_pile
    assert outage_phase[1, 9] == StaticObject.WALL


def test_split_runtime_enters_closed_phase_at_step_40():
    env = OvercookedV3(layout="split_no_sig", max_steps=220)
    _, state = env.reset(jax.random.PRNGKey(0))
    state = state.replace(step=jnp.array(39))
    actions = {agent: jnp.array(OvercookedActionsEnum.stay) for agent in env.agents}

    _, state, _, _, infos = jax.jit(env.step_env)(jax.random.PRNGKey(1), state, actions)

    assert state.step.item() == 40
    assert state.layout_index.item() == 1
    assert state.grid[3, 5, 0].item() == StaticObject.WALL
    assert jnp.all(infos["layout_changed"])


@pytest.mark.parametrize(
    ("layout_name", "signal_position", "standing_position"),
    (
        ("split_sig", (5, 2), (4, 2)),
        ("outage_sig", (5, 3), (4, 3)),
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

    _, state, rewards, _, _ = jax.jit(env.step_env)(
        jax.random.PRNGKey(1), state, actions
    )

    assert state.grid[signal_y, signal_x, 2].item() > 0
    assert rewards[env.agents[0]].item() == -INDICATOR_ACTIVATION_COST


def test_no_sig_counter_does_not_activate():
    env = OvercookedV3(layout="split_no_sig", max_steps=20)
    _, state = env.reset(jax.random.PRNGKey(0))
    state = state.replace(
        agents=state.agents.replace(
            pos=Position(
                x=state.agents.pos.x.at[0].set(4),
                y=state.agents.pos.y.at[0].set(2),
            ),
            dir=state.agents.dir.at[0].set(Direction.RIGHT),
        )
    )
    actions = {
        env.agents[0]: jnp.array(OvercookedActionsEnum.interact),
        env.agents[1]: jnp.array(OvercookedActionsEnum.stay),
    }

    _, state, rewards, _, _ = jax.jit(env.step_env)(
        jax.random.PRNGKey(1), state, actions
    )

    assert state.grid[2, 5, 2].item() == 0
    assert rewards[env.agents[0]].item() == 0
