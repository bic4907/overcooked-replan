"""Coverage for the V3 ``layout_mode`` conditions and the silent arms.

``cyclic`` walks the authored phase schedule inside one episode; the multimap
condition, ``episode_random``, draws one kitchen at reset and holds it, so a
run under it never crosses a boundary.

All three arms of the 75-step experiment share one observation width. The
countdown and change-mask channels stay in the observation and are held at
zero -- by the observer gate for cdoff, and by there being no boundary ahead
for multimap -- so the arms differ in what an agent can read, never in the
shape of its input.
"""

from collections import Counter

import jax
import jax.numpy as jnp
import pytest

from jaxmarl.environments.overcooked_v3.dynamic_overcooked import OvercookedV3

SCENARIOS = (
    "split_0",
    "split_1",
    "outage_0",
    "outage_1",
    "distance_0",
    "distance_1",
)


def _env(layout, **kwargs):
    options = {
        "include_transition_countdown": False,
        "include_layout_change_mask": False,
    }
    options.update(kwargs)
    return OvercookedV3(layout=layout, max_steps=450, **options)


def test_layout_mode_rejects_unknown_values():
    with pytest.raises(ValueError, match="layout_mode"):
        _env("split_0", layout_mode="random")


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


@pytest.mark.parametrize("layout_name", SCENARIOS)
def test_episode_random_installs_the_drawn_kitchen(layout_name):
    """The grid, the spawns and the recipe all follow the drawn phase."""
    env = _env(layout_name, layout_mode="episode_random")
    _, states = jax.vmap(env.reset)(jax.random.split(jax.random.PRNGKey(2), 32))

    assert jnp.array_equal(
        states.grid[:, :, :, 0], env.phase_static_objects[states.layout_index]
    )
    spawns = env.phase_agent_positions[states.layout_index]
    assert jnp.array_equal(states.agents.pos.x, spawns[:, :, 0])
    assert jnp.array_equal(states.agents.pos.y, spawns[:, :, 1])
    expected_recipe = jnp.where(
        env.phase_has_recipe[states.layout_index],
        env.phase_recipes[states.layout_index],
        states.recipe,
    )
    assert jnp.array_equal(states.recipe, expected_recipe)


@pytest.mark.parametrize("layout_name", SCENARIOS)
def test_episode_random_silences_every_transition_signal(layout_name):
    """No boundary ahead means no countdown and no change mask to read."""
    env = _env(
        layout_name,
        layout_mode="episode_random",
        include_transition_countdown=True,
        include_layout_change_mask=True,
    )
    for step in (0, 60, 74, 149, 300, 449):
        assert not bool(env.get_transition_warning_active(jnp.array(step)))
        assert float(env.get_transition_countdown(jnp.array(step))) == 0.0
        assert not bool(jnp.any(env.get_layout_change_mask(jnp.array(step))))


@pytest.mark.parametrize("layout_name", SCENARIOS)
def test_cyclic_is_unchanged_by_the_port(layout_name):
    """The default condition still switches on the authored 75-step clock."""
    env = _env(layout_name)
    actions = {agent: jnp.array(4, dtype=jnp.int32) for agent in env.agents}
    step = jax.jit(env.step_env)
    _, state = env.reset(jax.random.PRNGKey(0))

    boundaries = []
    for _ in range(450):
        _, state, _, done, info = step(jax.random.PRNGKey(0), state, actions)
        if bool(jnp.all(info["layout_changed"])):
            boundaries.append(int(state.step))
        if bool(done["__all__"]):
            break
    assert boundaries[:2] == [75, 150]


@pytest.mark.parametrize("layout_name", SCENARIOS)
def test_the_three_arms_share_one_observation_width(layout_name):
    """cdoff and multimap keep the channels the baseline reads, zero-filled."""
    baseline = OvercookedV3(
        layout=layout_name,
        max_steps=450,
        include_transition_countdown=True,
        include_layout_change_mask=True,
        transition_observer="both",
    )
    cdoff = OvercookedV3(
        layout=layout_name,
        max_steps=450,
        include_transition_countdown=True,
        include_layout_change_mask=True,
        transition_observer="none",
    )
    multimap = OvercookedV3(
        layout=layout_name,
        max_steps=450,
        include_transition_countdown=True,
        include_layout_change_mask=True,
        transition_observer="both",
        layout_mode="episode_random",
    )

    width = baseline.observation_space().shape[-1]
    assert cdoff.observation_space().shape[-1] == width
    assert multimap.observation_space().shape[-1] == width

    def transition_channels(env, step):
        _, state = env.reset(jax.random.PRNGKey(0))
        obs = env.get_obs(state.replace(step=jnp.array(step)))
        return jnp.stack([obs[agent] for agent in env.agents])[..., -2:]

    # The boundary sits at 75 and the warning window is the 20 steps before it.
    for step in (0, 40, 56, 70, 74, 149, 300):
        assert not jnp.any(transition_channels(cdoff, step))
        assert not jnp.any(transition_channels(multimap, step))
    assert jnp.any(transition_channels(baseline, 70))
