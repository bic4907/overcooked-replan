"""The heuristic human proxy, and the dials that turn it into a person."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yaml

import jaxmarl
from baselines.planner.planner import GreedyPlanner
from jaxmarl.environments.overcooked_v3.common import OvercookedActionsEnum

ROOT = Path(__file__).resolve().parents[2]


def _sweep(name):
    return yaml.safe_load((ROOT / "experiment/planner" / name).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def env():
    return jaxmarl.make(
        "overcooked_v3",
        layout="split_0",
        max_steps=450,
        random_agent_positions=False,
        include_transition_countdown=True,
        include_layout_change_mask=True,
        transition_observer="both",
        transition_warning_steps=20,
    )


def _states(env, steps):
    planner = GreedyPlanner(env)
    act = jax.jit(planner.actions)
    step = jax.jit(env.step_env)
    key = jax.random.PRNGKey(0)
    _, state = env.reset(key)
    carry = planner.initial_carry(state)
    seen = [(carry, state)]
    for _ in range(steps):
        key, plan_key, step_key = jax.random.split(key, 3)
        carry, actions = act(carry, state, plan_key)
        _, state, _, _, _ = step(
            step_key, state, {agent: actions[i] for i, agent in enumerate(env.agents)}
        )
        seen.append((carry, state))
    return planner, seen


def test_planner_cooks(env):
    """The noise-free planner delivers soup rather than idling."""
    planner = GreedyPlanner(env)
    act = jax.jit(planner.actions)
    step = jax.jit(env.step_env)
    key = jax.random.PRNGKey(0)
    _, state = env.reset(key)
    carry = planner.initial_carry(state)
    total = 0.0
    for _ in range(450):
        key, plan_key, step_key = jax.random.split(key, 3)
        carry, actions = act(carry, state, plan_key)
        _, state, reward, _, _ = step(
            step_key, state, {agent: actions[i] for i, agent in enumerate(env.agents)}
        )
        total += float(reward[env.agents[0]])
    assert total > 100.0


def test_action_probs_are_distributions(env):
    """Every dial setting gives each seat a proper distribution over actions."""
    _, seen = _states(env, 20)
    for dials in (
        dict(prob_wait=0.0, lltemp=0.0, hltemp=0.0),
        dict(prob_wait=0.45, lltemp=0.286, hltemp=0.072),
        dict(prob_wait=0.4, lltemp=0.8, hltemp=2.0),
    ):
        probs_fn = jax.jit(GreedyPlanner(env, **dials).action_probs)
        for carry, state in seen:
            probs = np.asarray(probs_fn(carry, state))
            assert probs.shape == (env.num_agents, 6)
            assert np.all(np.isfinite(probs))
            assert np.all(probs >= 0.0)
            np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)


def test_probs_agree_with_the_greedy_action(env):
    """With the dials at zero the likeliest action is the one it would play."""
    planner, seen = _states(env, 20)
    probs_fn = jax.jit(planner.action_probs)
    act = jax.jit(planner.actions)
    agreed = 0
    for carry, state in seen:
        probs = np.asarray(probs_fn(carry, state))
        _, actions = act(carry, state, jax.random.PRNGKey(0))
        agreed += int(np.sum(np.argmax(probs, axis=1) == np.asarray(actions)))
    # The unstuck rule and stepping aside are deliberately not modelled, so the
    # steps where one cook is blocked disagree; the plan itself must not. On the
    # benchmark kitchens that is nothing at all away from a doorway -- outage_0
    # and distance_0 agree on every step of a hundred and fifty -- and about one
    # step in ten on split_0, where the two of them share one corridor.
    assert agreed >= int(0.85 * len(seen) * env.num_agents)


def test_waiting_shows_up_in_the_distribution(env):
    """prob_wait is mixed in on top of whatever the planner would do."""
    _, seen = _states(env, 5)
    carry, state = seen[0]
    plain = np.asarray(jax.jit(GreedyPlanner(env).action_probs)(carry, state))
    waiting = np.asarray(
        jax.jit(GreedyPlanner(env, prob_wait=0.3).action_probs)(carry, state)
    )
    stay = int(OvercookedActionsEnum.stay)
    np.testing.assert_allclose(waiting[:, stay], 0.3 + 0.7 * plain[:, stay], atol=1e-5)


def test_matrix_sweep_covers_every_pairing():
    """The sweep plays each simulated human against every agent we have."""
    sweep = _sweep("human_matrix.yaml")
    parameters = sweep["parameters"]
    assert parameters["human"]["values"] == ["br", "h0", "h1", "h2"]
    assert parameters["partner"]["values"] == [
        "br",
        "h0",
        "h1",
        "h2",
        "cnn",
        "rnn",
        "fcp",
    ]
    assert parameters["seeds"]["value"] == 6
    assert len(parameters["layout"]["values"]) == 10
    assert sweep["program"] == "baselines/planner/eval_planner_partner.py"
