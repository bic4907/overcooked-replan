"""The heuristic human proxy, and the dials that turn it into a person."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yaml

import jaxmarl
from baselines.OBP.planner import GreedyPlanner
from jaxmarl.environments.overcooked_v3.common import OvercookedActionsEnum

ROOT = Path(__file__).resolve().parents[2]


def _sweep(name):
    return yaml.safe_load((ROOT / "experiment/obp" / name).read_text(encoding="utf-8"))


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
