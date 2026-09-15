"""Score the planner in self-play, one mean per kitchen.

Usage::

    python -m baselines.planner.eval_planner --layouts split_0 split_wide \\
        --prob-wait 0.395 --seeds 10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import numpy as np


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layouts", nargs="+", default=["split_0", "split_wide"])
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=450)
    parser.add_argument("--prob-wait", type=float, default=0.0)
    parser.add_argument("--lltemp", type=float, default=0.0)
    parser.add_argument("--hltemp", type=float, default=0.0)
    parser.add_argument("--output", default=None, help="Append one JSON line here.")
    parser.add_argument("--name", default=None, help="Label written with --output.")
    return parser.parse_args(argv)


def play(env, planner, seed, max_steps):
    act = jax.jit(planner.actions)
    step = jax.jit(env.step_env)
    key = jax.random.PRNGKey(seed)
    _, state = env.reset(key)
    carry = planner.initial_carry(state)
    total = 0.0
    for _ in range(max_steps):
        key, plan_key, step_key = jax.random.split(key, 3)
        carry, actions = act(carry, state, plan_key)
        _, state, reward, dones, _ = step(
            step_key, state, {agent: actions[i] for i, agent in enumerate(env.agents)}
        )
        total += float(reward[env.agents[0]])
        if bool(dones["__all__"]):
            break
    return total


def main(argv=None):
    args = parse_args(argv)
    import jaxmarl
    from baselines.planner.planner import GreedyPlanner

    means = {}
    for layout in args.layouts:
        env = jaxmarl.make(
            "overcooked_v3",
            layout=layout,
            max_steps=args.max_steps,
            random_agent_positions=False,
            include_transition_countdown=True,
            include_layout_change_mask=True,
            transition_observer="both",
            transition_warning_steps=20,
        )
        planner = GreedyPlanner(
            env, prob_wait=args.prob_wait, lltemp=args.lltemp, hltemp=args.hltemp
        )
        scores = [play(env, planner, args.seed + i, args.max_steps) for i in range(args.seeds)]
        means[layout] = float(np.mean(scores))
        print(f"  {layout}: mean {means[layout]:.1f}  std {np.std(scores):.1f}  {scores}", flush=True)
    overall = float(np.mean(list(means.values())))
    print(f"  mean over {len(means)} maps: {overall:.1f}")
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as handle:
            handle.write(
                json.dumps(
                    dict(
                        name=args.name,
                        prob_wait=args.prob_wait,
                        lltemp=args.lltemp,
                        hltemp=args.hltemp,
                        seeds=args.seeds,
                        maps=means,
                        mean=overall,
                    )
                )
                + "\n"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
