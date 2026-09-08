"""Render a random-policy rollout in one Overcooked V3 role scenario."""

import argparse
from pathlib import Path

import jax

from jaxmarl import make
from jaxmarl.environments.overcooked_v3 import ALL_ROLE_SCENARIO_LAYOUT_NAMES


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--layout", choices=ALL_ROLE_SCENARIO_LAYOUT_NAMES, required=True
    )
    parser.add_argument(
        "--steps",
        type=int,
        help="rollout length (defaults to 450 for hard layouts, 220 otherwise)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gif", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    hard_mode = "_hard_" in args.layout
    steps = args.steps if args.steps is not None else (450 if hard_mode else 220)
    if steps < 1:
        raise ValueError("--steps must be at least 1")

    env = make(
        "overcooked_v3",
        layout=args.layout,
        # Keep the final requested transition from triggering the base API's
        # automatic reset before we print/save the last state.
        max_steps=steps + 1,
        random_agent_positions=False,
        agent_view_size=4 if hard_mode else None,
        distinguish_blockers=hard_mode,
    )
    key = jax.random.PRNGKey(args.seed)
    key, reset_key = jax.random.split(key)
    _, state = env.reset(reset_key)

    episode_return = 0.0
    states = [state] if args.gif else None
    captions = ["step=0 phase=0 return=0"] if args.gif else None
    print(f"layout={args.layout} phase=0 step=0")

    for step in range(steps):
        key, action_key, step_key = jax.random.split(key, 3)
        action_keys = jax.random.split(action_key, env.num_agents)
        actions = {
            agent: env.action_space(agent).sample(action_keys[index])
            for index, agent in enumerate(env.agents)
        }
        _, state, rewards, dones, infos = env.step(step_key, state, actions)
        episode_return += float(rewards[env.agents[0]])
        phase = int(state.layout_index)

        if bool(infos["layout_changed"][0]):
            print(f"layout={args.layout} phase={phase} step={step + 1}")

        if states is not None and captions is not None:
            states.append(state)
            captions.append(f"step={step + 1} phase={phase} return={episode_return:g}")

        if bool(dones["__all__"]):
            break

    print(f"episode_return={episode_return:g} final_step={int(state.step)}")

    if args.gif is not None:
        from jaxmarl.viz.overcooked_v3_visualizer import OvercookedV3Visualizer

        args.gif.parent.mkdir(parents=True, exist_ok=True)
        OvercookedV3Visualizer(
            transition_warning_steps=env.transition_warning_steps,
            distinguish_blockers=env.distinguish_blockers,
        ).animate(
            states,
            filename=str(args.gif),
            agent_view_size=env.agent_view_size,
            captions=captions,
        )
        print(f"gif={args.gif}")


if __name__ == "__main__":
    main()
