"""Render a random-policy rollout in one Overcooked V3 role scenario."""

import argparse
from pathlib import Path

import jax

from jaxmarl import make

ROLE_SCENARIOS = (
    "split_no_sig",
    "split_sig",
    "outage_no_sig",
    "outage_sig",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout", choices=ROLE_SCENARIOS, required=True)
    parser.add_argument("--steps", type=int, default=220)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gif", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.steps < 1:
        raise ValueError("--steps must be at least 1")

    env = make(
        "overcooked_v3",
        layout=args.layout,
        # Keep the final requested transition from triggering the base API's
        # automatic reset before we print/save the last state.
        max_steps=args.steps + 1,
        random_agent_positions=False,
    )
    key = jax.random.PRNGKey(args.seed)
    key, reset_key = jax.random.split(key)
    _, state = env.reset(reset_key)

    episode_return = 0.0
    states = [state] if args.gif else None
    captions = ["step=0 phase=0 return=0"] if args.gif else None
    print(f"layout={args.layout} phase=0 step=0")

    for step in range(args.steps):
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
            transition_warning_steps=env.transition_warning_steps
        ).animate(
            states,
            filename=str(args.gif),
            agent_view_size=env.agent_view_size,
            captions=captions,
        )
        print(f"gif={args.gif}")


if __name__ == "__main__":
    main()
