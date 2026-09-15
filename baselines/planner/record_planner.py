"""Watch the planner cook, one episode at a time.

Numbers say the planner is slower than the policy it is meant to stand in for;
they do not say why. A run at five frames a second does, and the failures this
planner has had so far -- an onion put down a few steps from the pot, two cooks
pushing at each other in a corridor, a soup that never crosses -- all look like
something rather than like a lower mean.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout", default="split_0")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=450)
    parser.add_argument("--prob-wait", type=float, default=0.0)
    parser.add_argument("--lltemp", type=float, default=0.0)
    parser.add_argument("--hltemp", type=float, default=0.0)
    parser.add_argument(
        "--region-only",
        action="store_true",
        help="Keep each cook in its own half even while the door is open.",
    )
    parser.add_argument("--share-mode", default="continuous", choices=("two", "continuous"))
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Play this many seeds from --seed and record the one that scores "
        "closest to their mean, so a noisy human model is shown at its typical.",
    )
    parser.add_argument(
        "--pot-specialist",
        type=int,
        default=2,
        help="Pots one seat must hold before the other stops loading them.",
    )
    parser.add_argument(
        "--seat",
        type=int,
        default=None,
        help="Plan for this seat only; the other is played by the same planner.",
    )
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--tile-size", type=int, default=32)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    import jaxmarl
    import imageio.v2 as imageio

    from baselines.planner.planner import GreedyPlanner
    from jaxmarl.viz.overcooked_v3_visualizer import OvercookedV3Visualizer

    env = jaxmarl.make(
        "overcooked_v3",
        layout=args.layout,
        max_steps=args.max_steps,
        random_agent_positions=False,
        include_transition_countdown=True,
        include_layout_change_mask=True,
        transition_observer="both",
        transition_warning_steps=20,
    )
    planner = GreedyPlanner(
        env,
        prob_wait=args.prob_wait,
        lltemp=args.lltemp,
        hltemp=args.hltemp,
        free_movement=False if args.region_only else None,
        pot_specialist=args.pot_specialist,
        share_mode=args.share_mode,
    )
    act = jax.jit(planner.actions)
    step = jax.jit(env.step_env)

    def play(seed):
        key = jax.random.PRNGKey(seed)
        _, state = env.reset(key)
        carry = planner.initial_carry(state)
        states, captions = [state], ["step 0  score 0"]
        total = 0.0
        for number in range(args.max_steps):
            key, plan_key, step_key = jax.random.split(key, 3)
            carry, actions = act(carry, state, plan_key)
            _, state, reward, dones, _ = step(
                step_key, state, {agent: actions[i] for i, agent in enumerate(env.agents)}
            )
            total += float(reward[env.agents[0]])
            states.append(state)
            captions.append(f"step {number + 1}  score {total:g}")
            if bool(dones["__all__"]):
                break
        return total, states, captions

    played = [play(args.seed + offset) for offset in range(args.episodes)]
    scores = np.array([score for score, _, _ in played])
    mean = float(scores.mean())
    shown = int(np.argmin(np.abs(scores - mean)))
    total, states, captions = played[shown]

    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    visualizer = OvercookedV3Visualizer(
        tile_size=args.tile_size,
        seconds_per_step=1.0 / args.fps,
        transition_warning_steps=env.transition_warning_steps,
    )
    frames = visualizer._animation_frames(states, captions=captions)
    imageio.mimsave(
        str(path),
        frames,
        format="GIF",
        duration=1.0 / args.fps,
        loop=0,
        # Without this a GIF keeps whatever the frame before it drew, so the
        # countdown text smears across the frames that follow.
        disposal=2,
    )
    print(
        f"  {args.layout} seed{args.seed + shown}: score {total:g} "
        f"(mean of {args.episodes}: {mean:.1f}) -> {path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
