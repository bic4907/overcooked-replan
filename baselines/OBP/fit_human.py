"""Fit the planner's three dials to recorded people, the way the paper fits H1.

The paper's simulated humans are one planner with three dials (see
``planner.py``): H0 only waits, H1 is the maximum-likelihood setting of all
three on human play, H2 waits as much as it takes to score like people. This
script computes the two data-driven numbers for one kitchen:

* the proportion of recorded steps in which a person did nothing (H0's
  ``prob_wait``), and
* the cross-entropy of every recorded human action under a grid of
  (hltemp, lltemp, prob_wait), whose minimum is H1.

Each recorded state is scored under the planner's own reading of the kitchen;
the carry (which half a seat serves, what it handed over) is rolled along the
recorded trajectory with the planner's ``actions`` so the roles it assumes are
the ones it would assume playing that game itself.

Usage::

    python -m baselines.OBP.fit_human --layout split_0
"""

from __future__ import annotations

import argparse
import functools
import itertools
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import jaxmarl
from baselines.OBP.data import _read_archive, _state_from_arrays, find_episodes
from baselines.OBP.planner import GreedyPlanner
from jaxmarl.environments.overcooked_v3.common import OvercookedActionsEnum


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", default="data/human")
    parser.add_argument("--layout", default="split_0")
    parser.add_argument("--output", default=None, help="Default outputs/human_fit/<layout>.json")
    parser.add_argument("--hltemp", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.3, 0.45, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0])
    parser.add_argument("--lltemp", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.286, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0])
    parser.add_argument("--prob-wait", type=float, nargs="+", default=[0.0, 0.05, 0.072, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
    parser.add_argument("--max-episodes", type=int, default=None)
    return parser.parse_args()


def episode_features(planner, env, arrays):
    """Per recorded step, the dial-free features of every seat and the actions."""
    states = _state_from_arrays(arrays)
    actions = np.asarray(arrays["actions"])
    human = np.asarray(arrays["human_mask"]).astype(bool)
    steps = actions.shape[0]
    features_fn = jax.jit(planner.seat_features)
    act = jax.jit(planner.actions)
    state_at = lambda t: jax.tree_util.tree_map(lambda x: x[t], states)  # noqa: E731
    key = jax.random.PRNGKey(0)
    carry = planner.initial_carry(state_at(0))
    rows = []
    for t in range(steps):
        state = state_at(t)
        feats = features_fn(carry, state)
        for seat in range(env.num_agents):
            if human[t, seat]:
                rows.append((jax.tree_util.tree_map(np.asarray, feats[seat]), int(actions[t, seat])))
        key, sub = jax.random.split(key)
        carry, _ = act(carry, state, sub)
    return rows


def stack_rows(rows):
    return (
        {k: jnp.asarray(np.stack([r[0][k] for r in rows])) for k in rows[0][0]},
        jnp.asarray([r[1] for r in rows]),
    )


def main():
    args = parse_args()
    paths = find_episodes(args.data, [args.layout])
    if args.max_episodes:
        paths = paths[: args.max_episodes]
    if not paths:
        raise SystemExit(f"No demonstrations for {args.layout} under {args.data}")

    metadata, _ = _read_archive(paths[0])
    env_kwargs = dict(metadata["env_kwargs"])
    env_kwargs.pop("layout")
    env_kwargs.setdefault("transition_warning_steps", 20)
    env = jaxmarl.make("overcooked_v3", layout=args.layout, **env_kwargs)
    planner = GreedyPlanner(env)

    rows = []
    for path in paths:
        _, arrays = _read_archive(path)
        rows.extend(episode_features(planner, env, arrays))
        print(f"{path.name}: {len(rows)} rows so far", flush=True)
    features, actions = stack_rows(rows)
    stay_share = float(jnp.mean(actions == OvercookedActionsEnum.stay))
    print(f"{len(rows)} human actions, stay share {stay_share:.3f}")

    @functools.partial(jax.jit, static_argnums=(0, 1))
    def cross_entropy(hltemp, lltemp, prob_wait):
        probs = jax.vmap(
            lambda f: GreedyPlanner.probs_from_features(f, hltemp, lltemp, prob_wait)
        )(features)
        picked = probs[jnp.arange(actions.shape[0]), actions]
        return -jnp.mean(jnp.log(jnp.clip(picked, 1e-6)))

    # The zero-temperature branches are Python-level, so the temperatures are
    # static and the jit is traced once per temperature pair.
    results = []
    for hl, ll, pw in itertools.product(args.hltemp, args.lltemp, args.prob_wait):
        ce = float(cross_entropy(hl, ll, pw))
        results.append(dict(hltemp=hl, lltemp=ll, prob_wait=pw, cross_entropy=ce))
    results.sort(key=lambda r: r["cross_entropy"])
    best = results[0]
    print("best:", best)
    print("top 10:")
    for r in results[:10]:
        print(f"  hltemp={r['hltemp']:<5} lltemp={r['lltemp']:<6} prob_wait={r['prob_wait']:<5} CE={r['cross_entropy']:.4f}")

    output = Path(args.output or f"outputs/human_fit/{args.layout}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            dict(
                layout=args.layout,
                episodes=len(paths),
                rows=len(rows),
                stay_share=stay_share,
                H0=dict(prob_wait=stay_share, lltemp=0.0, hltemp=0.0),
                H1=dict(prob_wait=best["prob_wait"], lltemp=best["lltemp"], hltemp=best["hltemp"]),
                grid=results,
            ),
            indent=1,
        )
    )
    print("saved", output)


if __name__ == "__main__":
    main()
