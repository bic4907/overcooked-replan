"""Evaluate BC self-play across independent action-sampling seeds."""

import argparse
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from baselines.BC.policy import load_policy
from jaxmarl.environments.overcooked_v3 import OvercookedV3


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("artifacts/bc/split_0_v1"))
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.episodes < 1 or args.seed < 0:
        parser.error("episodes must be positive and seed nonnegative")
    if args.output.exists():
        parser.error("Choose a new output file")
    logits_fn, config = load_policy(args.model)
    env = OvercookedV3(**config["env_kwargs"])

    @jax.jit
    def episode(seed):
        key, reset_key = jax.random.split(jax.random.PRNGKey(seed))
        obs, state = env.reset(reset_key)

        def advance(carry, unused):
            key, obs, state, finished = carry
            key, action_key, step_key = jax.random.split(key, 3)
            logits = logits_fn(jnp.stack([obs[a] for a in env.agents]))
            sampled = jax.random.categorical(action_key, logits)
            actions = {a: sampled[i] for i, a in enumerate(env.agents)}
            next_obs, next_state, rewards, dones, _ = env.step_env(
                step_key, state, actions
            )
            # Rewards are shared: count one agent's reward, not their sum.
            reward = jnp.where(finished, 0.0, rewards[env.agents[0]])
            valid = ~finished
            finished = finished | dones["__all__"]
            return (key, next_obs, next_state, finished), (reward, valid)

        _, (rewards, valid) = jax.lax.scan(
            advance,
            (key, obs, state, jnp.asarray(False)),
            None,
            length=config["env_kwargs"]["max_steps"],
        )
        return rewards, valid

    results = []
    for seed in range(args.seed, args.seed + args.episodes):
        rewards, valid = episode(seed)
        rewards, valid = np.asarray(rewards), np.asarray(valid)
        result = {
            "seed": seed,
            "team_return": float(rewards.sum()),
            "steps": int(valid.sum()),
            "per_150_steps_return": [
                float(r.sum()) for r in np.array_split(rewards, 3)
            ],
        }
        results.append(result)
        print(
            f"{len(results)}/{args.episodes} seed={seed} score={result['team_return']:g}",
            flush=True,
        )
    scores = np.asarray([r["team_return"] for r in results])
    unique, counts = np.unique(scores, return_counts=True)
    report = {
        "model": str(args.model),
        "policy_sha256": hashlib.sha256(
            (args.model / "policy.msgpack").read_bytes()
        ).hexdigest(),
        "env_kwargs": config["env_kwargs"],
        "action_selection": "categorical sampling; same checkpoint for both agents",
        "episodes": args.episodes,
        "mean": float(scores.mean()),
        "sample_std": float(scores.std(ddof=1)) if len(scores) > 1 else None,
        "median": float(np.median(scores)),
        "min": float(scores.min()),
        "max": float(scores.max()),
        "zero_score_episodes": int((scores == 0).sum()),
        "score_counts": {str(float(k)): int(v) for k, v in zip(unique, counts)},
        "mean_per_150_steps_return": np.mean(
            [r["per_150_steps_return"] for r in results], axis=0
        ).tolist(),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps({k: v for k, v in report.items() if k != "results"}, indent=2),
        flush=True,
    )


if __name__ == "__main__":
    main()
