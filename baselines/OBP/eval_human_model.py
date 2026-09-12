"""Play a human model and report what it scores.

Fitting held-out actions says the model predicts people; playing says it can
still cook. A model that has collapsed onto the majority action scores a
respectable accuracy and zero reward, which is exactly the failure the optimal
behavior prior is meant to remove, so both numbers are reported side by side.

Both seats are the model itself. The demonstrations were two people in one
kitchen, so the comparison the reward is read against -- the mean return those
people achieved -- is a pair, not a soloist.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model", help="Directory written by train_bc.py.")
    source.add_argument(
        "--prior-checkpoint",
        help="Raw self-play checkpoint, to score the prior with no cloning.",
    )
    source.add_argument(
        "--model-root",
        help=(
            "Compose the model directory from --arm and --seed, as "
            "<root>/<arm>_seed<seed>. A grid sweep varies those two and cannot "
            "build a path out of them itself."
        ),
    )
    source.add_argument(
        "--prior-root",
        help="Compose the prior checkpoint from --seed, the way train_bc does.",
    )
    parser.add_argument("--prior-architecture", default="cnn", choices=("cnn", "rnn"))
    parser.add_argument("--prior-layout", default="obp10")
    parser.add_argument(
        "--prior-env-name",
        default="overcooked_v3_multilayout",
        help="Which environment the prior was trained in; part of its filename.",
    )
    parser.add_argument("--arm", default=None, help="Label for reporting.")
    partner = parser.add_mutually_exclusive_group()
    partner.add_argument(
        "--partner-model",
        help=(
            "Directory written by train_bc.py to seat opposite the model. "
            "Without one, both seats are the model itself."
        ),
    )
    partner.add_argument(
        "--partner-arm",
        choices=("bc", "obp", "obp-frozen", "prior"),
        help=(
            "Seat a collaborative AI opposite the model, composing its "
            "checkpoint from the arm and --seed: one of the three best "
            "responses, or the self-play prior, which is its own best "
            "response and so is the paper's SP baseline."
        ),
    )
    partner.add_argument(
        "--partner-checkpoint",
        help=(
            "Raw IPPO or FCP checkpoint to seat opposite the model. It has to "
            "have been trained on the same canvas and observation config; the "
            "architecture may differ."
        ),
    )
    parser.add_argument("--partner-architecture", default="cnn", choices=("cnn", "rnn"))
    parser.add_argument(
        "--partner-br-root",
        default="saves/obp_br",
        help="Where --partner-arm looks for a trained best response.",
    )
    parser.add_argument(
        "--partner-prior-root",
        default="saves/obp_prior",
        help="Where --partner-arm=prior looks for the self-play checkpoint.",
    )
    parser.add_argument("--partner-label", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--layout", default="split_0")
    parser.add_argument(
        "--observation",
        default="canvas",
        choices=("canvas", "native"),
        help=(
            "Play on the shared 13x7 canvas, or at the layout's own size. It "
            "has to match what the model was trained on."
        ),
    )
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=450)
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help=(
            "Take the most likely action instead of sampling. People vary, and "
            "the environment is deterministic, so sampling is the default. Note "
            "that two deterministic copies of one policy deadlock: neither ever "
            "breaks a tie the other does not."
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help=(
            "Divide the logits before sampling. A human model fitted on data "
            "where most actions are 'stay' hesitates, and hesitation compounds "
            "over 450 steps; sharpening the distribution cuts that without "
            "making the policy deterministic. Measured optimum on split_0 is "
            "around 0.4-0.5, which is worth about half the model's score."
        ),
    )
    parser.add_argument("--data", default="data/human")
    parser.add_argument(
        "--entity", default=os.getenv("WANDB_ENTITY", "cilab-overcooked")
    )
    parser.add_argument(
        "--project", default=os.getenv("WANDB_PROJECT", "overcooked-v3-obp-eval")
    )
    parser.add_argument("--group", default="obp-eval")
    parser.add_argument("--tags", nargs="*", default=["obp"])
    parser.add_argument(
        "--wandb-mode",
        default=os.getenv("WANDB_MODE", "online"),
        choices=("online", "offline", "disabled"),
    )
    return parser.parse_args(argv)


def find_best_response(root, arm: str, seed: int, architecture="cnn", layout="obp10"):
    """The best response trained against this arm's human model, at this seed.

    The trainer names the folder after the arm and the file after the layout,
    so the pair identifies one run without a glob over everything.
    """
    folder = Path(root) / f"{layout}_{architecture}_obp_br_{arm}_seed{seed}"
    if not folder.is_dir():
        raise FileNotFoundError(
            f"No best-response run at {folder}. Train the BR sweep for arm "
            f"{arm!r} seed {seed} first."
        )
    matches = sorted(folder.glob(f"fcp_{architecture}_*_seed{seed}_vmap0.safetensors"))
    # An interrupted run leaves intermediate checkpoints behind; the final one
    # carries no update number, so it sorts last and is the one to play.
    final = [path for path in matches if "_update" not in path.name]
    if not final:
        raise FileNotFoundError(
            f"{folder} holds no final checkpoint (found {[p.name for p in matches]})."
        )
    if len(final) > 1:
        raise RuntimeError(f"{len(final)} final checkpoints in {folder}: {final[:3]}")
    return final[0]


def human_reference(data_root, layout: str) -> dict:
    """What the people in the demonstrations scored on this layout."""
    from baselines.OBP.data import find_episodes

    returns = []
    for path in find_episodes(data_root, [layout]):
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"]))
        value = metadata.get("episode_return")
        if value is not None:
            returns.append(float(value))
    if not returns:
        return {}
    return {
        "human/return_mean": float(np.mean(returns)),
        "human/return_std": float(np.std(returns)),
        "human/episodes": len(returns),
    }


def main(argv=None):
    args = parse_args(argv)

    import jax
    import jax.numpy as jnp
    import jaxmarl
    import wandb

    from baselines.OBP import bc
    from jaxmarl.environments.overcooked_v3.multi_layout import padded_dynamic_layout

    from baselines.OBP.train_bc import find_prior

    model = args.model
    prior_checkpoint = args.prior_checkpoint
    if args.model_root:
        if not args.arm:
            raise ValueError("--model-root needs --arm to build the directory")
        model = str(Path(args.model_root) / f"{args.arm}_seed{args.seed}")
    elif args.prior_root:
        prior_checkpoint = str(
            find_prior(
                args.prior_root,
                args.prior_architecture,
                args.prior_layout,
                args.seed,
                env_name=args.prior_env_name,
            )
        )

    layout = (
        padded_dynamic_layout(args.layout)
        if args.observation == "canvas"
        else args.layout
    )
    env = jaxmarl.make(
        "overcooked_v3",
        layout=layout,
        max_steps=args.max_steps,
        random_agent_positions=False,
        include_transition_countdown=True,
        include_layout_change_mask=True,
        transition_observer="both",
    )
    expected = tuple(env.observation_space("agent_0").shape)

    if model:
        logits_fn, params, config = bc.load(model)
        arm = args.arm or config.get("arm", Path(model).name)
        if config.get("split_side") == "validation":
            # A model fitted on the held-out episodes is the evaluation human,
            # not the arm whose folder name it shares.
            arm = f"{arm}-heldout"
        observation_shape = tuple(config["observation_shape"])
    else:
        network, logits_fn = bc.build_policy()
        # Whatever the environment gives: the prior is scored on the same
        # observation the cloned models are, so the arms read off one axis.
        observation_shape = expected
        params = bc.initial_params(
            network,
            observation_shape,
            jax.random.PRNGKey(0),
            prior_checkpoint=prior_checkpoint,
        )
        arm = args.arm or "prior"

    partner_logits = None
    partner_params = None
    partner_label = "self"
    if args.partner_model:
        partner_logits, partner_params, partner_config = bc.load(args.partner_model)
        partner_label = args.partner_label or partner_config.get(
            "arm", Path(args.partner_model).name
        )
        if tuple(partner_config["observation_shape"]) != observation_shape:
            raise ValueError(
                f"The partner reads {tuple(partner_config['observation_shape'])} "
                f"but the model reads {observation_shape}."
            )
    elif args.partner_arm:
        if args.partner_arm == "prior":
            partner_path = find_prior(
                args.partner_prior_root,
                args.prior_architecture,
                args.prior_layout,
                args.seed,
                env_name=args.prior_env_name,
            )
        else:
            partner_path = find_best_response(
                args.partner_br_root,
                args.partner_arm,
                args.seed,
                architecture=args.partner_architecture,
                layout=args.prior_layout,
            )
        partner_network, partner_logits = bc.build_policy(
            config={"ARCHITECTURE": args.partner_architecture}
        )
        partner_params = bc.initial_params(
            partner_network,
            observation_shape,
            jax.random.PRNGKey(0),
            prior_checkpoint=partner_path,
        )
        partner_label = args.partner_label or (
            "SP" if args.partner_arm == "prior" else f"BR({args.partner_arm})"
        )
    elif args.partner_checkpoint:
        partner_network, partner_logits = bc.build_policy(
            config={"ARCHITECTURE": args.partner_architecture}
        )
        partner_params = bc.initial_params(
            partner_network,
            observation_shape,
            jax.random.PRNGKey(0),
            prior_checkpoint=args.partner_checkpoint,
        )
        partner_label = args.partner_label or Path(args.partner_checkpoint).stem

    if expected != observation_shape:
        raise ValueError(
            f"The model expects {observation_shape} but {args.layout} at the "
            f"{args.observation} size gives {expected}."
        )

    def rollout(key):
        """One self-play episode; both seats are this model."""

        def choose(fn, own_params, observation, action_key):
            logits = fn(own_params, observation[None, :])[0]
            return (
                jnp.argmax(logits)
                if args.deterministic
                else jax.random.categorical(action_key, logits / args.temperature)
            )

        def step(carry, _):
            observations, state, key = carry
            key, first_key, second_key, env_key = jax.random.split(key, 4)
            first, second = env.agents
            actions = {
                first: choose(logits_fn, params, observations[first], first_key),
                second: choose(
                    partner_logits if partner_logits is not None else logits_fn,
                    partner_params if partner_params is not None else params,
                    observations[second],
                    second_key,
                ),
            }
            observations, state, reward, _, _ = env.step_env(env_key, state, actions)
            return (observations, state, key), reward[first]

        observations, state = env.reset(key)
        _, rewards = jax.lax.scan(
            step, (observations, state, key), jnp.arange(args.max_steps)
        )
        return jnp.sum(rewards)

    keys = jax.random.split(jax.random.PRNGKey(args.seed), args.episodes)
    returns = np.asarray(jax.jit(jax.vmap(rollout))(keys))

    reference = human_reference(args.data, args.layout)
    summary = {
        "arm": arm,
        "partner": partner_label,
        "return_mean": float(returns.mean()),
        "return_std": float(returns.std()),
        "return_min": float(returns.min()),
        "return_max": float(returns.max()),
        "scored_zero_fraction": float((returns < 1).mean()),
        **reference,
    }
    if "human/return_mean" in reference:
        # One is a model that cooks like the people it was cloned from.
        summary["return_vs_human"] = (
            summary["return_mean"] / reference["human/return_mean"]
            if reference["human/return_mean"]
            else float("nan")
        )

    run = wandb.init(
        entity=args.entity,
        project=args.project,
        group=args.group,
        tags=list(args.tags) + [arm],
        name=f"eval-{arm}-vs-{partner_label}-{args.layout}-seed{args.seed}",
        mode=args.wandb_mode,
        config={
            "arm": arm,
            "partner": partner_label,
            "seed": args.seed,
            "layout": args.layout,
            "episodes": args.episodes,
            "max_steps": args.max_steps,
            "deterministic": args.deterministic,
            "temperature": args.temperature,
            "model": model,
            "prior_checkpoint": prior_checkpoint,
        },
    )
    run.summary.update(summary)
    for key, value in summary.items():
        print(f"  {key:<28} {value}")
    run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
