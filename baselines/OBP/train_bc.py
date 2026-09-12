"""Train one human-model arm and record how well it fits held-out people.

Three arms answer the paper's question directly:

    bc          behavior cloning from random weights
    obp         the same, started from the self-play prior
    obp-frozen  the same again, with the prior's convolutional trunk held fixed

They differ in nothing else -- same data, same split, same optimiser, same
seed -- so the gap between them is the prior and only the prior.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

ARMS = ("bc", "obp", "obp-frozen")

#: The trunk the frozen arm holds fixed: everything the prior learned about
#: reading a kitchen, leaving the action and value heads to the human data.
TRUNK_PREFIX = "params/CNN_0"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--data", default="data/human", help="Root of the demonstration archives."
    )
    parser.add_argument(
        "--layouts",
        nargs="+",
        default=None,
        help="Restrict to these layouts; default is every layout under --data.",
    )
    parser.add_argument(
        "--prior-root",
        default="saves/obp_prior",
        help=(
            "Where the self-play prior's checkpoints live. The arm's own seed "
            "selects one, so a prior and the model built on it share a seed."
        ),
    )
    parser.add_argument("--prior-architecture", default="cnn", choices=("cnn", "rnn"))
    parser.add_argument("--prior-layout", default="obp10")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--val-fraction", type=float, default=0.25)
    parser.add_argument("--output-root", default="saves/obp_bc")
    parser.add_argument(
        "--entity", default=os.getenv("WANDB_ENTITY", "cilab-overcooked")
    )
    parser.add_argument(
        "--project", default=os.getenv("WANDB_PROJECT", "overcooked-v3-obp")
    )
    parser.add_argument("--group", default="obp")
    parser.add_argument("--tags", nargs="*", default=["obp"])
    parser.add_argument(
        "--wandb-mode",
        default=os.getenv("WANDB_MODE", "online"),
        choices=("online", "offline", "disabled"),
    )
    parser.add_argument("--upload-checkpoint", action="store_true", default=True)
    parser.add_argument(
        "--no-upload-checkpoint", dest="upload_checkpoint", action="store_false"
    )
    return parser.parse_args(argv)


def find_prior(root, architecture: str, layout: str, seed: int) -> Path:
    """The prior trained with this seed, or a message saying what is missing."""
    root = Path(root)
    name = f"ippo_{architecture}_overcooked_v3_multilayout_{layout}_seed{seed}_vmap0.safetensors"
    matches = sorted(root.rglob(name))
    if not matches:
        available = sorted({path.name for path in root.rglob("*.safetensors")})[:6]
        raise FileNotFoundError(
            f"No prior for seed {seed} under {root} (looked for {name}). "
            f"Found instead: {available or 'nothing'}. Train the prior sweep first."
        )
    if len(matches) > 1:
        raise RuntimeError(
            f"{len(matches)} priors match {name} under {root}: {matches[:3]}. "
            "Point --prior-root at one experiment folder."
        )
    return matches[0]


def main(argv=None):
    args = parse_args(argv)

    # Imported here so --help stays fast and does not need a GPU.
    import wandb

    from baselines.OBP import bc
    from baselines.OBP.data import (
        build_dataset,
        find_episodes,
        load_episode,
        split_by_episode,
    )

    paths = find_episodes(args.data, args.layouts)
    if not paths:
        raise FileNotFoundError(
            f"No demonstrations under {args.data}"
            + (f" for layouts {args.layouts}" if args.layouts else "")
        )
    episodes = [load_episode(path) for path in paths]
    dataset = build_dataset(episodes)
    train_rows, validation_rows = split_by_episode(
        dataset, args.val_fraction, seed=args.seed
    )
    train_rows = np.asarray(train_rows)
    validation_rows = np.asarray(validation_rows)

    observations = dataset.observations[train_rows]
    actions = dataset.actions[train_rows]
    held_out = dataset.observations[validation_rows] if validation_rows.any() else None
    held_out_actions = (
        dataset.actions[validation_rows] if validation_rows.any() else None
    )

    prior = None
    freeze = ()
    if args.arm in ("obp", "obp-frozen"):
        prior = find_prior(
            args.prior_root, args.prior_architecture, args.prior_layout, args.seed
        )
    if args.arm == "obp-frozen":
        freeze = (TRUNK_PREFIX,)

    layouts = sorted({episode.layout for episode in episodes})
    run = wandb.init(
        entity=args.entity,
        project=args.project,
        group=args.group,
        tags=list(args.tags) + [args.arm],
        name=f"{args.arm}-seed{args.seed}",
        mode=args.wandb_mode,
        config={
            "arm": args.arm,
            "seed": args.seed,
            "layouts": layouts,
            "episodes": len(episodes),
            "train_rows": int(train_rows.sum()),
            "validation_rows": int(validation_rows.sum()),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "val_fraction": args.val_fraction,
            "prior_checkpoint": str(prior) if prior else None,
            "frozen_prefixes": list(freeze),
            "human_return_mean": float(
                np.mean([episode.episode_return for episode in episodes])
            ),
        },
    )

    report = bc.train(
        observations,
        actions,
        held_out,
        held_out_actions,
        prior_checkpoint=prior,
        freeze_prefixes=freeze,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        log=lambda record: wandb.log(record, step=record["epoch"]),
    )

    # Score the action mix on whatever the model was held out against, falling
    # back to the training rows when nothing was held out.
    _, logits_fn = bc.build_policy()
    if held_out is not None and held_out_actions is not None:
        scored_observations, scored_actions = held_out, held_out_actions
    else:
        scored_observations, scored_actions = observations, actions

    action_names = ["right", "down", "left", "up", "stay", "interact"]
    shares = bc.action_distribution(
        logits_fn, report.params, scored_observations[:8192]
    )
    truth = np.bincount(
        np.asarray(scored_actions[:8192]), minlength=len(action_names)
    ).astype(float)
    truth /= truth.sum()

    last = report.history[-1]
    summary = {
        **{f"final/{key}": value for key, value in last.items() if key != "epoch"},
        **{
            f"predicted/{name}": float(share)
            for name, share in zip(action_names, shares)
        },
        **{f"actual/{name}": float(share) for name, share in zip(action_names, truth)},
        "frozen_parameters": len(report.frozen),
        # Total variation between the model's action mix and the people's: zero
        # when they agree, one when the model never picks what a person picks.
        # A model that has collapsed onto the majority action reads high here
        # even while its accuracy looks respectable.
        "action_mix_distance": float(np.abs(shares - truth).sum() / 2),
    }
    run.summary.update(summary)

    output = Path(args.output_root) / f"{args.arm}_seed{args.seed}"
    bc.save(
        output,
        report.params,
        observations.shape[1:],
        metadata={
            "arm": args.arm,
            "seed": args.seed,
            "layouts": layouts,
            "prior_checkpoint": str(prior) if prior else None,
            "frozen_prefixes": list(freeze),
        },
    )
    (output / "history.json").write_text(json.dumps(report.history, indent=1))
    print(f"Saved human model: {output}")

    if args.upload_checkpoint and args.wandb_mode == "online":
        artifact = wandb.Artifact(f"obp-{args.arm}-seed{args.seed}", type="human-model")
        artifact.add_dir(str(output))
        run.log_artifact(artifact, aliases=["final"])
        run.summary["checkpoint/uploaded"] = True

    run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
