"""Bring trained self-play checkpoints down from W&B as priors.

The prior does not have to come from a sweep run on this machine: any self-play
run on the layout the demonstrations were collected in will do, and several
already exist. This pulls their final checkpoints into one directory laid out
the way ``find_prior`` expects.

Runs that never learned are skipped by default -- a prior that scores nothing
is not a prior, and starting a human model from one is the failure this is
meant to avoid.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", help="W&B project holding the self-play runs.")
    parser.add_argument(
        "--entity", default=os.getenv("WANDB_ENTITY", "cilab-overcooked")
    )
    parser.add_argument("--layout", default="split_0")
    parser.add_argument("--architecture", default="cnn")
    parser.add_argument("--output", default="saves/obp_prior_split0")
    parser.add_argument(
        "--metric",
        default="train/episode_return",
        help="Summary key read to decide whether a run learned anything.",
    )
    parser.add_argument(
        "--min-return",
        type=float,
        default=1.0,
        help="Skip runs below this. Set to 0 to take every finished run.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    import wandb

    api = wandb.Api(timeout=60)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    taken, skipped = [], []
    for run in api.runs(f"{args.entity}/{args.project}"):
        if run.state != "finished":
            continue
        env_kwargs = run.config.get("ENV_KWARGS") or {}
        if env_kwargs.get("layout") != args.layout:
            continue
        if str(run.config.get("ARCHITECTURE", "")).lower() != args.architecture:
            continue
        seed = run.config.get("SEED")
        score = run.summary.get(args.metric)
        if not isinstance(score, (int, float)) or score < args.min_return:
            skipped.append((run.name, seed, score))
            continue
        for artifact in run.logged_artifacts():
            if artifact.type != "checkpoint":
                continue
            artifact.download(root=str(output / f"seed{seed}"))
            taken.append((run.name, seed, float(score)))
            break

    for name, seed, score in sorted(taken, key=lambda row: row[1]):
        print(f"  seed{seed}: {name} ({args.metric}={score:.1f})")
    for name, seed, score in sorted(skipped, key=lambda row: (row[1] is None, row[1])):
        print(f"  skipped seed{seed}: {name} ({args.metric}={score})")
    print(f"{len(taken)} priors under {output}")
    if not taken:
        raise SystemExit(
            f"No {args.architecture.upper()} run on {args.layout} scored at "
            f"least {args.min_return} in {args.entity}/{args.project}."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
