"""Attach checkpoints on disk to the W&B runs that should already carry them.

The 75-step recurrent runs were logged as references: they name the sweep the
weights came from, and that project has since been deleted, so the runs describe
policies nobody can load. The weights themselves are still on the machine that
trained them.

This walks a directory of ``.safetensors`` files, works out which kitchen and
seed each belongs to, finds the run in the project that says the same, and logs
the file to that run as its final checkpoint. It resumes the existing run rather
than starting one, so nothing new appears in the project:

    python scripts/overcooked_v3/attach_checkpoints.py --root saves/rnn75 --dry-run
    python scripts/overcooked_v3/attach_checkpoints.py --root saves/rnn75

A run that already carries a checkpoint is left alone unless --replace is given.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

LAYOUTS = (
    "split_0", "split_1", "outage_0", "outage_1", "distance_0", "distance_1",
)
SEED = re.compile(r"seed[_-]?(\d+)")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", required=True, help="Directory holding the checkpoints.")
    parser.add_argument(
        "--project", default="overcooked-v3-ippo-rnn-75step-plate-0915_train"
    )
    parser.add_argument("--entity", default=os.getenv("WANDB_ENTITY", "cilab-overcooked"))
    parser.add_argument(
        "--observer",
        default="both",
        help="Only take checkpoints from this observer arm, when the path says.",
    )
    parser.add_argument("--alias", default="final")
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Attach even to runs that already carry a checkpoint artifact.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def describe(path: Path, observer: str):
    """The kitchen and seed a checkpoint belongs to, or None if it is not ours."""
    text = str(path)
    layouts = [name for name in LAYOUTS if name in text]
    if not layouts:
        return None
    # split_0 is a substring of nothing, but distance_1 and distance_10 would
    # collide if the benchmark ever grew; take the longest name that matches.
    layout = max(layouts, key=len)
    seed = SEED.search(text)
    if seed is None:
        return None
    # Observer arms live side by side in the same tree -- as a suffix on the
    # save directory, and as a directory of its own under the evaluation copies
    # -- and only one of them is the benchmark policy. A file that names no arm
    # at all is skipped rather than guessed at: the four arms are the same
    # kitchen and seed, and picking the wrong one is silent.
    arms = ("none", "agent_0", "agent_1", "both")
    named = {
        arm
        for arm in arms
        if f"observer-{arm}" in text
        or f"observer_{arm}" in text
        or arm in path.parts
    }
    if named != {observer}:
        return None
    return layout, int(seed.group(1))


def main(argv=None):
    args = parse_args(argv)
    import wandb

    root = Path(args.root)
    found = {}
    for path in sorted(root.rglob("*.safetensors")):
        if "_update" in path.stem:
            continue
        described = describe(path, args.observer)
        if described is None:
            continue
        found.setdefault(described, path)
    if not found:
        raise SystemExit(f"No checkpoints under {root} name a kitchen and a seed")
    print(f"{len(found)} checkpoints under {root}")

    api = wandb.Api()
    runs = {}
    for run in api.runs(f"{args.entity}/{args.project}", per_page=500):
        config = run.config
        layout = (config.get("ENV_KWARGS") or {}).get("layout") or config.get("layout")
        seed = config.get("SEED")
        seed = config.get("seed") if seed is None else seed
        if layout is None or seed is None:
            continue
        runs.setdefault((layout, int(seed)), run)

    missing = sorted(set(found) - set(runs))
    if missing:
        print(f"no run for: {missing}")

    for key in sorted(found):
        run = runs.get(key)
        if run is None:
            continue
        path = found[key]
        already = [a for a in run.logged_artifacts() if a.type == "checkpoint"]
        if already and not args.replace:
            print(f"  {key[0]} seed {key[1]}: {run.id} already has one, skipped")
            continue
        print(f"  {key[0]} seed {key[1]}: {run.id} <- {path}")
        if args.dry_run:
            continue
        # resume="must" fails rather than opening a new run, which is the point:
        # the project must not grow a second copy of every policy.
        resumed = wandb.init(
            entity=args.entity, project=args.project, id=run.id, resume="must"
        )
        artifact = wandb.Artifact(
            f"overcooked-v3-{run.id}-final-checkpoint", type="checkpoint"
        )
        artifact.add_file(str(path))
        resumed.log_artifact(artifact, aliases=[args.alias])
        resumed.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
