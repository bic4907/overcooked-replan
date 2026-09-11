"""List/review collected episodes and export human-labelled BC samples."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from jaxmarl.environments.overcooked_v3.human_data import (
    ACTION_NAMES,
    SCHEMA_VERSION,
    atomic_save,
    load_episode,
)


def episode_paths(directory):
    paths = sorted(directory.rglob("*.npz"))
    if not paths:
        raise ValueError(f"No episode archives found in {directory}")
    return paths


def list_episodes(args):
    for path in episode_paths(args.input):
        _, meta = load_episode(path)
        print(
            f"{meta['status']:8}  {meta['env_kwargs']['layout']:22}  "
            f"steps={meta['length']:3}  score={meta['episode_return']:6g}  "
            f"complete={meta['complete']}  {path}"
        )


def review_episode(args):
    arrays, meta = load_episode(args.episode)
    meta.update(status=args.status, reviewed_at=datetime.now(timezone.utc).isoformat())
    arrays["metadata"] = np.asarray(json.dumps(meta, sort_keys=True))
    atomic_save(args.episode, arrays)
    print(f"{args.status}: {args.episode}")


def validate(arrays, meta, path):
    count = meta["length"]
    agents = len(meta["agents"])
    shape = tuple(meta["observation_shape"])
    if count < 1 or meta["action_names"] != ACTION_NAMES:
        raise ValueError(f"Empty episode or incompatible action mapping: {path}")
    expected = {
        "observations": (count + 1, agents, *shape),
        "actions": (count, agents),
        "human_mask": (count, agents),
        "rewards": (count, agents),
        "shaped_rewards": (count, agents),
        "dones": (count,),
        "state/layout_index": (count + 1,),
        "state/step": (count + 1,),
    }
    for key, wanted in expected.items():
        if key not in arrays or arrays[key].shape != wanted:
            raise ValueError(f"Invalid {key} shape in {path}; expected {wanted}")
    actions = arrays["actions"]
    if not np.issubdtype(actions.dtype, np.integer) or np.any(
        (actions < 0) | (actions >= len(ACTION_NAMES))
    ):
        raise ValueError(f"Invalid action labels: {path}")
    if arrays["human_mask"].dtype != np.bool_ or arrays["dones"].dtype != np.bool_:
        raise ValueError(f"Invalid mask/done dtype: {path}")
    if bool(arrays["dones"][-1]) != meta["complete"] or np.any(arrays["dones"][:-1]):
        raise ValueError(f"Inconsistent terminal flags: {path}")
    if not np.array_equal(arrays["state/step"], np.arange(count + 1)):
        raise ValueError(f"Non-contiguous state sequence: {path}")
    if not np.isfinite(arrays["observations"]).all():
        raise ValueError(f"Non-finite observations: {path}")


def flatten_episode(arrays, meta):
    # Observation at t precedes the selected action. Final obs at T is not a label.
    steps, agents = np.nonzero(arrays["human_mask"])
    return {
        "observations": arrays["observations"][steps, agents],
        "actions": arrays["actions"][steps, agents].astype(np.int32),
        "agent_index": agents.astype(np.int32),
        "timestep": steps.astype(np.int32),
        "episode_id": np.full(len(steps), meta["episode_id"]),
        "rewards": arrays["rewards"][steps, agents],
        "dones": arrays["dones"][steps],
        "layout_index": arrays["state/layout_index"][steps],
    }


def export(args):
    if not 0 <= args.val_fraction < 1:
        raise ValueError("--val-fraction must be in [0, 1)")
    if args.output.exists() and (
        not args.output.is_dir() or any(args.output.iterdir())
    ):
        raise ValueError(
            "Choose a new/empty --output directory to preserve previous exports"
        )
    selected = []
    seen_ids = set()
    fingerprint = None
    # Validate every selected archive before writing the dataset.
    for path in episode_paths(args.input):
        arrays, meta = load_episode(path)
        if args.layout and meta["env_kwargs"]["layout"] != args.layout:
            continue
        if meta["status"] != "accepted" or (
            not meta["complete"] and not args.include_partial
        ):
            continue
        validate(arrays, meta, path)
        if not arrays["human_mask"].any():
            continue
        if meta["episode_id"] in seen_ids:
            raise ValueError(f"Duplicate episode ID (copied archive): {path}")
        seen_ids.add(meta["episode_id"])
        if fingerprint is not None and fingerprint != meta["env_fingerprint"]:
            raise ValueError(
                "Selected episodes use different environment versions/configurations. "
                "Export each layout/version separately (use --layout or separate input folders)."
            )
        fingerprint = meta["env_fingerprint"]
        selected.append((path, meta))
    if not selected:
        raise ValueError(
            "No accepted, complete, human-labelled episodes. Review recordings first; use --include-partial only if intended."
        )
    if args.val_fraction > 0 and len(selected) < 2:
        raise ValueError(
            "An episode-level validation split needs at least two episodes; use --val-fraction 0 for training only."
        )
    order = np.random.default_rng(args.seed).permutation(len(selected))
    val_count = (
        0
        if args.val_fraction == 0
        else min(len(selected) - 1, max(1, round(len(selected) * args.val_fraction)))
    )
    groups = {"train": order[val_count:], "val": order[:val_count]}
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "env_fingerprint": fingerprint,
        "env_kwargs": selected[0][1]["env_kwargs"],
        "action_names": ACTION_NAMES,
        "seed": args.seed,
        "validation_fraction": args.val_fraction,
        "split_unit": "episode",
        "include_partial": args.include_partial,
        "observation_axes": "sample, height, width, channel",
        "splits": {},
    }
    for name, indices in groups.items():
        pieces = []
        episodes = []
        for index in indices:
            path, meta = selected[index]
            arrays, loaded_meta = load_episode(path)
            # Refuse a concurrently reviewed/changed input after selection.
            if loaded_meta != meta:
                raise ValueError(
                    f"Episode changed during export; retry into a new output: {path}"
                )
            pieces.append(flatten_episode(arrays, meta))
            episodes.append({"path": str(path.resolve()), "metadata": meta})
        if pieces:
            data = {
                key: np.concatenate([piece[key] for piece in pieces])
                for key in pieces[0]
            }
        else:
            arrays, meta = load_episode(selected[0][0])
            data = {
                key: value[:0] for key, value in flatten_episode(arrays, meta).items()
            }
        atomic_save(args.output / f"{name}.npz", data)
        histogram = np.bincount(data["actions"], minlength=len(ACTION_NAMES)).tolist()
        manifest["splits"][name] = {
            "episodes": episodes,
            "samples": len(data["actions"]),
            "action_counts": dict(zip(ACTION_NAMES, histogram)),
        }
        print(
            f"{name}: {len(episodes)} episodes / {len(data['actions'])} samples / {dict(zip(ACTION_NAMES, histogram))}"
        )
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"BC dataset: {args.output.resolve()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    listing = commands.add_parser("list", help="Show episode status and scores")
    listing.add_argument("input", type=Path)
    listing.set_defaults(run=list_episodes)
    review = commands.add_parser(
        "review", help="Update a saved episode's curation status"
    )
    review.add_argument("episode", type=Path)
    review.add_argument(
        "--status", choices=("accepted", "rejected", "pending"), required=True
    )
    review.set_defaults(run=review_episode)
    prepare = commands.add_parser(
        "export", help="Export accepted demonstrations as BC arrays"
    )
    prepare.add_argument("input", type=Path)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--layout")
    prepare.add_argument("--val-fraction", type=float, default=0.2)
    prepare.add_argument("--seed", type=int, default=0)
    prepare.add_argument("--include-partial", action="store_true")
    prepare.set_defaults(run=export)
    args = parser.parse_args()
    try:
        args.run(args)
    except (ValueError, KeyError, OSError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
