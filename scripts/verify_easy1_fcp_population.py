#!/usr/bin/env python3
"""Fail unless the Easy-1 FCP population contains every expected snapshot."""

from __future__ import annotations

import argparse
from pathlib import Path


LAYOUTS = ("split_1", "outage_1", "distance_switch_1")
SEEDS = range(3)
SNAPSHOT_SUFFIXES = (
    "_update000046.safetensors",
    "_update000229.safetensors",
    ".safetensors",
)


def expected_paths(root: Path, layouts, seeds) -> set[Path]:
    paths = set()
    for layout in layouts:
        for seed in seeds:
            folder = root / f"{layout}_rnn_seed{seed}"
            stem = f"ippo_rnn_overcooked_v3_{layout}_seed{seed}_vmap0"
            paths.update(folder / f"{stem}{suffix}" for suffix in SNAPSHOT_SUFFIXES)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("population_dir", type=Path)
    parser.add_argument("--layouts", nargs="+", default=list(LAYOUTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    args = parser.parse_args()

    root = args.population_dir.expanduser().resolve()
    expected = expected_paths(root, args.layouts, args.seeds)
    actual = set(root.rglob("*.safetensors")) if root.is_dir() else set()
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    empty = sorted(path for path in expected & actual if path.stat().st_size == 0)

    if missing or unexpected or empty:
        print(f"FCP population validation failed under {root}")
        if missing:
            print("Missing:")
            print("\n".join(f"  {path.relative_to(root)}" for path in missing))
        if unexpected:
            print("Unexpected:")
            print("\n".join(f"  {path.relative_to(root)}" for path in unexpected))
        if empty:
            print("Empty:")
            print("\n".join(f"  {path.relative_to(root)}" for path in empty))
        return 1

    print(f"Validated {len(actual)} FCP population checkpoints under {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
