#!/usr/bin/env python3
"""Translate one correlated recovery job into Hydra scenario/seed overrides."""

import os
from pathlib import Path
import sys


RECOVERY_JOBS = {
    "split_1:3",
    "split_1:4",
    "split_1:5",
    "outage_1:3",
    "outage_1:4",
    "outage_1:5",
    "distance_switch_1:2",
    "distance_switch_1:3",
    "distance_switch_1:4",
    "distance_switch_1:5",
}


def main() -> None:
    job = None
    forwarded = []
    for argument in sys.argv[1:]:
        if argument.startswith("job="):
            if job is not None:
                raise SystemExit("job may only be specified once")
            job = argument.removeprefix("job=")
        else:
            forwarded.append(argument)

    if job not in RECOVERY_JOBS:
        expected = ", ".join(sorted(RECOVERY_JOBS))
        raise SystemExit(f"Unknown recovery job {job!r}; expected one of: {expected}")

    scenario, seed = job.rsplit(":", 1)
    entrypoint = (
        Path(__file__).resolve().parents[1]
        / "baselines"
        / "IPPO"
        / "ippo_overcooked_v3.py"
    )
    argv = [
        sys.executable,
        str(entrypoint),
        *forwarded,
        f"scenario={scenario}",
        f"SEED={seed}",
    ]
    os.execv(sys.executable, argv)


if __name__ == "__main__":
    main()
