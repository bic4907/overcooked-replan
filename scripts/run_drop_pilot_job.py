#!/usr/bin/env python3
"""Run one correlated W&B checkpoint job for the Drop-metric pilot."""

import os
import sys
from pathlib import Path


JOBS = {
    "ippo:split_0": ("overcooked-v3-ippo_train", "split_0", "ippo"),
    "ippo:outage_0": ("overcooked-v3-ippo_train", "outage_0", "ippo"),
    "ippo:distance_switch_0": (
        "overcooked-v3-ippo_train",
        "distance_switch_0",
        "ippo",
    ),
    "ippo:split_1": ("overcooked-v3-ippo-easy1_train", "split_1", "ippo"),
    "ippo:outage_1": (
        "overcooked-v3-ippo-outage1-v2_train",
        "outage_1",
        "ippo",
    ),
    "ippo:distance_switch_1": (
        "overcooked-v3-ippo-easy1_train",
        "distance_switch_1",
        "ippo",
    ),
    "ippo-rnn:split_0": (
        "overcooked-v3-ippo-rnn_train",
        "split_0",
        "ippo-rnn",
    ),
    "ippo-rnn:outage_0": (
        "overcooked-v3-ippo-rnn_train",
        "outage_0",
        "ippo-rnn",
    ),
    "ippo-rnn:distance_switch_0": (
        "overcooked-v3-ippo-rnn_train",
        "distance_switch_0",
        "ippo-rnn",
    ),
    "ippo-rnn:split_1": (
        "overcooked-v3-ippo-rnn-easy1_train",
        "split_1",
        "ippo-rnn",
    ),
    "ippo-rnn:outage_1": (
        "overcooked-v3-ippo-rnn-outage1-v2_train",
        "outage_1",
        "ippo-rnn",
    ),
    "ippo-rnn:distance_switch_1": (
        "overcooked-v3-ippo-rnn-easy1_train",
        "distance_switch_1",
        "ippo-rnn",
    ),
    "fcp:split_0": ("overcooked-v3-fcp_train", "split_0", "fcp"),
    "fcp:outage_0": ("overcooked-v3-fcp_train", "outage_0", "fcp"),
    "fcp:distance_switch_0": (
        "overcooked-v3-fcp_train",
        "distance_switch_0",
        "fcp",
    ),
}

# The public Split candidate was renamed from split_2 to split_0 after the
# baseline checkpoints were trained. Keep checkpoint selection separate from
# the layout used by the current evaluator.
SOURCE_LAYOUTS = {
    "ippo:split_0": "split_2",
    "ippo-rnn:split_0": "split_2",
    "fcp:split_0": "split_2",
}


def main() -> None:
    job = None
    for argument in sys.argv[1:]:
        if argument.startswith("job="):
            if job is not None:
                raise SystemExit("job may only be specified once")
            job = argument.removeprefix("job=")
        else:
            raise SystemExit(f"Unexpected argument: {argument}")

    if job not in JOBS:
        expected = ", ".join(sorted(JOBS))
        raise SystemExit(f"Unknown Drop pilot job {job!r}; expected one of: {expected}")

    source_project, layout, run_label = JOBS[job]
    source_layout = SOURCE_LAYOUTS.get(job, layout)
    algorithm = "FCP" if job.startswith("fcp:") else "IPPO"
    workers_per_gpu = os.getenv("DROP_PILOT_WORKERS_PER_GPU", "4")
    entrypoint = (
        Path(__file__).resolve().parents[1]
        / "baselines"
        / "IPPO"
        / "eval_crossplay_overcooked_v3.py"
    )
    argv = [
        sys.executable,
        str(entrypoint),
        f"cilab-overcooked/{source_project}",
        "--algorithms",
        algorithm,
        "--layout",
        layout,
        "--source-layout",
        source_layout,
        "--entity",
        "cilab-overcooked",
        "--episodes",
        "10",
        "--max-steps",
        "450",
        "--seeds",
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "--workers-per-gpu",
        workers_per_gpu,
        "--run-label",
        run_label,
        "--save-adaptation-traces",
    ]
    os.execv(sys.executable, argv)


if __name__ == "__main__":
    main()
