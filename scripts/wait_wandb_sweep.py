#!/usr/bin/env python3
"""Wait until every expected run in a W&B sweep finishes successfully."""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import sys
import time

import wandb


ACTIVE_STATES = {"pending", "queued", "running", "starting"}
SUCCESS_STATE = "finished"


def _timestamp() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def _state_counts(runs) -> collections.Counter[str]:
    return collections.Counter(str(run.state).lower() for run in runs)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("sweep_ref", help="ENTITY/PROJECT/SWEEP_ID")
    parser.add_argument("--expected-runs", type=int, required=True)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--timeout-hours", type=float, default=96.0)
    parser.add_argument("--api-timeout", type=int, default=30)
    args = parser.parse_args()

    if args.expected_runs <= 0:
        parser.error("--expected-runs must be positive")
    if args.poll_seconds <= 0:
        parser.error("--poll-seconds must be positive")
    if args.timeout_hours <= 0:
        parser.error("--timeout-hours must be positive")

    deadline = time.monotonic() + args.timeout_hours * 3600
    last_status = None

    while True:
        try:
            # W&B's public API caches sweep and run objects. Recreate the API
            # client so a long-lived barrier observes state changes made by
            # agents on other hosts.
            api = wandb.Api(timeout=args.api_timeout)
            sweep = api.sweep(args.sweep_ref)
            runs = list(sweep.runs)
            counts = _state_counts(runs)
            status = (
                str(sweep.state),
                len(runs),
                tuple(sorted(counts.items())),
            )
            if status != last_status:
                print(
                    f"[{_timestamp()}] {args.sweep_ref}: "
                    f"sweep={sweep.state} created={len(runs)}/"
                    f"{args.expected_runs} states={dict(counts)}",
                    flush=True,
                )
                last_status = status

            if len(runs) > args.expected_runs:
                print(
                    f"Sweep created {len(runs)} runs; expected exactly "
                    f"{args.expected_runs}",
                    file=sys.stderr,
                )
                return 1

            unexpected = {
                state: count
                for state, count in counts.items()
                if state != SUCCESS_STATE and state not in ACTIVE_STATES
            }
            if unexpected:
                print(
                    f"Sweep has non-success terminal runs: {unexpected}",
                    file=sys.stderr,
                )
                return 1

            if (
                len(runs) == args.expected_runs
                and counts.get(SUCCESS_STATE, 0) == args.expected_runs
            ):
                print(
                    f"[{_timestamp()}] Barrier passed for "
                    f"{args.sweep_ref}: {args.expected_runs} finished runs",
                    flush=True,
                )
                return 0
        except Exception as exc:  # Network/API failures are retried until timeout.
            print(
                f"[{_timestamp()}] W&B barrier query failed: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )

        if time.monotonic() >= deadline:
            print(
                f"Timed out waiting for {args.sweep_ref} after "
                f"{args.timeout_hours:g} hours",
                file=sys.stderr,
            )
            return 1
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
