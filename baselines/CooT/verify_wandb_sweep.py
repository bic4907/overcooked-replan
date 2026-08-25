"""Fail when a completed W&B grid sweep contains incomplete or empty runs."""

from __future__ import annotations

import argparse
import math
from collections.abc import Mapping, Sequence

import wandb


def _sweep_parts(value: str) -> tuple[str, str, str]:
    parts = value.strip().split("/")
    if len(parts) != 3 or any(not part for part in parts):
        raise ValueError(
            f"Invalid sweep reference {value!r}; expected ENTITY/PROJECT/SWEEP_ID"
        )
    return parts[0], parts[1], parts[2]


def _objective_metric(config: Mapping[str, object]) -> str | None:
    metric = config.get("metric")
    if not isinstance(metric, Mapping):
        return None
    name = metric.get("name")
    return str(name) if name else None


def _has_finite_metric(summary: Mapping[str, object], name: str) -> bool:
    value = summary.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value))


def verify_sweep(sweep_ref: str, *, timeout: int = 90) -> list[str]:
    """Return human-readable success messages or raise on incomplete work."""

    entity, project, sweep_id = _sweep_parts(sweep_ref)
    api = wandb.Api(timeout=timeout)
    sweep = api.sweep(sweep_ref)
    runs = list(
        api.runs(
            f"{entity}/{project}",
            filters={"$and": [{"sweep": sweep_id}]},
            per_page=500,
            include_sweeps=False,
            lazy=False,
        )
    )
    if not runs:
        raise RuntimeError(f"Sweep {sweep_ref} has no runs")
    expected = sweep.expected_run_count
    if expected is not None and len(runs) != int(expected):
        raise RuntimeError(
            f"Sweep {sweep_ref} produced {len(runs)} run(s); expected {expected}"
        )

    bad_states = [
        f"{run.id} ({run.name}: {run.state})"
        for run in runs
        if str(run.state).lower() != "finished"
    ]
    if bad_states:
        preview = ", ".join(bad_states[:20])
        suffix = " ..." if len(bad_states) > 20 else ""
        raise RuntimeError(
            f"Sweep {sweep_ref} has {len(bad_states)} non-finished run(s): "
            f"{preview}{suffix}"
        )

    metric_name = _objective_metric(sweep.config)
    if metric_name:
        missing_metric = [
            f"{run.id} ({run.name})"
            for run in runs
            if not _has_finite_metric(dict(run.summary), metric_name)
        ]
        if missing_metric:
            preview = ", ".join(missing_metric[:20])
            suffix = " ..." if len(missing_metric) > 20 else ""
            raise RuntimeError(
                f"Sweep {sweep_ref} has {len(missing_metric)} run(s) without a "
                f"finite {metric_name!r} summary metric: {preview}{suffix}"
            )

    messages = [f"verified {len(runs)} finished run(s)"]
    if expected is not None:
        messages.append(f"grid count matches expected {expected}")
    if metric_name:
        messages.append(f"all runs logged {metric_name}")
    return messages


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sweep", help="ENTITY/PROJECT/SWEEP_ID")
    parser.add_argument("--timeout", type=int, default=90)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.timeout < 1:
        raise ValueError("--timeout must be positive")
    for message in verify_sweep(args.sweep, timeout=args.timeout):
        print(f"[OK] {message}")


if __name__ == "__main__":
    main()
