#!/usr/bin/env python3
"""Create an Overcooked V3 W&B sweep and remember its full path."""

from __future__ import annotations

import argparse
import os
from collections.abc import Mapping, MutableMapping
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "sweeps" / "overcooked_v3_role_scenarios.yaml"
DEFAULT_OUTPUT = PROJECT_ROOT / "sweeps" / ".last_sweep_id"
PLACEHOLDER_ENTITIES = {"your-team-slug", "your-entity", "entity"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a W&B sweep from a YAML config and save its full path."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Sweep YAML path (default: {DEFAULT_CONFIG.relative_to(PROJECT_ROOT)})",
    )
    parser.add_argument(
        "--entity",
        help="W&B team slug. Defaults to WANDB_ENTITY from the shell or .env.",
    )
    parser.add_argument(
        "--project",
        help="W&B project. Defaults to WANDB_PROJECT or the sweep YAML project.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"File that receives ENTITY/PROJECT/SWEEP_ID (default: {DEFAULT_OUTPUT.relative_to(PROJECT_ROOT)})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and display the resolved settings without contacting W&B.",
    )
    return parser.parse_args()


def load_sweep_config(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"Sweep config does not exist: {path}")

    config = OmegaConf.to_container(OmegaConf.load(path), resolve=False)
    if not isinstance(config, MutableMapping):
        raise ValueError(f"Sweep config must contain a YAML mapping: {path}")
    if "program" not in config or "parameters" not in config:
        raise ValueError("Sweep config must define both 'program' and 'parameters'.")
    return dict(config)


def resolve_destination(
    args: argparse.Namespace, config: MutableMapping[str, Any]
) -> tuple[str, str]:
    entity = args.entity or os.environ.get("WANDB_ENTITY")
    yaml_project = config.pop("project", None)
    project = args.project or os.environ.get("WANDB_PROJECT") or yaml_project

    if not entity or entity.lower() in PLACEHOLDER_ENTITIES:
        raise ValueError(
            "Set WANDB_ENTITY in .env to a W&B team slug (not an organization slug), "
            "or pass --entity."
        )
    if not project:
        raise ValueError(
            "Set WANDB_PROJECT in .env, pass --project, or define project in YAML."
        )

    parameters = config.get("parameters")
    if isinstance(parameters, MutableMapping) and "PROJECT" in parameters:
        project_parameter = parameters["PROJECT"]
        if isinstance(project_parameter, MutableMapping):
            project_parameter.clear()
            project_parameter["value"] = project

    return entity, str(project)


def full_sweep_path(entity: str, project: str, sweep_id: str) -> str:
    parts = str(sweep_id).strip().strip("/").split("/")
    if len(parts) >= 3:
        return "/".join(parts[-3:])
    return f"{entity}/{project}/{parts[-1]}"


def validate_parameter_mapping(config: Mapping[str, Any]) -> None:
    parameters = config.get("parameters")
    if not isinstance(parameters, Mapping) or not parameters:
        raise ValueError("Sweep config 'parameters' must be a non-empty mapping.")


def main() -> int:
    args = parse_args()
    loaded_env = load_dotenv(PROJECT_ROOT / ".env", override=False)
    config = load_sweep_config(args.config)
    validate_parameter_mapping(config)
    entity, project = resolve_destination(args, config)

    print(f"Sweep config: {args.config.expanduser().resolve()}")
    print(f"W&B destination: {entity}/{project}")
    if loaded_env:
        print("Loaded project .env")

    if args.dry_run:
        print("Dry run complete; no sweep was created.")
        return 0

    import wandb

    sweep_id = wandb.sweep(config, entity=entity, project=project)
    sweep_path = full_sweep_path(entity, project, sweep_id)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(f"{sweep_path}\n", encoding="utf-8")

    print(f"Created sweep: {sweep_path}")
    print(f"Saved sweep path: {output}")
    print('Run agents: GPUS="0 1" bash scripts/overcooked_v3/run_wandb_agents.sh')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
