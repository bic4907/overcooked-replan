"""Validate a CooT W&B sweep and its local stage inputs before launching agents."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[2]
CONF_DIR = REPO_ROOT / "conf"


def _load_mapping(path: Path) -> dict[str, Any]:
    payload = OmegaConf.to_container(OmegaConf.load(path), resolve=False)
    if not isinstance(payload, dict):
        raise ValueError(f"Sweep must be a YAML mapping: {path}")
    return payload


def _parameter_values(sweep: Mapping[str, Any], name: str) -> list[Any]:
    parameters = sweep.get("parameters")
    if not isinstance(parameters, Mapping) or name not in parameters:
        return []
    specification = parameters[name]
    if not isinstance(specification, Mapping):
        raise ValueError(f"Sweep parameter {name!r} must be a mapping")
    if "values" in specification:
        values = specification["values"]
        if not isinstance(values, list) or not values:
            raise ValueError(f"Sweep parameter {name!r}.values must be non-empty")
        return list(values)
    if "value" in specification:
        return [specification["value"]]
    raise ValueError(f"Sweep parameter {name!r} needs value or values")


def _single_parameter(sweep: Mapping[str, Any], name: str) -> Any | None:
    values = _parameter_values(sweep, name)
    if not values:
        return None
    if len(values) != 1:
        raise ValueError(f"Sweep parameter {name!r} must have exactly one value")
    return values[0]


def _resolve_repo_path(value: object) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _compose(config_name: str, overrides: Sequence[str]) -> dict[str, Any]:
    with initialize_config_dir(version_base=None, config_dir=str(CONF_DIR)):
        config = compose(config_name=config_name, overrides=list(overrides))
    payload = OmegaConf.to_container(config, resolve=True)
    if not isinstance(payload, dict):
        raise TypeError(f"Hydra config {config_name!r} did not resolve to a mapping")
    return payload


def _project(sweep: Mapping[str, Any]) -> str:
    value = _single_parameter(sweep, "PROJECT")
    if value is None:
        value = _single_parameter(sweep, "project")
    if not value:
        raise ValueError("CooT sweep must declare one PROJECT/project parameter")
    project = str(value)
    if not project.startswith("overcooked-v3-coot-"):
        raise ValueError(
            f"CooT project must use the isolated overcooked-v3-coot-* namespace: {project}"
        )
    return project


def _other_baseline_projects() -> set[str]:
    projects: set[str] = set()
    experiment_root = REPO_ROOT / "experiment"
    for path in experiment_root.glob("*/*.yaml"):
        if path.parent.name == "coot":
            continue
        sweep = _load_mapping(path)
        for key in ("PROJECT", "project", "output-project"):
            for value in _parameter_values(sweep, key):
                project = str(value).rsplit("/", 1)[-1]
                projects.add(project)
    return projects


def _validate_static(sweep_path: Path, sweep: Mapping[str, Any]) -> tuple[Path, str]:
    if Path.cwd().resolve() != REPO_ROOT:
        raise ValueError(
            f"Run the preflight and W&B agent from repository root: {REPO_ROOT}"
        )
    program_value = sweep.get("program")
    if not program_value:
        raise ValueError("Sweep is missing program")
    program = _resolve_repo_path(program_value)
    if not program.is_file():
        raise FileNotFoundError(f"Sweep program not found: {program}")
    command = sweep.get("command")
    if not isinstance(command, list) or not any(
        str(value) in {"${args_no_hyphens}", "${args_no_equals}"} for value in command
    ):
        raise ValueError("Sweep command must forward W&B parameters")
    project = _project(sweep)
    if project in _other_baseline_projects():
        raise ValueError(f"CooT project collides with another baseline: {project}")
    if not sweep_path.is_relative_to(REPO_ROOT / "experiment" / "coot"):
        raise ValueError("Only experiment/coot sweep files are accepted")
    return program, project


def _response_manifest(config: Mapping[str, Any], layout: str) -> Path:
    explicit = config.get("RESPONSE_JOB_MANIFEST")
    if explicit:
        return _resolve_repo_path(explicit)
    stage = str(config.get("RESPONSE_JOB_STAGE") or "exact")
    root = config.get("RESPONSE_JOB_ROOT")
    roots = config.get("RESPONSE_JOB_ROOTS")
    if not root and isinstance(roots, Mapping):
        root = roots.get(stage)
    if not root:
        raise ValueError(f"No response manifest root for stage {stage!r}")
    return _resolve_repo_path(root) / f"{layout}.json"


def _validate_response_inputs(sweep: Mapping[str, Any]) -> list[str]:
    scenarios = [str(value) for value in _parameter_values(sweep, "scenario")]
    stage = str(_single_parameter(sweep, "RESPONSE_JOB_STAGE") or "exact")
    job_indices = [int(value) for value in _parameter_values(sweep, "JOB_INDEX")]
    messages = []
    for scenario in scenarios:
        config = _compose(
            "coot_br_overcooked_v3",
            [f"scenario={scenario}", f"RESPONSE_JOB_STAGE={stage}"],
        )
        layout = str(config["ENV_KWARGS"]["layout"])
        manifest = _response_manifest(config, layout)
        if not manifest.is_file():
            raise FileNotFoundError(f"Response manifest not found: {manifest}")
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        jobs = payload.get("jobs") if isinstance(payload, Mapping) else payload
        if not isinstance(jobs, list) or not jobs:
            raise ValueError(f"Response manifest has no jobs: {manifest}")
        if job_indices and max(job_indices) >= len(jobs):
            raise IndexError(
                f"{manifest} has {len(jobs)} jobs but sweep requests "
                f"JOB_INDEX={max(job_indices)}"
            )
        selected = job_indices or list(range(len(jobs)))
        for index in selected:
            job = jobs[index]
            if not isinstance(job, Mapping):
                raise ValueError(f"Response job {index} is not a mapping: {manifest}")
            checkpoint_value = job.get("partner_checkpoint")
            partner = job.get("partner")
            if checkpoint_value is None and isinstance(partner, Mapping):
                checkpoint_value = partner.get("checkpoint")
            if not checkpoint_value:
                raise ValueError(f"Response job {index} has no partner checkpoint")
            checkpoint = Path(str(checkpoint_value)).expanduser()
            if not checkpoint.is_absolute():
                checkpoint = (manifest.parent / checkpoint).resolve()
            if not checkpoint.is_file():
                raise FileNotFoundError(
                    f"Response job {index} partner checkpoint not found: {checkpoint}"
                )
        messages.append(f"response manifest ready: {manifest} ({len(jobs)} jobs)")
    return messages


def _validate_train_inputs(sweep: Mapping[str, Any]) -> list[str]:
    scenarios = [str(value) for value in _parameter_values(sweep, "scenario")]
    dataset_override = _single_parameter(sweep, "DATASET_ROOT")
    messages = []
    for scenario in scenarios:
        overrides = [f"scenario={scenario}"]
        if dataset_override is not None:
            overrides.append(f"DATASET_ROOT={dataset_override}")
        config = _compose("coot_overcooked_v3", overrides)
        layout = str(config["ENV_KWARGS"]["layout"])
        dataset = _resolve_repo_path(config["DATASET_ROOT"]) / layout
        metadata = dataset / "metadata.json"
        shards = sorted(dataset.glob("pair_*.npz")) if dataset.is_dir() else []
        if not metadata.is_file() or not shards:
            raise FileNotFoundError(
                f"CooT dataset is incomplete for {layout}: expected {metadata} and pair_*.npz"
            )
        messages.append(f"dataset ready: {dataset} ({len(shards)} shards)")
    return messages


def _command_seeds(sweep: Mapping[str, Any]) -> list[int]:
    command = [str(value) for value in sweep.get("command") or []]
    if "--seeds" not in command:
        return list(range(6))
    start = command.index("--seeds") + 1
    seeds = []
    for value in command[start:]:
        if value.startswith("${") or value.startswith("--"):
            break
        seeds.append(int(value))
    if len(seeds) < 2:
        raise ValueError("Evaluation command needs at least two --seeds")
    return seeds


def _validate_eval_inputs(sweep: Mapping[str, Any]) -> list[str]:
    layouts = [str(value) for value in _parameter_values(sweep, "layout")]
    root_value = _single_parameter(sweep, "checkpoint-root")
    root = _resolve_repo_path(
        root_value or os.getenv("COOT_CHECKPOINT_ROOT", "saves/coot_train")
    )
    if not root.is_dir():
        raise FileNotFoundError(f"CooT checkpoint root not found: {root}")
    seeds = _command_seeds(sweep)
    messages = []
    for layout in layouts:
        for seed in seeds:
            pattern = f"coot_overcooked_v3_{layout}_seed{seed}_best.safetensors"
            matches = sorted(root.rglob(pattern))
            if not matches:
                raise FileNotFoundError(f"No {pattern} under {root}")
            if not matches[-1].with_suffix(".json").is_file():
                raise FileNotFoundError(
                    f"Checkpoint sidecar not found: {matches[-1].with_suffix('.json')}"
                )
        messages.append(f"eval checkpoints ready: {layout} seeds={seeds}")
    return messages


def preflight(sweep_path: Path, *, check_inputs: bool = True) -> list[str]:
    path = sweep_path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Sweep file not found: {path}")
    sweep = _load_mapping(path)
    program, project = _validate_static(path, sweep)
    messages = [f"program: {program.relative_to(REPO_ROOT)}", f"project: {project}"]
    if not check_inputs:
        return messages
    program_relative = program.relative_to(REPO_ROOT).as_posix()
    if program_relative.endswith("train_best_response_overcooked_v3.py"):
        messages.extend(_validate_response_inputs(sweep))
    elif program_relative.endswith("train_overcooked_v3.py"):
        messages.extend(_validate_train_inputs(sweep))
    elif program_relative.endswith("eval_crossplay_overcooked_v3.py"):
        messages.extend(_validate_eval_inputs(sweep))
    else:
        messages.append("inputs: population stage has no external file prerequisite")
    return messages


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sweep", type=Path)
    parser.add_argument(
        "--static-only",
        action="store_true",
        help="Validate naming/configuration but skip stage input files.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    messages = preflight(args.sweep, check_inputs=not args.static_only)
    for message in messages:
        print(f"[OK] {message}")
    print("CooT sweep preflight passed.")


if __name__ == "__main__":
    main()
