"""Evaluate an ordered cross-play matrix from W&B checkpoint artifacts."""

import argparse
import csv
import json
import logging
import os
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import numpy as np
import wandb

from jaxmarl._env import load_project_env
from jaxmarl.environments.overcooked_v3 import overcooked_v3_layouts
from jaxmarl.wrappers.baselines import load_params

try:
    from .eval_wandb_crossplay_overcooked_v3 import (
        _artifact_reference,
        _source_layout,
        evaluate_crossplay,
        evaluation_signature,
        prepare_crossplay_runtime,
        select_final_artifact,
    )
except ImportError:  # Direct execution: python baselines/IPPO/<script>.py
    from eval_wandb_crossplay_overcooked_v3 import (
        _artifact_reference,
        _source_layout,
        evaluate_crossplay,
        evaluation_signature,
        prepare_crossplay_runtime,
        select_final_artifact,
    )


from matplotlib import pyplot as plt  # noqa: E402


LOGGER = logging.getLogger("crossplay-matrix")
VMAP_PATTERN = re.compile(r"_vmap(?P<index>\d+)\.safetensors$")
ALGORITHM_CONFIG_KEYS = ("ALGORITHM", "algorithm", "ALG_NAME", "algo_name")


@dataclass
class RunCandidate:
    run: object
    artifact: object
    algorithm: str
    layout: str
    seed: int | None
    config: dict


@dataclass
class PolicyModel:
    algorithm: str
    layout: str
    seed: int | None
    run_id: str
    run_path: str
    vmap_index: int
    checkpoint: Path
    config: dict
    artifact_name: str

    @property
    def identity(self):
        return f"{self.run_path}:{self.artifact_name}:vmap{self.vmap_index}"

    @property
    def label(self):
        seed = "?" if self.seed is None else self.seed
        return f"{self.algorithm}|s{seed}|{self.run_id}|v{self.vmap_index}"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Discover trained policies in one W&B project, evaluate the full "
            "ordered self-/cross-play matrix, and log aggregate results to a "
            "separate W&B project."
        )
    )
    parser.add_argument(
        "source_project",
        help="Training project as PROJECT or ENTITY/PROJECT.",
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        required=True,
        help=(
            "Algorithms to compare. A run matches by ALGORITHM-like config key "
            "or W&B tag (for example: IPPO)."
        ),
    )
    parser.add_argument(
        "--layout",
        "--layouts",
        dest="layout",
        required=True,
        choices=sorted(overcooked_v3_layouts),
        help=(
            "Single training/evaluation map. --layouts is retained as a "
            "backward-compatible alias but also accepts exactly one map."
        ),
    )
    parser.add_argument(
        "--entity",
        default=os.getenv("WANDB_ENTITY"),
        help="Entity for a bare source project name.",
    )
    parser.add_argument(
        "--output-project",
        help=(
            "Evaluation project as PROJECT or ENTITY/PROJECT. Defaults to "
            "SOURCE_PROJECT-crossplay."
        ),
    )
    parser.add_argument("--output-entity", help="Entity for the evaluation project.")
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default=os.getenv("WANDB_MODE", "online"),
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0, help="Evaluation RNG seed.")
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        help="Optional training-seed filter.",
    )
    parser.add_argument(
        "--vmap-indices",
        nargs="+",
        type=int,
        help="Optional checkpoint vmap-index filter. Defaults to every final model.",
    )
    parser.add_argument(
        "--artifact-alias",
        default="final",
        help="Required checkpoint artifact alias (default: final).",
    )
    parser.add_argument(
        "--run-state",
        choices=("finished", "running", "all"),
        default="finished",
    )
    parser.add_argument(
        "--latest-per-seed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep only the newest run for each algorithm/layout/training seed.",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample policy actions instead of using their modes.",
    )
    parser.add_argument(
        "--gpus",
        nargs="+",
        help=(
            "CUDA device IDs for parallel pair workers, for example --gpus 0 1 2 3. "
            "One long-lived worker is started per GPU. The user-facing entrypoint "
            "defaults to the first visible GPU."
        ),
    )
    parser.add_argument(
        "--workers-per-gpu",
        type=int,
        default=8,
        help=(
            "Independent evaluation worker processes per GPU (default: 8). "
            "Adjust this to match GPU memory and utilization."
        ),
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        help=(
            "Checkpoint download directory. Defaults to artifacts/ inside this "
            "evaluation run's output directory."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Local result/cache directory. Reusing it resumes completed pairs. "
            "Defaults to a unique run directory under saves/crossplay/."
        ),
    )
    return parser.parse_args(argv)


def split_project_path(value, default_entity=None):
    components = value.strip("/").split("/")
    if len(components) == 2 and all(components):
        return components[0], components[1]
    if len(components) == 1 and components[0] and default_entity:
        return default_entity, components[0]
    raise ValueError(
        f"Invalid W&B project '{value}'. Use ENTITY/PROJECT or provide --entity."
    )


def _run_path(run):
    path = getattr(run, "path", None)
    if isinstance(path, (tuple, list)):
        return "/".join(path)
    if isinstance(path, str) and path:
        return path.strip("/")
    return f"{run.entity}/{run.project}/{run.id}"


def match_algorithm(run, requested_algorithms):
    """Return the requested canonical label matching a config value or run tag."""
    config = dict(getattr(run, "config", {}) or {})
    configured = [
        str(config[key]).casefold()
        for key in ALGORITHM_CONFIG_KEYS
        if config.get(key) is not None
    ]
    if configured:
        for requested in requested_algorithms:
            if requested.casefold() in configured:
                return requested
        return None

    candidates = {str(tag).casefold() for tag in (getattr(run, "tags", []) or [])}
    matches = [
        requested
        for requested in requested_algorithms
        if requested.casefold() in candidates
    ]
    if len(matches) > 1:
        raise ValueError(
            f"Run {_run_path(run)} matches multiple algorithm tags {matches}. "
            "Set ALGORITHM in its W&B config to disambiguate it."
        )
    return matches[0] if matches else None


def _training_seed(config):
    value = config.get("SEED", config.get("seed"))
    return None if value is None else int(value)


def discover_run_candidates(
    runs,
    algorithms,
    layouts,
    artifact_alias="final",
    seeds=None,
    latest_per_seed=True,
):
    """Filter W&B runs locally and retain checkpoint-bearing candidates."""
    layouts = set(layouts)
    seeds = None if seeds is None else set(seeds)
    candidates = []
    for run in runs:
        algorithm = match_algorithm(run, algorithms)
        if algorithm is None:
            continue
        config = dict(getattr(run, "config", {}) or {})
        layout = _source_layout(config)
        seed = _training_seed(config)
        if layout not in layouts or (seeds is not None and seed not in seeds):
            continue
        try:
            artifact = select_final_artifact(run, artifact_alias)
        except FileNotFoundError as error:
            LOGGER.warning("Skipping %s: %s", _run_path(run), error)
            continue
        candidates.append(
            RunCandidate(
                run=run,
                artifact=artifact,
                algorithm=algorithm,
                layout=layout,
                seed=seed,
                config=config,
            )
        )

    if latest_per_seed:
        latest = {}
        for candidate in candidates:
            seed_key = (
                candidate.seed
                if candidate.seed is not None
                else getattr(candidate.run, "id", _run_path(candidate.run))
            )
            key = (candidate.algorithm.casefold(), candidate.layout, seed_key)
            current = latest.get(key)
            candidate_time = str(getattr(candidate.run, "created_at", ""))
            current_time = (
                str(getattr(current.run, "created_at", "")) if current else ""
            )
            if current is None or candidate_time >= current_time:
                latest[key] = candidate
        candidates = list(latest.values())

    algorithm_order = {name.casefold(): index for index, name in enumerate(algorithms)}
    return sorted(
        candidates,
        key=lambda item: (
            item.layout,
            algorithm_order[item.algorithm.casefold()],
            item.seed is None,
            -1 if item.seed is None else item.seed,
            getattr(item.run, "id", ""),
        ),
    )


def resolve_vmap_checkpoints(artifact_dir, requested_indices=None):
    requested = None if requested_indices is None else set(requested_indices)
    checkpoints = {}
    for path in Path(artifact_dir).rglob("*.safetensors"):
        if "_update" in path.stem:
            continue
        match = VMAP_PATTERN.search(path.name)
        if match is None:
            continue
        index = int(match.group("index"))
        if requested is None or index in requested:
            if index in checkpoints:
                raise FileNotFoundError(
                    f"Multiple final vmap{index} checkpoints found in {artifact_dir}"
                )
            checkpoints[index] = path
    if requested is not None:
        missing = sorted(requested - checkpoints.keys())
        if missing:
            raise FileNotFoundError(
                f"Missing final vmap checkpoint(s) {missing} in {artifact_dir}"
            )
    if not checkpoints:
        raise FileNotFoundError(f"No final vmap checkpoints found in {artifact_dir}")
    return sorted(checkpoints.items())


def is_self_play(left_model_id, right_model_id):
    """Only the exact same artifact checkpoint is self-play."""
    return left_model_id == right_model_id


def summarize_records(records):
    self_play = np.asarray(
        [record["mean_return"] for record in records if record["pair_type"] == "SP"],
        dtype=float,
    )
    cross_play = np.asarray(
        [record["mean_return"] for record in records if record["pair_type"] == "XP"],
        dtype=float,
    )
    sp = float(np.mean(self_play)) if self_play.size else float("nan")
    xp = float(np.mean(cross_play)) if cross_play.size else float("nan")
    return {
        "SP": sp,
        "XP": xp,
        "SP-XP_gap": sp - xp,
        "SP_pairs": int(self_play.size),
        "XP_pairs": int(cross_play.size),
    }


def build_model_matrix(records, models):
    indices = {model.identity: index for index, model in enumerate(models)}
    matrix = np.full((len(models), len(models)), np.nan, dtype=float)
    for record in records:
        row = indices.get(record["agent_0_model_id"])
        column = indices.get(record["agent_1_model_id"])
        if row is not None and column is not None:
            matrix[row, column] = float(record["mean_return"])
    return matrix


def build_algorithm_matrix(records, algorithms):
    matrix = np.full((len(algorithms), len(algorithms)), np.nan, dtype=float)
    for row, left_algorithm in enumerate(algorithms):
        for column, right_algorithm in enumerate(algorithms):
            values = [
                record["mean_return"]
                for record in records
                if record["agent_0_algorithm"].casefold() == left_algorithm.casefold()
                and record["agent_1_algorithm"].casefold() == right_algorithm.casefold()
            ]
            if values:
                matrix[row, column] = float(np.mean(values))
    return matrix


def select_matrix_views(models, algorithms):
    """Choose non-redundant W&B matrix views for the selected policies."""
    if len(algorithms) == 1:
        return ("models",)

    model_counts = {algorithm.casefold(): 0 for algorithm in algorithms}
    for model in models:
        model_counts[model.algorithm.casefold()] += 1
    if all(count == 1 for count in model_counts.values()):
        return ("algorithms",)
    return ("models", "algorithms")


def _json_safe(value):
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _atomic_write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _load_records(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return []
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid resume cache: {path}") from error


def _record_key(layout, left_model_id, right_model_id, args):
    return (
        layout,
        left_model_id,
        right_model_id,
        args.episodes,
        args.max_steps,
        args.seed,
        args.stochastic,
    )


def _cached_record_key(record):
    return (
        record["layout"],
        record["agent_0_model_id"],
        record["agent_1_model_id"],
        record["episodes"],
        record["max_steps"],
        record["evaluation_seed"],
        record["stochastic"],
    )


def _matrix_table(matrix, labels):
    data = []
    for label, values in zip(labels, matrix):
        data.append(
            [label]
            + [None if not np.isfinite(value) else float(value) for value in values]
        )
    return wandb.Table(columns=["agent_0 \\ agent_1", *labels], data=data)


def _algorithm_slug(algorithms):
    cleaned = []
    for algorithm in algorithms:
        slug = "".join(
            character.lower() if character.isalnum() else "-" for character in algorithm
        ).strip("-")
        cleaned.append(slug or "alg")
    return "+".join(cleaned)[:24].rstrip("-+")


def evaluation_run_name(args):
    """Build a concise W&B run name from algorithm and map only."""
    return f"xp-{_algorithm_slug(args.algorithms)}-{args.layout}"


def resolve_run_paths(args, timestamp, process_id):
    """Resolve one self-contained local directory for an evaluation run."""
    run_name = evaluation_run_name(args)
    output_dir = args.output_dir or (
        Path("saves/crossplay") / f"{run_name}-{timestamp}-p{process_id}"
    )
    artifact_dir = args.artifact_dir or output_dir / "artifacts"
    return output_dir, artifact_dir


def _jsonable_args(args):
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def write_reproducibility_bundle(output_dir, args, run_name, artifact_dir):
    """Save the command, resolved settings, and executable source snapshots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "command.txt").write_text(
        f"{shlex.join(sys.argv)}\n", encoding="utf-8"
    )
    _atomic_write_json(
        output_dir / "run_config.json",
        {
            **_jsonable_args(args),
            "wandb_run_name": run_name,
            "output_dir": str(output_dir),
            "artifact_dir": str(artifact_dir),
            "CUDA_VISIBLE_DEVICES": os.getenv("CUDA_VISIBLE_DEVICES"),
            "JAX_PLATFORMS": os.getenv("JAX_PLATFORMS"),
        },
    )

    source_dir = output_dir / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    script_dir = Path(__file__).resolve().parent
    for filename in (
        "eval_crossplay_overcooked_v3.py",
        "eval_crossplay_worker_overcooked_v3.py",
        "eval_wandb_crossplay_matrix_overcooked_v3.py",
    ):
        source_path = script_dir / filename
        if source_path.is_file():
            shutil.copy2(source_path, source_dir / filename)

    sweep_path = (
        script_dir.parents[1] / "experiment/sweeps/ippo_seedwise_crossplay.yaml"
    )
    if sweep_path.is_file():
        shutil.copy2(sweep_path, source_dir / sweep_path.name)


def add_run_outputs_to_artifact(result_artifact, output_dir, artifact_dir):
    """Upload results without duplicating checkpoints or W&B's local run cache."""
    excluded_roots = (artifact_dir.resolve(), (output_dir / "wandb").resolve())
    for path in sorted(output_dir.rglob("*")):
        resolved_path = path.resolve()
        if not path.is_file() or any(
            resolved_path.is_relative_to(root) for root in excluded_roots
        ):
            continue
        result_artifact.add_file(str(path), name=str(path.relative_to(output_dir)))


def _save_heatmap(matrix, labels, title, path):
    size = max(6, min(18, 0.55 * len(labels) + 3))
    figure, axis = plt.subplots(figsize=(size, size))
    image = axis.imshow(matrix, cmap="viridis")
    axis.set_title(title)
    axis.set_xlabel("agent_1 policy")
    axis.set_ylabel("agent_0 policy")
    axis.set_xticks(range(len(labels)), labels=labels, rotation=45, ha="right")
    axis.set_yticks(range(len(labels)), labels=labels)
    if len(labels) <= 12:
        for row in range(len(labels)):
            for column in range(len(labels)):
                value = matrix[row, column]
                if np.isfinite(value):
                    axis.text(
                        column,
                        row,
                        f"{value:.1f}",
                        ha="center",
                        va="center",
                        color="white" if value < np.nanmean(matrix) else "black",
                        fontsize=8,
                    )
    figure.colorbar(image, ax=axis, label="mean episode return")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _write_records_csv(path, records):
    if not records:
        return
    source_columns = list(records[0])
    if "layout" in source_columns:
        columns = ["map", *[column for column in source_columns if column != "layout"]]
        csv_records = [
            {
                "map": record["layout"],
                **{key: value for key, value in record.items() if key != "layout"},
            }
            for record in records
        ]
    else:
        columns = source_columns
        csv_records = records
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(csv_records)


def _validate_args(args):
    if args.episodes < 1:
        raise ValueError("--episodes must be at least 1")
    if args.max_steps < 1:
        raise ValueError("--max-steps must be at least 1")
    if len({algorithm.casefold() for algorithm in args.algorithms}) != len(
        args.algorithms
    ):
        raise ValueError("--algorithms must not contain duplicates")
    if args.gpus and len(set(args.gpus)) != len(args.gpus):
        raise ValueError("--gpus must not contain duplicates")
    if args.gpus and any("," in gpu for gpu in args.gpus):
        raise ValueError("Pass GPU IDs separated by spaces, for example --gpus 0 1")
    if args.workers_per_gpu < 1:
        raise ValueError("--workers-per-gpu must be at least 1")


def build_run_filters(layouts, seeds=None, run_state="finished"):
    """Push stable layout/seed/state selectors into the W&B API query."""
    filters = {"config.ENV_KWARGS.layout": {"$in": list(layouts)}}
    if seeds is not None:
        filters["config.SEED"] = {"$in": list(seeds)}
    if run_state != "all":
        filters["state"] = run_state
    return filters


def build_pair_task(layout, left, right, args, progress_index, total_pairs):
    return {
        "layout": layout,
        "pair_type": "SP" if is_self_play(left.identity, right.identity) else "XP",
        "agent_0_model_id": left.identity,
        "agent_1_model_id": right.identity,
        "agent_0_label": left.label,
        "agent_1_label": right.label,
        "agent_0_algorithm": left.algorithm,
        "agent_1_algorithm": right.algorithm,
        "agent_0_seed": left.seed,
        "agent_1_seed": right.seed,
        "agent_0_run": left.run_path,
        "agent_1_run": right.run_path,
        "agent_0_vmap": left.vmap_index,
        "agent_1_vmap": right.vmap_index,
        "agent_0_checkpoint": str(left.checkpoint),
        "agent_1_checkpoint": str(right.checkpoint),
        "agent_0_config": left.config,
        "agent_1_config": right.config,
        "progress_index": progress_index,
        "total_pairs": total_pairs,
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "evaluation_seed": args.seed,
        "stochastic": args.stochastic,
    }


def evaluate_pair_task(task, runtime_cache=None, params_cache=None):
    """Evaluate one serializable pair task inside a sequential or GPU worker."""
    runtime_cache = {} if runtime_cache is None else runtime_cache
    params_cache = {} if params_cache is None else params_cache
    pair_args = SimpleNamespace(
        layout=task["layout"],
        episodes=task["episodes"],
        max_steps=task["max_steps"],
        seed=task["evaluation_seed"],
        stochastic=task["stochastic"],
    )
    run_configs = (task["agent_0_config"], task["agent_1_config"])
    signature = evaluation_signature(run_configs, pair_args)
    runtime = runtime_cache.get(signature)
    if runtime is None:
        runtime = prepare_crossplay_runtime(run_configs, pair_args)
        runtime_cache[signature] = runtime

    checkpoint_paths = (
        task["agent_0_checkpoint"],
        task["agent_1_checkpoint"],
    )
    params = []
    for checkpoint_path in checkpoint_paths:
        if checkpoint_path not in params_cache:
            params_cache[checkpoint_path] = load_params(checkpoint_path)
        params.append(params_cache[checkpoint_path])

    result = evaluate_crossplay(
        checkpoint_paths,
        run_configs,
        pair_args,
        runtime=runtime,
        params=tuple(params),
        record_trajectory=False,
    )
    record_keys = (
        "layout",
        "pair_type",
        "agent_0_model_id",
        "agent_1_model_id",
        "agent_0_label",
        "agent_1_label",
        "agent_0_algorithm",
        "agent_1_algorithm",
        "agent_0_seed",
        "agent_1_seed",
        "agent_0_run",
        "agent_1_run",
        "agent_0_vmap",
        "agent_1_vmap",
        "episodes",
        "max_steps",
        "evaluation_seed",
        "stochastic",
    )
    return {
        **{key: task[key] for key in record_keys},
        "mean_return": float(np.mean(result["returns"])),
        "std_return": float(np.std(result["returns"])),
        "mean_episode_length": float(np.mean(result["lengths"])),
    }


def shard_tasks(tasks, worker_count):
    """Deterministically distribute ordered pairs across long-lived workers."""
    shards = [[] for _ in range(worker_count)]
    for index, task in enumerate(tasks):
        shards[index % worker_count].append(task)
    return shards


def run_gpu_workers(tasks, gpu_ids, output_dir, workers_per_gpu=8):
    worker_slots = [
        (gpu_id, instance_index)
        for gpu_id in gpu_ids
        for instance_index in range(workers_per_gpu)
    ]
    worker_count = min(len(worker_slots), len(tasks))
    worker_slots = worker_slots[:worker_count]
    shards = shard_tasks(tasks, worker_count)
    worker_script = Path(__file__).with_name("eval_crossplay_worker_overcooked_v3.py")
    processes = []
    result_paths = []
    for worker_index, ((gpu_id, instance_index), shard) in enumerate(
        zip(worker_slots, shards)
    ):
        task_path = output_dir / f"worker_{worker_index}_tasks.json"
        result_path = output_dir / f"worker_{worker_index}_results.json"
        _atomic_write_json(task_path, shard)
        result_path.unlink(missing_ok=True)
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = gpu_id
        environment["JAX_PLATFORMS"] = "cuda"
        environment.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        command = [
            sys.executable,
            "-u",
            str(worker_script),
            "--tasks",
            str(task_path),
            "--output",
            str(result_path),
            "--worker-label",
            f"gpu={gpu_id},instance={instance_index}",
        ]
        LOGGER.info(
            "Starting GPU worker %d on CUDA device %s instance %d with %d pair(s)",
            worker_index,
            gpu_id,
            instance_index,
            len(shard),
        )
        worker_label = f"gpu={gpu_id},instance={instance_index}"
        processes.append((worker_label, subprocess.Popen(command, env=environment)))
        result_paths.append(result_path)

    failures = []
    try:
        for worker_label, process in processes:
            return_code = process.wait()
            if return_code != 0:
                failures.append((worker_label, return_code))
    except KeyboardInterrupt:
        for _worker_label, process in processes:
            if process.poll() is None:
                process.terminate()
        for _worker_label, process in processes:
            process.wait()
        raise

    completed_records = []
    for result_path in result_paths:
        completed_records.extend(_load_records(result_path))
    return completed_records, failures


def recover_worker_records(output_dir, records):
    """Merge results left by workers when a previous parent run was interrupted."""
    recovered = {_cached_record_key(record): record for record in records}
    for result_path in sorted(output_dir.glob("worker_*_results.json")):
        for record in _load_records(result_path):
            recovered[_cached_record_key(record)] = record
    return list(recovered.values())


def main():
    load_project_env()
    args = parse_args()
    _validate_args(args)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    source_entity, source_project = split_project_path(args.source_project, args.entity)
    output_project_value = args.output_project or f"{source_project}-crossplay"
    output_entity, output_project = split_project_path(
        output_project_value,
        args.output_entity or source_entity,
    )
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = evaluation_run_name(args)
    output_dir, artifact_root = resolve_run_paths(args, timestamp, os.getpid())
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_root.mkdir(parents=True, exist_ok=True)
    write_reproducibility_bundle(output_dir, args, run_name, artifact_root)

    file_handler = logging.FileHandler(output_dir / "run.log", encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S"
        )
    )
    logging.getLogger().addHandler(file_handler)
    LOGGER.info("Run directory: %s", output_dir)
    records_path = output_dir / "pair_cache.json"

    api = wandb.Api()
    filters = build_run_filters([args.layout], args.seeds, args.run_state)
    LOGGER.info("Scanning W&B project %s/%s", source_entity, source_project)
    runs = api.runs(f"{source_entity}/{source_project}", filters=filters)
    candidates = discover_run_candidates(
        runs,
        args.algorithms,
        [args.layout],
        artifact_alias=args.artifact_alias,
        seeds=args.seeds,
        latest_per_seed=args.latest_per_seed,
    )
    LOGGER.info("Selected %d checkpoint artifact(s)", len(candidates))
    if not candidates:
        raise RuntimeError(
            "No matching runs with final checkpoint artifacts were found. Check "
            "--algorithms, --layout, --seeds, and --artifact-alias."
        )

    with wandb.init(
        entity=output_entity,
        project=output_project,
        mode=args.wandb_mode,
        name=run_name,
        dir=str(output_dir),
        group="crossplay-matrix",
        job_type="cross-play-matrix-evaluation",
        config={
            "source_project": f"{source_entity}/{source_project}",
            "algorithms": args.algorithms,
            "layout": args.layout,
            "training_seeds": args.seeds,
            "vmap_indices": args.vmap_indices,
            "episodes": args.episodes,
            "max_steps": args.max_steps,
            "evaluation_seed": args.seed,
            "stochastic": args.stochastic,
            "gpus": args.gpus,
            "workers_per_gpu": args.workers_per_gpu,
            "artifact_alias": args.artifact_alias,
            "latest_per_seed": args.latest_per_seed,
            "local_run_dir": str(output_dir),
            "local_artifact_dir": str(artifact_root),
        },
    ) as evaluation_run:
        models = []
        for candidate in candidates:
            source_run = candidate.run
            artifact = candidate.artifact
            artifact_ref = _artifact_reference(artifact, source_run)
            downloadable = artifact
            if args.wandb_mode == "online":
                downloadable = evaluation_run.use_artifact(artifact_ref)
            download_root = (
                artifact_root / source_run.id / artifact.name.replace(":", "-")
            )
            artifact_dir = Path(downloadable.download(root=str(download_root)))
            for vmap_index, checkpoint in resolve_vmap_checkpoints(
                artifact_dir, args.vmap_indices
            ):
                models.append(
                    PolicyModel(
                        algorithm=candidate.algorithm,
                        layout=candidate.layout,
                        seed=candidate.seed,
                        run_id=source_run.id,
                        run_path=_run_path(source_run),
                        vmap_index=vmap_index,
                        checkpoint=checkpoint,
                        config=candidate.config,
                        artifact_name=artifact.name,
                    )
                )
                LOGGER.info(
                    "Model %-12s layout=%s seed=%s run=%s vmap=%d",
                    candidate.algorithm,
                    candidate.layout,
                    candidate.seed,
                    source_run.id,
                    vmap_index,
                )

        layout_models = [model for model in models if model.layout == args.layout]
        available = {model.algorithm.casefold() for model in layout_models}
        missing = [
            algorithm
            for algorithm in args.algorithms
            if algorithm.casefold() not in available
        ]
        if missing:
            raise RuntimeError(
                f"Layout {args.layout} has no checkpoint for algorithm(s): {missing}"
            )
        if len(layout_models) < 2:
            raise RuntimeError(
                f"Layout {args.layout} needs at least two models to compute XP"
            )

        model_manifest = [
            {
                "model_id": model.identity,
                "label": model.label,
                "algorithm": model.algorithm,
                "layout": model.layout,
                "training_seed": model.seed,
                "run": model.run_path,
                "vmap_index": model.vmap_index,
                "artifact": model.artifact_name,
            }
            for model in models
        ]
        _atomic_write_json(output_dir / "models.json", model_manifest)
        evaluation_run.config.update(
            {"selected_models": model_manifest}, allow_val_change=True
        )

        records = recover_worker_records(output_dir, _load_records(records_path))
        _atomic_write_json(records_path, records)
        cached = {_cached_record_key(record): record for record in records}
        total_pairs = len(layout_models) ** 2
        complete = 0
        pending_tasks = []

        for left in layout_models:
            for right in layout_models:
                complete += 1
                key = _record_key(args.layout, left.identity, right.identity, args)
                if key in cached:
                    LOGGER.info(
                        "[%d/%d] cached %s %s x %s",
                        complete,
                        total_pairs,
                        cached[key]["pair_type"],
                        left.label,
                        right.label,
                    )
                    continue
                pending_tasks.append(
                    build_pair_task(
                        args.layout,
                        left,
                        right,
                        args,
                        progress_index=complete,
                        total_pairs=total_pairs,
                    )
                )

        new_records = []
        failures = []
        if args.gpus and pending_tasks:
            new_records, failures = run_gpu_workers(
                pending_tasks,
                args.gpus,
                output_dir,
                workers_per_gpu=args.workers_per_gpu,
            )
        else:
            runtime_cache = {}
            params_cache = {}
            for task in pending_tasks:
                LOGGER.info(
                    "[%d/%d] evaluating %s %s x %s",
                    task["progress_index"],
                    task["total_pairs"],
                    task["pair_type"],
                    task["agent_0_label"],
                    task["agent_1_label"],
                )
                record = evaluate_pair_task(task, runtime_cache, params_cache)
                new_records.append(record)
                LOGGER.info(
                    "[%d/%d] result mean=%.2f std=%.2f",
                    task["progress_index"],
                    task["total_pairs"],
                    record["mean_return"],
                    record["std_return"],
                )

        for record in new_records:
            key = _cached_record_key(record)
            if key in cached:
                continue
            records.append(record)
            cached[key] = record
            _atomic_write_json(records_path, records)
            wandb.log(
                {
                    "progress/completed_pairs": len(cached),
                    "pair/mean_return": record["mean_return"],
                    "pair/is_self_play": int(record["pair_type"] == "SP"),
                }
            )

        if failures:
            details = ", ".join(
                f"{worker_label}: exit {return_code}"
                for worker_label, return_code in failures
            )
            raise RuntimeError(
                f"Cross-play GPU worker failure(s): {details}. Completed pair "
                f"results were preserved in {records_path}; rerun with the same "
                "--output-dir to resume."
            )

        active_model_ids = {model.identity for model in models}
        active_records_by_key = {
            _cached_record_key(record): record
            for record in records
            if record["layout"] == args.layout
            and record["agent_0_model_id"] in active_model_ids
            and record["agent_1_model_id"] in active_model_ids
            and _cached_record_key(record)[3:]
            == (
                args.episodes,
                args.max_steps,
                args.seed,
                args.stochastic,
            )
        }
        active_records = list(active_records_by_key.values())
        _atomic_write_json(output_dir / "pair_results.json", active_records)
        summary = summarize_records(active_records)
        scalar_metrics = {
            "SP": summary["SP"],
            "XP": summary["XP"],
            "SP-XP_gap": summary["SP-XP_gap"],
            "counts/SP_pairs": summary["SP_pairs"],
            "counts/XP_pairs": summary["XP_pairs"],
        }
        matrix_views = select_matrix_views(layout_models, args.algorithms)
        model_heatmap = output_dir / f"{args.layout}_model_matrix.png"
        algorithm_heatmap = output_dir / f"{args.layout}_algorithm_matrix.png"
        wandb_payload = {}
        if "models" in matrix_views:
            model_matrix = build_model_matrix(active_records, layout_models)
            model_labels = [model.label for model in layout_models]
            _save_heatmap(
                model_matrix,
                model_labels,
                f"{args.layout}: checkpoint payoff matrix",
                model_heatmap,
            )
            wandb_payload["matrices/models_table"] = _matrix_table(
                model_matrix, model_labels
            )
            wandb_payload["matrices/models"] = wandb.Image(str(model_heatmap))
        else:
            model_heatmap.unlink(missing_ok=True)

        if "algorithms" in matrix_views:
            algorithm_matrix = build_algorithm_matrix(active_records, args.algorithms)
            _save_heatmap(
                algorithm_matrix,
                args.algorithms,
                f"{args.layout}: algorithm payoff matrix",
                algorithm_heatmap,
            )
            wandb_payload["matrices/algorithms_table"] = _matrix_table(
                algorithm_matrix, args.algorithms
            )
            wandb_payload["matrices/algorithms"] = wandb.Image(str(algorithm_heatmap))
        else:
            algorithm_heatmap.unlink(missing_ok=True)

        pair_columns = list(active_records[0])
        wandb_payload["results/pairs"] = wandb.Table(
            columns=pair_columns,
            data=[
                [record[column] for column in pair_columns] for record in active_records
            ],
        )
        wandb.log({**scalar_metrics, **wandb_payload})
        evaluation_run.summary.update(scalar_metrics)

        summaries_path = output_dir / "summary.json"
        _atomic_write_json(
            summaries_path,
            {
                "map": args.layout,
                "overall": {key: _json_safe(value) for key, value in summary.items()},
            },
        )
        _write_records_csv(output_dir / "pair_results.csv", active_records)
        result_artifact = wandb.Artifact(
            f"crossplay-matrix-{evaluation_run.id}",
            type="crossplay-evaluation",
            metadata={key: _json_safe(value) for key, value in summary.items()},
        )
        LOGGER.info(
            "Completed %d ordered pairs | SP=%.2f XP=%.2f SP-XP=%.2f",
            len(active_records),
            summary["SP"],
            summary["XP"],
            summary["SP-XP_gap"],
        )
        LOGGER.info("Results saved to %s", output_dir)
        file_handler.flush()
        add_run_outputs_to_artifact(result_artifact, output_dir, artifact_root)
        evaluation_run.log_artifact(result_artifact, aliases=["latest"])


if __name__ == "__main__":
    main()
