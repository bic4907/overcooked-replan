"""W&B-agent stages connecting the CooT population, training, and evaluation sweeps."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import wandb

from jaxmarl._wandb import require_sweep_target

try:
    from .build_population_manifest import main as build_manifest
    from .collect_overcooked_v3 import main as collect_dataset
    from .score_hsp_population_overcooked_v3 import main as score_population
except ImportError:  # Direct execution: python baselines/CooT/<script>.py
    from build_population_manifest import main as build_manifest
    from collect_overcooked_v3 import main as collect_dataset
    from score_hsp_population_overcooked_v3 import main as score_population


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_NAMESPACE = "default"


def _namespace(value: str) -> str:
    normalized = str(value).strip()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", normalized):
        raise ValueError(
            "Pipeline namespace must contain only letters, numbers, '.', '_', or '-'"
        )
    return normalized


def _paths(
    layout: str,
    namespace: str,
) -> dict[str, Path | str]:
    namespace = _namespace(namespace)
    manifest_root = Path("manifests/coot")
    score_root = Path("saves/coot_population_scores")
    dataset_root = Path("datasets/coot")
    if namespace != DEFAULT_NAMESPACE:
        manifest_root /= namespace
        score_root /= namespace
        dataset_root /= namespace
    return {
        "raw_catalog": manifest_root / "catalogs" / f"{layout}_raw.json",
        "scored_catalog": manifest_root / "catalogs" / f"{layout}_scored.json",
        "candidate_jobs": manifest_root / "response_candidates" / f"{layout}.json",
        "selected_jobs": (
            manifest_root / "response_jobs_hsp_only" / f"{layout}.json"
        ),
        "pair_manifest": manifest_root / "train" / f"{layout}.json",
        "score_cache": score_root / layout,
        "dataset_dir": dataset_root / layout,
    }


def _load_response_sweep_map(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Response sweep manifest not found: {resolved}")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or not payload:
        raise ValueError("Response sweep manifest must be a non-empty JSON object")
    result = {str(layout): str(sweep_ref) for layout, sweep_ref in payload.items()}
    for layout, sweep_ref in result.items():
        if layout != "*" and not layout.strip():
            raise ValueError("Response sweep manifest contains an empty layout")
        if len(sweep_ref.split("/")) != 3:
            raise ValueError(
                f"Invalid response sweep reference for {layout!r}: {sweep_ref!r}"
            )
    return result


def _run_layout(run: Any) -> str | None:
    config = run.config or {}
    value = config.get("scenario") or config.get("layout")
    return str(value) if value else None


def _sidecar_value(payload: Mapping[str, Any], key: str) -> Any:
    if payload.get(key) is not None:
        return payload[key]
    for nested_key in ("resolved_job", "original_job"):
        nested = payload.get(nested_key)
        if isinstance(nested, Mapping) and nested.get(key) is not None:
            return nested[key]
    return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _verified_population_results(
    sweep_ref: str,
    *,
    layout: str,
    population_root: Path,
) -> list[Path]:
    """Select one local candidate sidecar for every run in one finished sweep."""

    sweep = wandb.Api(timeout=90).sweep(sweep_ref)
    if str(sweep.state).upper() != "FINISHED":
        raise RuntimeError(
            f"Upstream population sweep {sweep_ref} is {sweep.state}, not FINISHED"
        )
    runs = [run for run in sweep.runs if _run_layout(run) == layout]
    if not runs:
        raise RuntimeError(f"Population sweep {sweep_ref} has no runs for {layout}")
    bad_runs = [run for run in runs if str(run.state).lower() != "finished"]
    if bad_runs:
        preview = ", ".join(
            f"{run.id} ({run.state})" for run in bad_runs[:20]
        )
        raise RuntimeError(
            f"Population sweep {sweep_ref} has {len(bad_runs)} non-finished "
            f"{layout} run(s): {preview}"
        )

    run_ids = {str(run.id) for run in runs}
    matches: dict[str, list[Path]] = {run_id: [] for run_id in run_ids}
    root = population_root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Population result root not found: {root}")
    for result_path in root.rglob("*candidate_result*.json"):
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, Mapping):
            continue
        run_id = str(payload.get("immutable_run_id") or "")
        if run_id not in run_ids:
            continue
        if str(payload.get("layout") or payload.get("scenario") or "") != layout:
            raise ValueError(
                f"Population result layout mismatch in {result_path}; expected "
                f"{layout!r}"
            )
        partner = payload.get("partner")
        if not isinstance(partner, Mapping):
            raise ValueError(f"Population result lacks partner policies: {result_path}")
        for skill in ("mid", "final"):
            policy = partner.get(skill)
            if not isinstance(policy, Mapping) or not policy.get("checkpoint"):
                raise ValueError(
                    f"Population result lacks partner.{skill}: {result_path}"
                )
            checkpoint = Path(str(policy["checkpoint"])).expanduser()
            if not checkpoint.is_absolute():
                checkpoint = (result_path.parent / checkpoint).resolve()
            if not checkpoint.is_file():
                raise FileNotFoundError(
                    f"Population {skill} checkpoint referenced by {result_path} "
                    f"is missing: {checkpoint}"
                )
        matches[run_id].append(result_path.resolve())

    missing = sorted(run_id for run_id, paths in matches.items() if not paths)
    duplicates = {
        run_id: paths for run_id, paths in matches.items() if len(paths) > 1
    }
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} local population sidecar(s) for {sweep_ref} / "
            f"{layout}: {', '.join(missing[:20])}"
        )
    if duplicates:
        preview = ", ".join(
            f"{run_id}={len(paths)}" for run_id, paths in list(duplicates.items())[:20]
        )
        raise ValueError(
            f"Duplicate local population sidecars for {sweep_ref} / {layout}: "
            f"{preview}"
        )
    return [matches[run_id][0] for run_id in sorted(run_ids)]


def _verified_response_results(
    sweep_ref: str,
    *,
    layout: str,
    expected_stage: str,
    response_root: Path,
) -> list[Path]:
    """Select only local sidecars produced by one fully successful W&B sweep."""

    sweep = wandb.Api(timeout=90).sweep(sweep_ref)
    if str(sweep.state).upper() != "FINISHED":
        raise RuntimeError(
            f"Upstream response sweep {sweep_ref} is {sweep.state}, not FINISHED"
        )
    runs = [run for run in sweep.runs if _run_layout(run) == layout]
    if not runs:
        raise RuntimeError(
            f"Upstream response sweep {sweep_ref} has no runs for {layout}"
        )
    bad_runs = [run for run in runs if str(run.state).lower() != "finished"]
    if bad_runs:
        preview = ", ".join(
            f"{run.id} ({run.state})" for run in bad_runs[:20]
        )
        raise RuntimeError(
            f"Upstream response sweep {sweep_ref} has {len(bad_runs)} "
            f"non-finished {layout} run(s): {preview}"
        )

    run_ids = {str(run.id) for run in runs}
    matches: dict[str, list[Path]] = {run_id: [] for run_id in run_ids}
    root = response_root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Response result root not found: {root}")
    for result_path in root.rglob("response_job*.json"):
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, Mapping):
            continue
        run_id = str(payload.get("run_id") or "")
        if run_id not in run_ids:
            continue
        if str(payload.get("status", "")).lower() != "completed":
            raise ValueError(f"Response result is not completed: {result_path}")
        result_layout = str(_sidecar_value(payload, "layout") or "")
        result_stage = str(
            payload.get("response_job_stage")
            or _sidecar_value(payload, "stage")
            or ""
        )
        if result_layout != layout or result_stage != expected_stage:
            raise ValueError(
                f"Response result provenance mismatch in {result_path}: "
                f"layout={result_layout!r}, stage={result_stage!r}; expected "
                f"layout={layout!r}, stage={expected_stage!r}"
            )
        checkpoint_value = payload.get("best_response_checkpoint")
        if not checkpoint_value:
            raise ValueError(
                f"Completed response result has no checkpoint: {result_path}"
            )
        checkpoint = Path(str(checkpoint_value)).expanduser()
        if not checkpoint.is_absolute():
            checkpoint = (result_path.parent / checkpoint).resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"Response checkpoint referenced by {result_path} is missing: "
                f"{checkpoint}"
            )
        partner_value = _sidecar_value(payload, "partner_checkpoint")
        partner_digest = _sidecar_value(payload, "partner_sha256")
        if not partner_value or not partner_digest:
            raise ValueError(
                f"Response result lacks partner checkpoint provenance: {result_path}"
            )
        partner_checkpoint = Path(str(partner_value)).expanduser()
        if not partner_checkpoint.is_absolute():
            partner_checkpoint = (result_path.parent / partner_checkpoint).resolve()
        if not partner_checkpoint.is_file():
            raise FileNotFoundError(
                f"Response partner checkpoint referenced by {result_path} is "
                f"missing: {partner_checkpoint}"
            )
        actual_digest = _sha256(partner_checkpoint)
        if actual_digest != str(partner_digest):
            raise ValueError(
                f"Response partner checkpoint changed after training: "
                f"{partner_checkpoint}; recorded sha256={partner_digest}, "
                f"current sha256={actual_digest}"
            )
        matches[run_id].append(result_path.resolve())

    missing = sorted(run_id for run_id, paths in matches.items() if not paths)
    duplicates = {
        run_id: paths for run_id, paths in matches.items() if len(paths) > 1
    }
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} local response sidecar(s) for {sweep_ref} / "
            f"{layout}: {', '.join(missing[:20])}"
        )
    if duplicates:
        preview = ", ".join(
            f"{run_id}={len(paths)}" for run_id, paths in list(duplicates.items())[:20]
        )
        raise ValueError(
            f"Duplicate local response sidecars for {sweep_ref} / {layout}: {preview}"
        )
    return [matches[run_id][0] for run_id in sorted(run_ids)]


def _stamp_outputs(
    outputs: Sequence[Path],
    *,
    namespace: str,
    stage: str,
    run: Any,
    upstream_population_sweep: str | None,
    upstream_sweep: str | None,
) -> None:
    provenance = {
        "namespace": namespace,
        "stage": stage,
        "wandb_run_id": str(run.id),
        "wandb_sweep_id": os.getenv("WANDB_SWEEP_ID"),
        "upstream_population_sweep": upstream_population_sweep,
        "upstream_response_sweep": upstream_sweep,
    }
    for path in outputs:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Pipeline output must be a JSON object: {path}")
        payload["pipeline_provenance"] = provenance
        temporary = path.with_suffix(f"{path.suffix}.tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)


def _prepare_candidates(
    layout: str,
    namespace: str,
    candidate_results: Sequence[Path],
) -> list[Path]:
    paths = _paths(layout, namespace)
    raw_catalog = paths["raw_catalog"]
    candidate_jobs = paths["candidate_jobs"]
    assert isinstance(raw_catalog, Path)
    assert isinstance(candidate_jobs, Path)
    raw_catalog.parent.mkdir(parents=True, exist_ok=True)
    candidate_jobs.parent.mkdir(parents=True, exist_ok=True)

    score_args = [
        "--layout",
        layout,
        "--merge-only",
        "--output",
        str(raw_catalog),
        "--overwrite",
    ]
    for result in candidate_results:
        score_args.extend(["--candidate-result", str(result)])
    score_population(score_args)
    build_manifest(
        [
            "response-jobs",
            "--hsp-catalog",
            str(raw_catalog),
            "--layout",
            layout,
            "--all-hsp-candidates",
            "--hsp-skill",
            "final",
            "--verify-checkpoints",
            "--output",
            str(candidate_jobs),
            "--overwrite",
        ]
    )
    return [raw_catalog, candidate_jobs]


def _score_and_select(
    layout: str,
    episodes: int,
    *,
    allow_low_return_fill: bool,
    namespace: str,
    response_results: Sequence[Path],
) -> list[Path]:
    paths = _paths(layout, namespace)
    raw_catalog = paths["raw_catalog"]
    scored_catalog = paths["scored_catalog"]
    selected_jobs = paths["selected_jobs"]
    score_cache = paths["score_cache"]
    assert isinstance(raw_catalog, Path)
    assert isinstance(scored_catalog, Path)
    assert isinstance(selected_jobs, Path)
    assert isinstance(score_cache, Path)
    scored_catalog.parent.mkdir(parents=True, exist_ok=True)
    selected_jobs.parent.mkdir(parents=True, exist_ok=True)
    score_cache.mkdir(parents=True, exist_ok=True)

    score_args = [
        "--catalog",
        str(raw_catalog),
        "--layout",
        layout,
        "--episodes",
        str(episodes),
        "--cache-dir",
        str(score_cache),
        "--output",
        str(scored_catalog),
        "--overwrite",
    ]
    for result in response_results:
        score_args.extend(["--response-result", str(result)])
    score_population(score_args)
    manifest_args = [
        "response-jobs",
        "--hsp-catalog",
        str(scored_catalog),
        "--layout",
        layout,
        "--allow-hsp-only",
        "--hsp-skill",
        "mid",
        "--verify-checkpoints",
        "--output",
        str(selected_jobs),
        "--overwrite",
    ]
    if allow_low_return_fill:
        manifest_args.append("--allow-low-return-fill")
    build_manifest(manifest_args)
    return [scored_catalog, selected_jobs]


def _build_dataset(
    layout: str,
    *,
    allow_low_return_fill: bool,
    namespace: str,
    response_results: Sequence[Path],
) -> list[Path]:
    paths = _paths(layout, namespace)
    scored_catalog = paths["scored_catalog"]
    pair_manifest = paths["pair_manifest"]
    dataset_dir = paths["dataset_dir"]
    assert isinstance(scored_catalog, Path)
    assert isinstance(pair_manifest, Path)
    assert isinstance(dataset_dir, Path)
    pair_manifest.parent.mkdir(parents=True, exist_ok=True)
    dataset_dir.parent.mkdir(parents=True, exist_ok=True)

    manifest_args = [
        "build-pairs",
        "--hsp-catalog",
        str(scored_catalog),
        "--layout",
        layout,
        "--allow-hsp-only",
        "--verify-checkpoints",
        "--output",
        str(pair_manifest),
        "--overwrite",
    ]
    if allow_low_return_fill:
        manifest_args.append("--allow-low-return-fill")
    for result in response_results:
        manifest_args.extend(["--response-results", str(result)])
    build_manifest(manifest_args)

    metadata = dataset_dir / "metadata.json"
    collect_dataset(
        [
            "--manifest",
            str(pair_manifest),
            "--output-root",
            str(dataset_dir.parent),
            "--layout",
            layout,
            "--overwrite",
        ]
    )
    return [pair_manifest, metadata, dataset_dir / "resolved_manifest.json"]


def _boolean(value: str) -> bool:
    normalized = str(value).lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        required=True,
        choices=("prepare_candidates", "score_and_select", "build_dataset"),
    )
    parser.add_argument("--layout", required=True)
    parser.add_argument("--score-episodes", type=int, default=50)
    parser.add_argument(
        "--allow-low-return-fill",
        type=_boolean,
        default=False,
        help=(
            "Allow the explicit HSP-only proxy to fill a short return-eligible "
            "population with diversity-selected low-return candidates."
        ),
    )
    parser.add_argument("--entity", default="cilab-overcooked")
    parser.add_argument("--project", default="overcooked-v3-coot-pipeline")
    parser.add_argument("--wandb-mode", default=os.getenv("WANDB_MODE", "online"))
    parser.add_argument(
        "--namespace",
        default=os.getenv("COOT_PIPELINE_NAMESPACE", DEFAULT_NAMESPACE),
    )
    parser.add_argument(
        "--population-root",
        type=Path,
        default=Path(os.getenv("COOT_POPULATION_SAVES_DIR", "saves/coot_population")),
    )
    parser.add_argument("--population-sweep-manifest", type=Path)
    parser.add_argument("--response-sweep-manifest", type=Path)
    parser.add_argument(
        "--response-result-root",
        type=Path,
        default=Path(os.getenv("COOT_RESPONSE_RESULT_ROOT", "saves/coot_responses")),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if Path.cwd().resolve() != REPO_ROOT:
        raise ValueError(f"Run the W&B agent from repository root: {REPO_ROOT}")
    if args.score_episodes < 1:
        raise ValueError("--score-episodes must be positive")
    namespace = _namespace(args.namespace)
    population_sweeps = _load_response_sweep_map(args.population_sweep_manifest)
    response_sweeps = _load_response_sweep_map(args.response_sweep_manifest)
    if args.stage == "prepare_candidates" and not population_sweeps:
        raise ValueError(
            "prepare_candidates requires --population-sweep-manifest for exact "
            "upstream provenance"
        )
    if args.stage != "prepare_candidates" and not response_sweeps:
        raise ValueError(
            f"{args.stage} requires --response-sweep-manifest for exact upstream "
            "provenance"
        )

    run_config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    init_kwargs = {
        "config": run_config,
        "group": f"coot-pipeline-{args.stage}",
        "name": f"coot-pipeline-{args.stage}-{args.layout}",
        "tags": ["CooT", "Pipeline", args.stage, "OvercookedV3"],
        "mode": args.wandb_mode,
    }
    if not os.getenv("WANDB_SWEEP_ID"):
        init_kwargs.update({"entity": args.entity, "project": args.project})

    with wandb.init(**init_kwargs) as run:
        require_sweep_target(
            run,
            {"ENTITY": args.entity, "PROJECT": args.project},
        )
        upstream_sweep = None
        upstream_population_sweep = None
        candidate_results: list[Path] = []
        response_results: list[Path] = []
        if args.stage == "prepare_candidates":
            upstream_population_sweep = population_sweeps.get(
                args.layout, population_sweeps.get("*")
            )
            if not upstream_population_sweep:
                raise ValueError(
                    f"Population sweep manifest has no entry for {args.layout!r}"
                )
            candidate_results = _verified_population_results(
                upstream_population_sweep,
                layout=args.layout,
                population_root=args.population_root,
            )
        else:
            upstream_sweep = response_sweeps.get(
                args.layout, response_sweeps.get("*")
            )
            if not upstream_sweep:
                raise ValueError(
                    f"Response sweep manifest has no entry for {args.layout!r}"
                )
            response_results = _verified_response_results(
                upstream_sweep,
                layout=args.layout,
                expected_stage=(
                    "candidates" if args.stage == "score_and_select" else "hsp_only"
                ),
                response_root=args.response_result_root,
            )

        if args.stage == "prepare_candidates":
            outputs = _prepare_candidates(
                args.layout,
                namespace,
                candidate_results,
            )
        elif args.stage == "score_and_select":
            outputs = _score_and_select(
                args.layout,
                args.score_episodes,
                allow_low_return_fill=args.allow_low_return_fill,
                namespace=namespace,
                response_results=response_results,
            )
        else:
            outputs = _build_dataset(
                args.layout,
                allow_low_return_fill=args.allow_low_return_fill,
                namespace=namespace,
                response_results=response_results,
            )

        missing = [str(path) for path in outputs if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Pipeline stage outputs are missing: {missing}")
        _stamp_outputs(
            outputs,
            namespace=namespace,
            stage=args.stage,
            run=run,
            upstream_population_sweep=upstream_population_sweep,
            upstream_sweep=upstream_sweep,
        )
        artifact = wandb.Artifact(
            name=f"coot-{args.stage}-{args.layout}-{run.id}",
            type="coot-pipeline-stage",
            metadata={"stage": args.stage, "layout": args.layout},
        )
        for path in outputs:
            artifact.add_file(str(path), name=path.name)
        run.log_artifact(artifact, aliases=["completed", args.layout])
        run.log({"pipeline/completed": 1})
        run.summary["pipeline/stage"] = args.stage
        run.summary["pipeline/layout"] = args.layout


if __name__ == "__main__":
    main()
