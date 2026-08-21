"""W&B-agent stages connecting the CooT population, training, and evaluation sweeps."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence

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


def _paths(layout: str) -> dict[str, Path | str]:
    return {
        "candidate_glob": (
            "saves/coot_population/"
            f"{layout}_rnn_hsp_population_hsp_*_candidate*_seed0/*candidate*.json"
        ),
        "response_glob": f"saves/coot_responses/{layout}_rnn_*/response_job*.json",
        "raw_catalog": Path(f"manifests/coot/catalogs/{layout}_raw.json"),
        "scored_catalog": Path(f"manifests/coot/catalogs/{layout}_scored.json"),
        "candidate_jobs": Path(f"manifests/coot/response_candidates/{layout}.json"),
        "selected_jobs": Path(f"manifests/coot/response_jobs_hsp_only/{layout}.json"),
        "pair_manifest": Path(f"manifests/coot/train/{layout}.json"),
        "score_cache": Path(f"saves/coot_population_scores/{layout}"),
        "dataset_dir": Path(f"datasets/coot/{layout}"),
    }


def _prepare_candidates(layout: str) -> list[Path]:
    paths = _paths(layout)
    raw_catalog = paths["raw_catalog"]
    candidate_jobs = paths["candidate_jobs"]
    assert isinstance(raw_catalog, Path)
    assert isinstance(candidate_jobs, Path)
    raw_catalog.parent.mkdir(parents=True, exist_ok=True)
    candidate_jobs.parent.mkdir(parents=True, exist_ok=True)

    score_population(
        [
            "--candidate-result",
            str(paths["candidate_glob"]),
            "--layout",
            layout,
            "--merge-only",
            "--output",
            str(raw_catalog),
            "--overwrite",
        ]
    )
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


def _score_and_select(layout: str, episodes: int) -> list[Path]:
    paths = _paths(layout)
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

    score_population(
        [
            "--catalog",
            str(raw_catalog),
            "--response-result",
            str(paths["response_glob"]),
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
    )
    build_manifest(
        [
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
    )
    return [scored_catalog, selected_jobs]


def _build_dataset(layout: str) -> list[Path]:
    paths = _paths(layout)
    scored_catalog = paths["scored_catalog"]
    pair_manifest = paths["pair_manifest"]
    dataset_dir = paths["dataset_dir"]
    assert isinstance(scored_catalog, Path)
    assert isinstance(pair_manifest, Path)
    assert isinstance(dataset_dir, Path)
    pair_manifest.parent.mkdir(parents=True, exist_ok=True)
    dataset_dir.parent.mkdir(parents=True, exist_ok=True)

    build_manifest(
        [
            "build-pairs",
            "--hsp-catalog",
            str(scored_catalog),
            "--layout",
            layout,
            "--allow-hsp-only",
            "--response-results",
            str(paths["response_glob"]),
            "--verify-checkpoints",
            "--output",
            str(pair_manifest),
            "--overwrite",
        ]
    )

    metadata = dataset_dir / "metadata.json"
    if not metadata.is_file():
        collect_dataset(
            [
                "--manifest",
                str(pair_manifest),
                "--output-root",
                "datasets/coot",
                "--layout",
                layout,
                "--overwrite",
            ]
        )
    return [pair_manifest, metadata, dataset_dir / "resolved_manifest.json"]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        required=True,
        choices=("prepare_candidates", "score_and_select", "build_dataset"),
    )
    parser.add_argument("--layout", required=True)
    parser.add_argument("--score-episodes", type=int, default=50)
    parser.add_argument("--entity", default="cilab-overcooked")
    parser.add_argument("--project", default="overcooked-v3-coot-pipeline")
    parser.add_argument("--wandb-mode", default=os.getenv("WANDB_MODE", "online"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if Path.cwd().resolve() != REPO_ROOT:
        raise ValueError(f"Run the W&B agent from repository root: {REPO_ROOT}")
    if args.score_episodes < 1:
        raise ValueError("--score-episodes must be positive")

    init_kwargs = {
        "config": vars(args),
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
        if args.stage == "prepare_candidates":
            outputs = _prepare_candidates(args.layout)
        elif args.stage == "score_and_select":
            outputs = _score_and_select(args.layout, args.score_episodes)
        else:
            outputs = _build_dataset(args.layout)

        missing = [str(path) for path in outputs if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Pipeline stage outputs are missing: {missing}")
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
